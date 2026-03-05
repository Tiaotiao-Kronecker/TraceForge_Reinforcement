import os
import torch
import numpy as np
from loguru import logger

from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image


def align_depth_scale(pred_depth, known_depth):
    """
    Align the scale of predicted depth to known depth using median scaling.
    Input:
        pred_depth: (H, W), torch tensor
        known_depth: (H, W), torch tensor
    Return:
        aligned_depth: (H, W), torch tensor
    """
    valid_mask = (known_depth > 0) & (pred_depth > 0)
    scale = np.median(known_depth[valid_mask]) / np.median(pred_depth[valid_mask])
    return scale

def align_video_depth_scale(pred_depth, known_depth):
    """
    Align the scale of predicted depth to known depth using median scaling.
    Input:
        pred_depth: (T, H, W), torch tensor
        known_depth: (T, H, W), torch tensor
    Return:
        aligned_depth: (T, H, W), torch tensor
    """
    scales = []
    for t in range(pred_depth.shape[0]):
        scales.append(
            align_depth_scale(pred_depth[t], known_depth[t])
        )
    scale = np.array(scales).mean()
    aligned_depth = pred_depth * scale
    return aligned_depth, scale


class BaseVideoDepthPoseWrapper:
    def __init__(self, args):
        self.args = args
        self.device = args.device

    def __call__(self, video_tensor, known_depth=None, stationary_camera=False, replace_with_known_depth=True):
        """
        Input:
            video_tensor: (T, 3, H, W), torch tensor, range [0, 1]
            known_depth: (T, H, W), torch tensor, range [0, inf), if provided, will be used as depth input
            stationary_camera: bool, if True, indicates the camera is stationary
            replace_with_known_depth: bool, only used when known_depth is provided. If True, return known_depth as the depth. If False, rescale the estimated depth to the scale of known depth
        Return:
            video_ten: (T, 3, H, W), torch tensor, range [0, 1], processed (resized) video tensor, will be used in 3D tracking as well
            depth_npy: (T, H, W), numpy array
            depth_conf: (T, H, W), numpy array
            extrs_npy: (T, 4, 4), numpy array, in camera-to-world format
            intrs_npy: (T, 3, 3), numpy array
        """
        raise NotImplementedError


class VGGT4Wrapper(BaseVideoDepthPoseWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.load_model()

    def load_model(self, checkpoint_path="Yuxihenry/SpatialTrackerV2_Front"):
        import time
        import os
        
        max_retries = 5
        retry_delay = 2  # 初始延迟2秒
        
        # 检查缓存是否存在
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_name = checkpoint_path.replace("/", "--")
        model_cache_path = os.path.join(cache_dir, f"models--{model_cache_name}")
        
        # 如果缓存存在，尝试使用local_files_only=True强制只使用本地文件
        use_local_only = os.path.exists(model_cache_path)
        
        if use_local_only:
            logger.info(f"检测到模型缓存，尝试仅使用本地文件加载: {checkpoint_path}")
        
        # 尝试加载模型（带重试）
        for attempt in range(max_retries):
            try:
                if use_local_only:
                    # 优先尝试仅使用本地文件
                    try:
                        model = VGGT4Track.from_pretrained(checkpoint_path, local_files_only=True)
                        logger.info(f"✅ 成功从本地缓存加载模型（无需网络）")
                    except Exception as local_error:
                        # 如果local_files_only失败，回退到允许网络（但会优先使用缓存）
                        logger.warning(f"仅使用本地文件失败，尝试允许网络连接: {local_error}")
                        model = VGGT4Track.from_pretrained(checkpoint_path, local_files_only=False)
                else:
                    # 缓存不存在，需要从网络下载
                    model = VGGT4Track.from_pretrained(checkpoint_path)
                
                logger.debug(f"load vggt4 from {checkpoint_path}")
                model = model.eval()
                model = model.to(self.device)
                return model
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 指数退避: 2, 4, 8, 16, 32秒
                    logger.warning(f"加载模型失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    logger.warning(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"加载模型失败，已重试 {max_retries} 次: {e}")
                    raise

    def __call__(self, video_tensor, known_depth=None, stationary_camera=False, replace_with_known_depth=True):
        video_tensor_processed = preprocess_image(video_tensor)[None]  # (1, T, 3, H, W)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                predictions = self.model(video_tensor_processed.to(self.device))
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = (
                    predictions["points_map"][..., 2],
                    predictions["unc_metric"],
                )

        depth_npy = depth_map.squeeze().cpu().numpy()
        extrs_npy = extrinsic.squeeze().cpu().numpy()
        intrs_npy = intrinsic.squeeze().cpu().numpy()
        video_ten = video_tensor_processed.squeeze()

        if known_depth is not None:
            known_depth = torch.nn.functional.interpolate(
                known_depth[:, None, :, :], size=depth_npy.shape[1:], mode="bilinear", align_corners=False
            )[:, 0, :, :]
            known_depth = known_depth.cpu().numpy()
            depth_npy, scale = align_video_depth_scale(
                depth_npy, known_depth
            )
            if replace_with_known_depth:
                depth_npy = known_depth
                depth_conf = (known_depth > 0).astype(np.float32)

            extrs_npy[:, :3, 3] *= scale

        if stationary_camera:
            extrs_npy = np.repeat(extrs_npy[0:1], extrs_npy.shape[0], axis=0)

        return video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy


class ExternalGeomWrapper(BaseVideoDepthPoseWrapper):
    """
    使用外部提供的几何（深度 + 相机内外参），完全不调用 VGGT。

    预期：
    - known_depth: (T, H, W) torch tensor，单位为米（m），由 --depth_path 加载
    - args.external_geom_npz: NPZ 文件路径，至少包含：
        - intrinsics: (T_ext, 3, 3)
        - extrinsics: (T_ext, 4, 4)  # camera-to-world 或 world-to-camera 由上游约定

    返回：
    - video_ten: (T_use, 3, H, W) torch tensor，值域 [0,1]，仅做简单归一化，不做 VGGT 的 resize/crop
    - depth_npy: (T_use, H, W) numpy array，单位米
    - depth_conf: (T_use, H, W) numpy array，简单取 (depth>0) 作为置信度
    - extrs_npy: (T_use, 4, 4) numpy array，直接来自外部 NPZ（按上游约定）
    - intrs_npy: (T_use, 3, 3) numpy array
    """

    def __init__(self, args):
        super().__init__(args)
        geom_path = getattr(args, "external_geom_npz", None)
        if geom_path is None:
            raise ValueError(
                "depth_pose_method='external' requires --external_geom_npz "
                "pointing to an NPZ file with 'intrinsics' and 'extrinsics'."
            )
        if not os.path.exists(geom_path):
            raise FileNotFoundError(
                f"external_geom_npz not found: {geom_path}"
            )

        data = np.load(geom_path)
        if "intrinsics" not in data or "extrinsics" not in data:
            data.close()
            raise KeyError(
                f"external_geom_npz={geom_path} must contain 'intrinsics' and 'extrinsics' arrays."
            )

        self.intrs_npy_full = data["intrinsics"]  # (T_ext, 3, 3)
        self.extrs_npy_full = data["extrinsics"]  # (T_ext, 4, 4)
        data.close()

    def __call__(self, video_tensor, known_depth=None, stationary_camera=False, replace_with_known_depth=True):
        if known_depth is None:
            raise ValueError(
                "depth_pose_method='external' requires known_depth via --depth_path "
                "(external depth)."
            )

        # video_tensor: (T, 3, H, W), 原始 0-255
        if not isinstance(video_tensor, torch.Tensor):
            video_tensor = torch.from_numpy(video_tensor)
        video_ten = (video_tensor.float() / 255.0).to(self.device)  # (T, 3, H, W), [0,1]
        T_vid = video_ten.shape[0]

        # 外部深度：假定单位为米 (m)
        if not isinstance(known_depth, torch.Tensor):
            known_depth = torch.from_numpy(known_depth)
        depth = known_depth.float().cpu().numpy()  # (T_depth, H, W)
        T_depth = depth.shape[0]

        T_ext = self.intrs_npy_full.shape[0]
        T_ext2 = self.extrs_npy_full.shape[0]
        if T_ext != T_ext2:
            raise ValueError(
                f"external_geom_npz intrinsics/extrinsics length mismatch: "
                f"{T_ext} vs {T_ext2}"
            )

        # 取三者最短长度对齐时间维度
        T_use = min(T_vid, T_depth, T_ext)
        if T_use == 0:
            raise ValueError(
                f"depth_pose_method='external' got zero usable frames: "
                f"T_vid={T_vid}, T_depth={T_depth}, T_ext={T_ext}"
            )

        if not (T_vid == T_depth == T_ext):
            logger.warning(
                f"[ExternalGeomWrapper] Time length mismatch: "
                f"video={T_vid}, depth={T_depth}, geom={T_ext}; "
                f"using first {T_use} frames."
            )

        video_ten = video_ten[:T_use]
        depth = depth[:T_use]
        intrs_npy = self.intrs_npy_full[:T_use]
        extrs_npy = self.extrs_npy_full[:T_use]

        depth_conf = (depth > 0).astype(np.float32)

        if stationary_camera:
            extrs_npy = np.repeat(extrs_npy[0:1], extrs_npy.shape[0], axis=0)

        return video_ten, depth, depth_conf, extrs_npy, intrs_npy


video_depth_pose_dict = {
    "vggt4": VGGT4Wrapper,
    "external": ExternalGeomWrapper,
}
