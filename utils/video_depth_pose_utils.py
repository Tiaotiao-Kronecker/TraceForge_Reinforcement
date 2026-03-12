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


def _load_external_extrinsics(geom_path, camera_name="hand_camera"):
    """
    从 NPZ 或 H5 文件加载外部外参。
    返回: (T, 4, 4) numpy array, float32
    """
    if not os.path.exists(geom_path):
        raise FileNotFoundError(f"external_geom_npz not found: {geom_path}")

    if geom_path.endswith('.h5'):
        import h5py
        with h5py.File(geom_path, 'r') as f:
            extr_key = f"observation/camera/extrinsics/{camera_name}_left"
            if extr_key not in f:
                available = list(f['observation/camera/extrinsics'].keys()) if 'observation/camera/extrinsics' in f else []
                raise KeyError(
                    f"H5 file must contain '{extr_key}'. "
                    f"Available cameras: {available}"
                )
            extrs = f[extr_key][:].astype(np.float32)  # (T, 4, 4)
    else:
        data = np.load(geom_path)
        if "extrinsics" not in data:
            data.close()
            raise KeyError("NPZ file must contain 'extrinsics' array.")
        extrs = data["extrinsics"].astype(np.float32)  # (T, 4, 4)
        data.close()

    logger.info(f"加载外部外参: {geom_path}, 共 {extrs.shape[0]} 帧")
    return extrs


def _load_external_geom(geom_path, camera_name="hand_camera"):
    """
    从 NPZ/H5 加载外部内外参，返回:
        intrs: (T, 3, 3) float32
        extrs: (T, 4, 4) float32
    """
    if not os.path.exists(geom_path):
        raise FileNotFoundError(f"external_geom_npz not found: {geom_path}")

    if geom_path.endswith(".h5"):
        import h5py

        # Try two formats: with _left suffix (droid) and without (other datasets)
        intr_key_with_suffix = f"observation/camera/intrinsics/{camera_name}_left"
        extr_key_with_suffix = f"observation/camera/extrinsics/{camera_name}_left"
        intr_key_no_suffix = f"observation/camera/intrinsics/{camera_name}"
        extr_key_no_suffix = f"observation/camera/extrinsics/{camera_name}"

        with h5py.File(geom_path, "r") as f:
            # Try with _left suffix first
            if intr_key_with_suffix in f and extr_key_with_suffix in f:
                intrs = f[intr_key_with_suffix][:].astype(np.float32)
                extrs = f[extr_key_with_suffix][:].astype(np.float32)
            # Try without suffix
            elif intr_key_no_suffix in f and extr_key_no_suffix in f:
                intrs = f[intr_key_no_suffix][:].astype(np.float32)
                extrs = f[extr_key_no_suffix][:].astype(np.float32)
            else:
                available = []
                if "observation/camera/intrinsics" in f:
                    available = list(f["observation/camera/intrinsics"].keys())
                raise KeyError(
                    f"H5 file must contain either '{intr_key_with_suffix}' or '{intr_key_no_suffix}'. "
                    f"Available cameras: {available}"
                )
    else:
        data = np.load(geom_path)
        if "intrinsics" not in data or "extrinsics" not in data:
            data.close()
            raise KeyError(
                f"NPZ file must contain 'intrinsics' and 'extrinsics' arrays: {geom_path}"
            )
        intrs = data["intrinsics"].astype(np.float32)
        extrs = data["extrinsics"].astype(np.float32)
        data.close()

    if intrs.shape[0] != extrs.shape[0]:
        raise ValueError(
            f"external geom intrinsics/extrinsics length mismatch: {intrs.shape[0]} vs {extrs.shape[0]}"
        )
    logger.info(f"加载外部几何: {geom_path}, 共 {intrs.shape[0]} 帧")
    return intrs, extrs


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

        # 可选：加载外部外参（用于替换 VGGT 预估的外参）
        geom_path = getattr(args, "external_geom_npz", None)
        if geom_path is not None:
            camera_name = getattr(args, "camera_name", "hand_camera")
            self.external_extrs = _load_external_extrinsics(geom_path, camera_name)
            logger.info(f"已加载外部外参，将替换 VGGT 外参（深度仍由 VGGT 预估）")
        else:
            self.external_extrs = None

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

        # 外部外参替换：用外部准确外参替换 VGGT 预估的外参，深度保留 VGGT 的
        if self.external_extrs is not None:
            T_vggt = extrs_npy.shape[0]
            T_ext = self.external_extrs.shape[0]

            if T_ext < T_vggt:
                logger.warning(
                    f"[VGGT4Wrapper] 外部外参帧数({T_ext}) < VGGT帧数({T_vggt})，"
                    f"截取前 {T_ext} 帧"
                )
                extrs_npy = self.external_extrs[:T_ext].copy()
                depth_npy = depth_npy[:T_ext]
                if isinstance(depth_conf, np.ndarray):
                    depth_conf = depth_conf[:T_ext]
                else:
                    depth_conf = depth_conf[:T_ext]
                intrs_npy = intrs_npy[:T_ext]
                video_ten = video_ten[:T_ext]
            elif T_ext > T_vggt:
                logger.info(
                    f"[VGGT4Wrapper] 外部外参帧数({T_ext}) > VGGT帧数({T_vggt})，"
                    f"取前 {T_vggt} 帧外参"
                )
                extrs_npy = self.external_extrs[:T_vggt].copy()
            else:
                extrs_npy = self.external_extrs[:T_vggt].copy()

            logger.info(f"已用外部外参替换 VGGT 外参，共 {extrs_npy.shape[0]} 帧")

        if stationary_camera:
            extrs_npy = np.repeat(extrs_npy[0:1], extrs_npy.shape[0], axis=0)

        return video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy


class ExternalGeomWrapper(BaseVideoDepthPoseWrapper):
    """
    使用外部深度 + 外部内外参，完全跳过 VGGT。
    """

    def __init__(self, args):
        super().__init__(args)
        geom_path = getattr(args, "external_geom_npz", None)
        if geom_path is None:
            raise ValueError(
                "depth_pose_method='external' requires --external_geom_npz "
                "pointing to NPZ/H5 with intrinsics and extrinsics."
            )
        camera_name = getattr(args, "camera_name", "hand_camera")
        extr_mode = getattr(args, "external_extr_mode", "w2c")
        self.external_intrs, external_extrs_raw = _load_external_geom(geom_path, camera_name)

        # 规范化为 TraceForge 内部约定的 w2c（world→camera）矩阵
        if extr_mode == "w2c":
            self.external_extrs = external_extrs_raw.astype(np.float32)
            logger.info("[ExternalGeomWrapper] external_extr_mode='w2c'，按 world→camera 直接使用外参。")
        elif extr_mode == "c2w":
            # 用户提供的是 camera→world，需要求逆得到 world→camera
            try:
                c2w = external_extrs_raw.astype(np.float32)
                w2c = np.linalg.inv(c2w)
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    "Failed to invert external extrinsics when external_extr_mode='c2w'. "
                    "Please check that all 4x4 matrices are valid rigid transforms."
                ) from e
            self.external_extrs = w2c
            logger.info("[ExternalGeomWrapper] external_extr_mode='c2w'，已对外参求逆并转换为 world→camera 形式。")
        else:
            raise ValueError(
                f"Unknown external_extr_mode='{extr_mode}'. Expected 'w2c' or 'c2w'."
            )

    def __call__(self, video_tensor, known_depth=None, stationary_camera=False, replace_with_known_depth=True):
        if known_depth is None:
            raise ValueError(
                "depth_pose_method='external' requires known_depth via --depth_path."
            )

        if not isinstance(video_tensor, torch.Tensor):
            video_tensor = torch.from_numpy(video_tensor)
        video_ten = video_tensor.float()
        # 兼容上游未归一化输入
        if torch.max(video_ten) > 1.5:
            video_ten = video_ten / 255.0
        video_ten = video_ten.to(self.device)
        t_vid = video_ten.shape[0]

        if not isinstance(known_depth, torch.Tensor):
            known_depth = torch.from_numpy(known_depth)
        depth_npy = known_depth.float().cpu().numpy()
        t_depth = depth_npy.shape[0]

        t_geom = self.external_extrs.shape[0]
        t_use = min(t_vid, t_depth, t_geom)
        if t_use <= 0:
            raise ValueError(
                f"depth_pose_method='external' got zero usable frames: "
                f"video={t_vid}, depth={t_depth}, geom={t_geom}"
            )
        if not (t_vid == t_depth == t_geom):
            logger.warning(
                f"[ExternalGeomWrapper] Time length mismatch: "
                f"video={t_vid}, depth={t_depth}, geom={t_geom}; using first {t_use} frames."
            )

        video_ten = video_ten[:t_use]
        depth_npy = depth_npy[:t_use]
        intrs_npy = self.external_intrs[:t_use]
        extrs_npy = self.external_extrs[:t_use]
        depth_conf = (depth_npy > 0).astype(np.float32)

        if stationary_camera:
            extrs_npy = np.repeat(extrs_npy[0:1], extrs_npy.shape[0], axis=0)

        return video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy


video_depth_pose_dict = {
    "vggt4": VGGT4Wrapper,
    "external": ExternalGeomWrapper,
}
