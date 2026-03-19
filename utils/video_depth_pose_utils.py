import os

import numpy as np
import torch
from loguru import logger

from utils.extrinsics_utils import normalize_extrinsics_to_w2c


DEFAULT_DEPTH_POSE_METHOD = "external"


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

        intr_key_with_suffix = f"observation/camera/intrinsics/{camera_name}_left"
        extr_key_with_suffix = f"observation/camera/extrinsics/{camera_name}_left"
        intr_key_no_suffix = f"observation/camera/intrinsics/{camera_name}"
        extr_key_no_suffix = f"observation/camera/extrinsics/{camera_name}"

        with h5py.File(geom_path, "r") as f:
            if intr_key_with_suffix in f and extr_key_with_suffix in f:
                intrs = f[intr_key_with_suffix][:].astype(np.float32)
                extrs = f[extr_key_with_suffix][:].astype(np.float32)
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


class ExternalGeomWrapper:
    """
    使用外部深度 + 外部内外参，完全跳过任何模型化深度/位姿前端。
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device

        geom_path = getattr(args, "external_geom_npz", None)
        if geom_path is None:
            raise ValueError(
                "external-only mode requires --external_geom_npz pointing to NPZ/H5 with intrinsics and extrinsics."
            )
        camera_name = getattr(args, "camera_name", "hand_camera")
        extr_mode = getattr(args, "external_extr_mode", "w2c")
        self.external_intrs, external_extrs_raw = _load_external_geom(geom_path, camera_name)
        self.external_extrs = normalize_extrinsics_to_w2c(
            external_extrs_raw,
            extr_mode=extr_mode,
            context="ExternalGeomWrapper external extrinsics",
        )
        logger.info(
            f"[ExternalGeomWrapper] external_extr_mode='{extr_mode}'，外参已统一为 world→camera (w2c)。"
        )

    def __call__(
        self,
        video_tensor,
        known_depth=None,
        stationary_camera=False,
        replace_with_known_depth=True,
    ):
        del replace_with_known_depth

        if known_depth is None:
            raise ValueError("external-only mode requires known_depth via --depth_path.")

        if not isinstance(video_tensor, torch.Tensor):
            video_tensor = torch.from_numpy(video_tensor)
        video_ten = video_tensor.float()
        if video_ten.numel() > 0 and torch.max(video_ten) > 1.5:
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
                f"external-only mode got zero usable frames: "
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
    DEFAULT_DEPTH_POSE_METHOD: ExternalGeomWrapper,
}
