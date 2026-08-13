"""
Preprocessing for D435 carpet segmentation inference.

Pipeline:
    1. Normalize depth to [0, 1]
    2. Compute geometric features (normals, edge, curvature) from dense depth
    3. Normalize RGB with ImageNet stats
    4. Stack into 9ch tensor: RGB(3) + Depth(1) + Normal(3) + Edge(1) + Curvature(1)

No dToF dependency — uses D435 dense depth only.
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Optional


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to [0, 1] using per-frame min/max."""
    d = depth.astype(np.float32).copy()
    valid = d > 0
    if valid.sum() == 0:
        return np.zeros_like(d)
    d_min, d_max = d[valid].min(), d[valid].max()
    if d_max - d_min < 1e-6:
        d[valid] = 0.5
    else:
        d[valid] = (d[valid] - d_min) / (d_max - d_min)
    d[~valid] = 0.0
    return d


def compute_dense_geo_features(
    depth: np.ndarray,
    sobel_kernel_size: int = 3,
) -> dict:
    """
    Extract geometric features from D435 dense depth map.

    Returns:
        normal:        (H, W, 3) float32 — surface normals (unit vectors)
        depth_edge:    (H, W)    float32 — Sobel gradient magnitude
        curvature:     (H, W)    float32 — |Laplacian| surface curvature
    """
    # Surface normals from depth gradients
    dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)

    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    normal = normal / norm

    # Depth edge magnitude
    edge_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

    # Surface curvature (Laplacian)
    laplacian = cv2.Laplacian(depth, cv2.CV_32F, ksize=sobel_kernel_size)
    curvature = np.abs(laplacian)

    return {
        'normal': normal.astype(np.float32),
        'depth_edge': edge_mag.astype(np.float32),
        'curvature': curvature.astype(np.float32),
    }


class Preprocessor:
    """
    Preprocessor for D435 carpet segmentation model.

    Prepares 9-channel input tensor:
        Ch 0-2:  RGB (ImageNet-normalized)
        Ch 3:    Depth (per-frame min/max normalized)
        Ch 4-6:  Surface normals (unit vectors, from depth gradient)
        Ch 7:    Depth edge (Sobel magnitude)
        Ch 8:    Curvature (|Laplacian|)

    Args:
        target_size: (H, W) resize target, default (480, 640)
        device:      torch device
    """

    # ImageNet statistics
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        target_size: Tuple[int, int] = (480, 640),
        device: str = 'cuda',
    ):
        self.target_size = target_size  # (H, W)
        self.device = device

    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> torch.Tensor:
        """
        Process a single RGB-D frame into a 9ch input tensor.

        Args:
            rgb:   (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth in meters, same spatial size as RGB

        Returns:
            tensor: (1, 9, H_target, W_target) float32 on self.device
        """
        H_t, W_t = self.target_size

        # ── Resize ──
        if rgb.shape[0] != H_t or rgb.shape[1] != W_t:
            rgb = cv2.resize(rgb, (W_t, H_t), interpolation=cv2.INTER_LINEAR)
        if depth.shape[0] != H_t or depth.shape[1] != W_t:
            depth = cv2.resize(depth, (W_t, H_t), interpolation=cv2.INTER_NEAREST)

        depth = depth.astype(np.float32)

        # ── RGB: normalize [0,255]→[0,1], then ImageNet stats ──
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_norm = (rgb_f - self.IMAGENET_MEAN) / self.IMAGENET_STD
        rgb_chw = rgb_norm.transpose(2, 0, 1)  # (3, H, W)

        # ── Depth: per-frame min/max normalize ──
        depth_norm = normalize_depth(depth)
        depth_ch = depth_norm[np.newaxis, ...]  # (1, H, W)

        # ── Geometric features ──
        geo = compute_dense_geo_features(depth)

        # Normal vectors: (H, W, 3) → (3, H, W)
        normal_chw = geo['normal'].transpose(2, 0, 1)

        # Edge: normalize to [0,1] then add channel dim
        edge_ch = normalize_depth(geo['depth_edge'])[np.newaxis, ...]

        # Curvature: normalize to [0,1] then add channel dim
        curv_ch = normalize_depth(geo['curvature'])[np.newaxis, ...]

        # ── Stack: 3 + 1 + 3 + 1 + 1 = 9 channels ──
        tensor = np.concatenate(
            [rgb_chw, depth_ch, normal_chw, edge_ch, curv_ch],
            axis=0,
        ).astype(np.float32)

        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def process_batch(
        self,
        rgbs: list,
        depths: list,
    ) -> torch.Tensor:
        """
        Process multiple RGB-D frames.
        Args:
            rgbs:   list of (H, W, 3) uint8 numpy arrays
            depths: list of (H, W) float32 numpy arrays
        Returns:
            tensor: (B, 9, H_target, W_target)
        """
        tensors = []
        for rgb, depth in zip(rgbs, depths):
            t = self.process(rgb, depth)
            tensors.append(t)
        return torch.cat(tensors, dim=0)
