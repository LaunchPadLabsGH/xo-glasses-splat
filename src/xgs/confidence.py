"""Per-point confidence: how well each part of the map is actually supported.

A point seen by only two nearby cameras is a guess; one seen from many
viewpoints at wide angles is well constrained. Confidence combines the number
of observations with the widest triangulation angle, so analysts can see which
regions were observed properly and which are thin or extrapolated.
"""

from __future__ import annotations

import numpy as np


def max_triangulation_angle_deg(point: np.ndarray, centers: np.ndarray, max_views: int = 12) -> float:
    """Largest angle subtended at `point` by any two observing camera centres."""
    if len(centers) < 2:
        return 0.0
    if len(centers) > max_views:  # evenly sample long tracks to bound cost
        centers = centers[np.round(np.linspace(0, len(centers) - 1, max_views)).astype(int)]
    rays = centers - point
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays / np.clip(norms, 1e-12, None)
    cos = np.clip(rays @ rays.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos.min())))


def point_confidence(track_len: np.ndarray, angles_deg: np.ndarray, full_obs: int = 6,
                     full_angle_deg: float = 10.0) -> np.ndarray:
    """Confidence in [0, 1]: geometric mean of observation support and angle support."""
    obs = np.clip((np.asarray(track_len, float) - 1.0) / max(1.0, full_obs - 1.0), 0.0, 1.0)
    ang = np.clip(np.asarray(angles_deg, float) / full_angle_deg, 0.0, 1.0)
    return np.sqrt(obs * ang)


def confidence_colors(conf: np.ndarray) -> np.ndarray:
    """Map confidence to RGB uint8: red (low) -> amber -> green (high)."""
    c = np.clip(np.asarray(conf, float), 0, 1)[:, None]
    low = np.array([220, 50, 47], float)
    mid = np.array([235, 170, 40], float)
    high = np.array([40, 170, 90], float)
    rgb = np.where(c < 0.5, low + (mid - low) * (c / 0.5), mid + (high - mid) * ((c - 0.5) / 0.5))
    return rgb.round().astype(np.uint8)


def nearest_confidence(query_xyz: np.ndarray, point_xyz: np.ndarray, point_conf: np.ndarray,
                       k: int = 4) -> np.ndarray:
    """Propagate confidence to arbitrary positions (e.g. Gaussians) from the nearest SfM points."""
    from scipy.spatial import cKDTree

    if len(point_xyz) == 0:
        return np.zeros(len(query_xyz))
    k = min(k, len(point_xyz))
    _, idx = cKDTree(point_xyz).query(query_xyz, k=k)
    idx = np.asarray(idx).reshape(len(query_xyz), -1)
    return point_conf[idx].mean(axis=1)
