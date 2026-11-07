"""Gaussian splat containers and file formats.

- `.ply`: the standard 3D Gaussian Splatting layout (x, y, z, f_dc_*, f_rest_*,
  opacity, scale_*, rot_*), readable by most splat viewers and tools.
- `.splat`: compact 32-byte-per-Gaussian web format (position, scale, RGBA,
  rotation), loaded by the browser viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb01: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb01, float) - 0.5) / SH_C0


def sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    return np.asarray(sh0, float) * SH_C0 + 0.5


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


@dataclass
class Gaussians:
    means: np.ndarray        # (N, 3)
    log_scales: np.ndarray   # (N, 3)
    quats: np.ndarray        # (N, 4) w, x, y, z
    opacity_logit: np.ndarray  # (N,)
    sh0: np.ndarray          # (N, 3) DC coefficients
    shN: np.ndarray | None = None  # (N, K-1, 3) higher-order SH coefficients

    def __len__(self) -> int:
        return len(self.means)

    def subset(self, mask: np.ndarray) -> Gaussians:
        return Gaussians(self.means[mask], self.log_scales[mask], self.quats[mask],
                         self.opacity_logit[mask], self.sh0[mask],
                         None if self.shN is None else self.shN[mask])


def from_points(xyz: np.ndarray, rgb_u8: np.ndarray, opacity: float = 0.95, k: int = 4,
                scale_mult: float = 0.3, max_scale: float | None = None) -> Gaussians:
    """Isotropic Gaussians sized by nearest-neighbour spacing: a fast, training-free 'live map' splat."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, float)
    n = len(xyz)
    if n == 0:
        raise ValueError("no points")
    kq = min(k + 1, n)
    d, _ = cKDTree(xyz).query(xyz, k=kq)
    d = np.asarray(d).reshape(n, -1)[:, 1:] if kq > 1 else np.ones((n, 1))
    s = scale_mult * np.mean(d, axis=1)
    if max_scale is None:
        max_scale = float(np.percentile(s, 75)) if n > 20 else float(s.max())
    s = np.clip(s, 1e-6, max(max_scale, 1e-6))
    return Gaussians(
        means=xyz,
        log_scales=np.log(np.repeat(s[:, None], 3, axis=1)),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        opacity_logit=np.full(n, logit(opacity)),
        sh0=rgb_to_sh0(np.asarray(rgb_u8, float) / 255.0),
    )


def write_ply(path: Path, g: Gaussians) -> None:
    n = len(g)
    rest = 0 if g.shN is None else g.shN.shape[1] * 3
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(rest)]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    cols = [g.means, np.zeros((n, 3)), g.sh0]
    if rest:
        # Channel-major like the reference implementation: (N, K-1, 3) -> (N, 3, K-1).
        cols.append(np.transpose(g.shN, (0, 2, 1)).reshape(n, -1))
    cols += [g.opacity_logit[:, None], g.log_scales, g.quats]
    data = np.concatenate(cols, axis=1).astype(np.float32)
    arr = np.empty(n, dtype=[(nm, "f4") for nm in names])
    for i, nm in enumerate(names):
        arr[nm] = data[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def read_ply(path: Path) -> Gaussians:
    v = PlyData.read(str(path))["vertex"].data
    names = v.dtype.names
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda s: int(s.split("_")[-1]))
    n = len(v)
    shN = None
    if rest_names:
        k1 = len(rest_names) // 3
        shN = np.stack([v[nm] for nm in rest_names], axis=1).reshape(n, 3, k1).transpose(0, 2, 1)
    return Gaussians(
        means=np.stack([v["x"], v["y"], v["z"]], axis=1).astype(float),
        log_scales=np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(float),
        quats=np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(float),
        opacity_logit=np.asarray(v["opacity"], float),
        sh0=np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(float),
        shN=shN,
    )


def write_splat(path: Path, g: Gaussians, max_splats: int | None = None) -> int:
    """Write the 32-byte .splat web format, most visually important Gaussians first."""
    scales = np.exp(g.log_scales)
    alpha = sigmoid(g.opacity_logit)
    importance = np.prod(scales, axis=1) * alpha
    order = np.argsort(-importance)
    if max_splats:
        order = order[:max_splats]
    q = g.quats[order]
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    rgba = np.concatenate([np.clip(sh0_to_rgb(g.sh0[order]), 0, 1), alpha[order, None]], axis=1)
    buf = np.zeros(len(order), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4),
                                      ("rot", "u1", 4)])
    buf["pos"] = g.means[order]
    buf["scale"] = scales[order]
    buf["rgba"] = np.clip(rgba * 255, 0, 255).round()
    buf["rot"] = np.clip(q * 128 + 128, 0, 255).round()
    buf.tofile(path)
    return len(order)


def read_splat(path: Path) -> dict[str, np.ndarray]:
    buf = np.fromfile(path, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4),
                                   ("rot", "u1", 4)])
    return {k: buf[k] for k in buf.dtype.names}


def prune(g: Gaussians, min_opacity: float = 0.02, max_scale_pct: float = 99.5,
          crop_pct: float | None = 99.0) -> Gaussians:
    """Drop near-transparent, oversized and far-outlier Gaussians before web export."""
    keep = sigmoid(g.opacity_logit) >= min_opacity
    smax = np.exp(g.log_scales).max(axis=1)
    keep &= smax <= np.percentile(smax[keep], max_scale_pct) if keep.any() else keep
    if crop_pct and keep.any():
        center = np.median(g.means[keep], axis=0)
        dist = np.linalg.norm(g.means - center, axis=1)
        keep &= dist <= np.percentile(dist[keep], crop_pct)
    return g.subset(keep)
