"""Metric scale and alignment.

A single moving camera recovers geometry only up to scale. Two ways to fix it:
- ArUco markers placed at tape-measured positions in the scene (capture
  protocol for own footage): `marker_scale` gives the scale and a distance error.
- Alignment to a reference trajectory or survey: `umeyama` finds the similarity
  transform (scale, rotation, translation) between corresponding points.
In the funded project, IMU data from the glasses provides scale on the device.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Sim3:
    s: float
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.s * (np.asarray(x) @ self.R.T) + self.t


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares similarity transform mapping src -> dst (Umeyama, 1991)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need >= 3 corresponding points with matching shapes")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return Sim3(s, R, t)


def detect_markers(gray: np.ndarray, dictionary: int = cv2.aruco.DICT_4X4_50) -> dict[int, np.ndarray]:
    """Return {marker_id: (4, 2) corner pixels}."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.reshape(-1), corners, strict=True)}


def marker_scale(reconstructed: dict[int, np.ndarray], measured: dict[int, np.ndarray]) -> dict:
    """Scale factor and distance errors from marker centres.

    reconstructed: marker id -> 3D centre in model units (triangulated from SfM views).
    measured:      marker id -> 3D centre in metres (tape-measured survey).
    Uses all pairwise distances; the scale is the median ratio, and the error is
    reported per pair after scaling.
    """
    ids = sorted(set(reconstructed) & set(measured))
    if len(ids) < 2:
        raise ValueError("need at least two markers seen in both reconstruction and survey")
    ratios, pairs = [], []
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            d_model = np.linalg.norm(reconstructed[a] - reconstructed[b])
            d_true = np.linalg.norm(measured[a] - measured[b])
            if d_model > 0:
                ratios.append(d_true / d_model)
                pairs.append((a, b, d_model, d_true))
    scale = float(np.median(ratios))
    errs = [abs(scale * dm - dt) for _, _, dm, dt in pairs]
    rel = [e / dt for e, (_, _, _, dt) in zip(errs, pairs, strict=True) if dt > 0]
    return {
        "scale": scale,
        "pairs": len(pairs),
        "mean_abs_error_m": float(np.mean(errs)),
        "max_abs_error_m": float(np.max(errs)),
        "mean_rel_error_pct": float(100 * np.mean(rel)) if rel else None,
    }
