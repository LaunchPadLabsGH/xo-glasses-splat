"""Accuracy metrics: trajectory error against ground truth, and image quality."""

from __future__ import annotations

import numpy as np

from xgs.sfm.scale import umeyama


def associate(t_est: np.ndarray, t_gt: np.ndarray, max_dt: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (est, gt) matching each estimate to the nearest ground-truth time within max_dt."""
    t_gt = np.asarray(t_gt)
    order = np.argsort(t_gt)
    ts = t_gt[order]
    idx = np.clip(np.searchsorted(ts, t_est), 1, len(ts) - 1)
    left, right = ts[idx - 1], ts[idx]
    nearest = np.where(np.abs(t_est - left) <= np.abs(right - t_est), idx - 1, idx)
    dt = np.abs(ts[nearest] - t_est)
    ok = dt <= max_dt
    return np.nonzero(ok)[0], order[nearest[ok]]


def ate(est: np.ndarray, gt: np.ndarray, max_dt: float = 0.02, with_scale: bool = True) -> dict:
    """Absolute trajectory error (camera centres) after similarity alignment.

    est, gt: rows t, tx, ty, tz, ... . Monocular estimates are scale-free, so a
    Sim3 alignment is used; the recovered scale is reported.
    """
    ie, ig = associate(est[:, 0], gt[:, 0], max_dt)
    if len(ie) < 3:
        return {"matched": int(len(ie)), "error": "too few matched poses"}
    p_est, p_gt = est[ie, 1:4], gt[ig, 1:4]
    tf = umeyama(p_est, p_gt, with_scale=with_scale)
    err = np.linalg.norm(tf.apply(p_est) - p_gt, axis=1)
    path_len = float(np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1)))
    return {
        "matched": int(len(ie)),
        "ate_rmse_m": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "ate_median_m": round(float(np.median(err)), 4),
        "ate_max_m": round(float(err.max()), 4),
        "gt_path_length_m": round(path_len, 3),
        "ate_pct_of_path": round(100.0 * float(np.sqrt(np.mean(err ** 2))) / path_len, 2) if path_len else None,
        "sim3_scale": round(float(tf.s), 4),
    }


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR for images in [0, 1]."""
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
