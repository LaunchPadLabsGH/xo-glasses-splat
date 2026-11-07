"""Keyframe selection for head-worn footage.

Smart-glasses video is full of frames that hurt reconstruction: motion blur from
fast head turns, near-duplicates while standing still, and frames with too
little baseline to triangulate. This stage keeps a compact, sharp, well-spaced
set and records a reason for every frame it drops.

Signals
- Sharpness: variance of the Laplacian, judged relative to a rolling local
  median (so it adapts to scene texture and camera) plus an absolute floor.
- Motion: peak gyro rate around the frame time when IMU data exists, otherwise
  median optical-flow magnitude between consecutive frames.
- Parallax: median displacement of features tracked since the last keyframe; a
  new keyframe is taken once it exceeds a threshold, choosing the sharpest of
  the last few acceptable frames.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from xgs.io.frames import Frame
from xgs.io.imu import ImuData

ANALYSIS_WIDTH = 640


@dataclass
class FrameScore:
    idx: int
    t: float
    file: str
    sharpness: float = 0.0
    rel_sharpness: float = 0.0
    flow_frac: float = 0.0
    gyro_deg_s: float | None = None
    parallax_frac: float = 0.0
    status: str = "pending"   # keyframe | rejected | skipped
    reason: str = ""


def load_gray(path: Path, width: int = ANALYSIS_WIDTH) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read {path}")
    h, w = img.shape
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return img


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp edges, low for blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rolling_median(values: np.ndarray, half_window: int) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    n = len(values)
    for i in range(n):
        a, b = max(0, i - half_window), min(n, i + half_window + 1)
        out[i] = np.median(values[a:b])
    return out


def _features(gray: np.ndarray, max_corners: int = 400) -> np.ndarray | None:
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
    return None if pts is None else pts.astype(np.float32)


def _track(prev: np.ndarray, cur: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1).astype(bool)
    return nxt, ok


def median_flow(prev: np.ndarray, cur: np.ndarray) -> float:
    pts = _features(prev, 300)
    if pts is None or len(pts) < 10:
        return 0.0
    nxt, ok = _track(prev, cur, pts)
    if ok.sum() < 10:
        return float("inf")  # tracking failed entirely: treat as extreme motion
    d = np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)
    return float(np.median(d))


def select_keyframes(frames_dir: Path, frames: list[Frame], params: dict,
                     imu: ImuData | None = None) -> list[FrameScore]:
    frames_dir = Path(frames_dir)
    scores = [FrameScore(f.idx, f.t, f.file) for f in frames]
    if not scores:
        return scores

    # Pass 1: sharpness and inter-frame motion.
    prev = None
    width = ANALYSIS_WIDTH
    for s in scores:
        g = load_gray(frames_dir / s.file)
        width = g.shape[1]
        s.sharpness = sharpness(g)
        s.flow_frac = 0.0 if prev is None else median_flow(prev, g) / width
        prev = g
    sharp = np.array([s.sharpness for s in scores])
    local = rolling_median(sharp, int(params["blur_window"]))
    if imu is not None:
        gyro = imu.angular_speed_deg_s(np.array([s.t for s in scores]))
        for s, gv in zip(scores, gyro, strict=True):
            s.gyro_deg_s = float(gv)

    acceptable = []
    for s, loc in zip(scores, local, strict=True):
        s.rel_sharpness = float(s.sharpness / loc) if loc > 0 else 0.0
        if s.sharpness < params["blur_abs_min"] or s.rel_sharpness < params["blur_rel_threshold"]:
            s.status, s.reason = "rejected", "blur"
        elif s.gyro_deg_s is not None and s.gyro_deg_s > params["max_gyro_deg_s"]:
            s.status, s.reason = "rejected", "fast_head_turn"
        elif s.gyro_deg_s is None and s.flow_frac > params["max_flow_frac"]:
            s.status, s.reason = "rejected", "fast_motion"
        acceptable.append(s.status == "pending")

    # Pass 2: parallax-driven keyframing over acceptable frames.
    min_par = float(params["min_parallax_frac"])
    lookback = int(params["lookback"])
    kf_gray = kf_pts = None
    chain_prev = None
    cur_pts = None
    recent: list[tuple[FrameScore, np.ndarray]] = []

    def start_keyframe(s: FrameScore, g: np.ndarray) -> None:
        nonlocal kf_gray, kf_pts, cur_pts, chain_prev, recent
        s.status, s.reason = "keyframe", s.reason or "parallax"
        kf_gray, kf_pts = g, _features(g)
        cur_pts, chain_prev, recent = kf_pts, g, []

    for s, ok in zip(scores, acceptable, strict=True):
        if not ok:
            continue
        g = load_gray(frames_dir / s.file)
        if kf_gray is None or kf_pts is None or len(kf_pts) < 20:
            s.reason = "first" if kf_gray is None else "reinit"
            start_keyframe(s, g)
            continue
        nxt, good = _track(chain_prev, g, cur_pts)
        if good.sum() < 30:
            s.reason = "tracking_refresh"
            start_keyframe(s, g)
            continue
        kf_pts, cur_pts = kf_pts[good], nxt[good]
        chain_prev = g
        disp = np.linalg.norm((cur_pts - kf_pts).reshape(-1, 2), axis=1)
        s.parallax_frac = float(np.median(disp) / g.shape[1])
        recent = (recent + [(s, g)])[-lookback:]
        if s.parallax_frac >= min_par:
            best, best_g = max(recent, key=lambda r: r[0].sharpness)
            for r, _ in recent:
                if r is not best and r.status == "pending":
                    r.status, r.reason = "skipped", "redundant"
            start_keyframe(best, best_g)
            # Re-anchor tracking on the chosen frame and catch up to the current one.
            if best is not s and kf_pts is not None and len(kf_pts) >= 20:
                nxt2, good2 = _track(best_g, g, kf_pts)
                kf_pts = kf_pts[good2]
                cur_pts, chain_prev = nxt2[good2], g
                s.status, s.reason = "skipped", "redundant"

    for s in scores:
        if s.status == "pending":
            s.status, s.reason = "skipped", "low_parallax"

    # Budget: thin uniformly if there are too many keyframes.
    kfs = [s for s in scores if s.status == "keyframe"]
    budget = int(params["max_keyframes"])
    if len(kfs) > budget:
        keep = set(np.round(np.linspace(0, len(kfs) - 1, budget)).astype(int).tolist())
        for i, s in enumerate(kfs):
            if i not in keep:
                s.status, s.reason = "skipped", "budget"
    return scores


def summarise(scores: list[FrameScore]) -> dict:
    status = Counter(s.status for s in scores)
    reasons = Counter(s.reason for s in scores if s.status != "keyframe")
    return {
        "frames_in": len(scores),
        "keyframes": status.get("keyframe", 0),
        "rejected": status.get("rejected", 0),
        "skipped": status.get("skipped", 0),
        "drop_reasons": dict(reasons),
    }


def scores_to_rows(scores: list[FrameScore]) -> list[dict]:
    return [asdict(s) for s in scores]
