"""Live-path replay: a lightweight monocular visual-odometry front end.

This simulates the on-device path: frames arrive in capture order, each one is
processed once, and per-frame latency is measured against the frame budget. It
produces a camera trajectory and a growing sparse map (the "live map").

It is deliberately simple (ORB features, essential-matrix relative pose,
two-view triangulation, scale chained through shared points) and is a CPU proxy
for the visual-inertial odometry planned for the funded project. Limitations:
monocular, so no metric scale and some scale drift; no loop closure; no bundle
adjustment. Timings are from the machine it runs on, not from glasses hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from xgs.io.frames import Frame

ANALYSIS_WIDTH = 960


@dataclass
class Keyframe:
    t: float
    R_wc: np.ndarray        # world_from_cam rotation
    c_w: np.ndarray         # camera centre in world
    kps: np.ndarray         # (N, 2) undistorted pixel coords
    desc: np.ndarray
    depth_by_kp: dict[int, float] = field(default_factory=dict)  # scaled depth of triangulated kps
    segment: int = 0          # map segment; a new one starts after tracking is lost


@dataclass
class LiveResult:
    trajectory: np.ndarray            # rows t, tx, ty, tz, qx, qy, qz, qw
    points: np.ndarray                # (M, 3)
    colors: np.ndarray                # (M, 3) uint8
    latency_ms: np.ndarray            # per processed frame
    is_keyframe: np.ndarray           # per processed frame (bool)
    frame_t: np.ndarray               # per processed frame
    n_tracking_failures: int = 0
    n_resets: int = 0
    segments: np.ndarray | None = None  # per trajectory row: map segment id
    point_segments: np.ndarray | None = None  # per map point: map segment id


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()  # x, y, z, w


def scaled_intrinsics(calib_K: np.ndarray, src_w: int, dst_w: int) -> np.ndarray:
    s = dst_w / src_w
    K = calib_K.copy().astype(float)
    K[:2] *= s
    return K


class LiveVO:
    def __init__(self, K: np.ndarray, dist: np.ndarray | None, params: dict, image_width: int):
        self.K = K
        self.dist = dist
        self.orb = cv2.ORB_create(nfeatures=int(params["orb_features"]), fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = float(params["ratio"])
        self.min_parallax_px = float(params["min_parallax_frac"]) * image_width
        self.min_inliers = int(params["min_inliers"])
        self.min_triangulated = int(params.get("min_triangulated", 40))
        self.max_points = int(params["max_points"])
        self.kfs: list[Keyframe] = []
        self.points: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.point_segments: list[np.ndarray] = []
        self.failures = 0
        self.consecutive_failures = 0
        self.resets = 0
        self.max_consecutive_failures = int(params.get("max_consecutive_failures", 8))

    def _detect(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        kps, desc = self.orb.detectAndCompute(gray, None)
        if not kps:
            return np.zeros((0, 2), np.float32), None
        pts = np.array([k.pt for k in kps], np.float32)
        if self.dist is not None and np.any(self.dist):
            pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        return pts, desc

    def _match(self, d0: np.ndarray, d1: np.ndarray) -> list[tuple[int, int]]:
        if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
            return []
        out = []
        for pair in self.matcher.knnMatch(d0, d1, k=2):
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance:
                out.append((pair[0].queryIdx, pair[0].trainIdx))
        return out

    def process(self, t: float, bgr: np.ndarray) -> bool:
        """Process one frame. Returns True if it became a keyframe."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        pts, desc = self._detect(gray)
        if not self.kfs:
            if desc is None or len(pts) < self.min_inliers:
                return False
            self.kfs.append(Keyframe(t, np.eye(3), np.zeros(3), pts, desc))
            return True
        kf = self.kfs[-1]
        matches = self._match(kf.desc, desc)
        if len(matches) < self.min_inliers:
            return self._fail(t, pts, desc)
        i0 = np.array([m[0] for m in matches])
        i1 = np.array([m[1] for m in matches])
        p0, p1 = kf.kps[i0], pts[i1]
        if np.median(np.linalg.norm(p1 - p0, axis=1)) < self.min_parallax_px:
            return False
        E, mask = cv2.findEssentialMat(p0, p1, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            return self._fail(t, pts, desc)
        n_in, R, tvec, mask = cv2.recoverPose(E, p0, p1, self.K, mask=mask)
        if n_in < self.min_inliers:
            return self._fail(t, pts, desc)
        inl = mask.reshape(-1).astype(bool)
        p0, p1, i0, i1 = p0[inl], p1[inl], i0[inl], i1[inl]
        tvec = tvec.reshape(3)

        # Triangulate in the previous keyframe's camera frame (unit baseline).
        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([R, tvec[:, None]])
        X = cv2.triangulatePoints(P0, P1, p0.T.astype(float), p1.T.astype(float))
        X = (X[:3] / X[3]).T
        X1 = X @ R.T + tvec
        ok = (X[:, 2] > 0) & (X1[:, 2] > 0)
        for P, p, Xc in ((P0, p0, X), (P1, p1, X)):
            proj = (P @ np.hstack([Xc, np.ones((len(Xc), 1))]).T).T
            err = np.linalg.norm(proj[:, :2] / proj[:, 2:3] - p, axis=1)
            ok &= err < 2.0
        cos_par = np.sum(X * (X - (-R.T @ tvec)), axis=1) / (
            np.linalg.norm(X, axis=1) * np.linalg.norm(X + R.T @ tvec, axis=1) + 1e-12)
        ok &= cos_par < np.cos(np.radians(1.0))
        if ok.sum() < self.min_triangulated:
            # Mostly rotation so far: not enough baseline to triangulate. Wait for more
            # translation rather than insert a weak keyframe that breaks scale chaining.
            return False

        # Relative scale from depths shared with the previous pair; fall back to the
        # previous step length, and clamp jumps (scale cannot change abruptly).
        prev_step = float(np.linalg.norm(kf.c_w - self.kfs[-2].c_w)) if len(self.kfs) > 1 else 0.0
        scale = prev_step or 1.0
        shared = [(k, z) for k, z in zip(i0[ok], X[ok, 2], strict=True) if int(k) in kf.depth_by_kp]
        if len(shared) >= 8:
            ratios = np.array([kf.depth_by_kp[int(k)] / z for k, z in shared])
            scale = float(np.median(ratios))
            if prev_step > 0:
                scale = float(np.clip(scale, prev_step / 4, prev_step * 4))
        self.consecutive_failures = 0

        # Compose the new pose: cam1_from_cam0 = (R, t*scale).
        R_wc = kf.R_wc @ R.T
        c_w = kf.c_w + kf.R_wc @ (-R.T @ (tvec * scale))
        Xw = (X[ok] * scale) @ kf.R_wc.T + kf.c_w
        new_kf = Keyframe(t, R_wc, c_w, pts, desc, segment=kf.segment)
        new_kf.depth_by_kp = {int(k): float(z * scale) for k, z in zip(i1[ok], X1[ok, 2], strict=True)}
        self.kfs.append(new_kf)

        if sum(len(p) for p in self.points) < self.max_points:
            uv = np.round(p1[ok] * 1).astype(int)
            h, w = bgr.shape[:2]
            uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
            uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)
            self.points.append(Xw)
            self.colors.append(bgr[uv[:, 1], uv[:, 0], ::-1])
            self.point_segments.append(np.full(len(Xw), kf.segment, dtype=int))
        return True

    def _fail(self, t: float, pts: np.ndarray, desc: np.ndarray | None) -> bool:
        """Tracking failed for this frame. After several in a row, start a new map segment."""
        self.failures += 1
        self.consecutive_failures += 1
        if (self.consecutive_failures >= self.max_consecutive_failures and desc is not None
                and len(pts) >= self.min_inliers):
            kf = self.kfs[-1]
            self.resets += 1
            self.kfs.append(Keyframe(t, kf.R_wc, kf.c_w, pts, desc, segment=self.resets))
            self.consecutive_failures = 0
            return True
        return False

    def result(self, latency: list[float], is_kf: list[bool], times: list[float]) -> LiveResult:
        traj = np.array([[k.t, *k.c_w, *rotmat_to_quat_xyzw(k.R_wc)] for k in self.kfs]).reshape(-1, 8)
        pts = np.concatenate(self.points) if self.points else np.zeros((0, 3))
        cols = np.concatenate(self.colors) if self.colors else np.zeros((0, 3), np.uint8)
        return LiveResult(traj, pts, cols.astype(np.uint8), np.array(latency), np.array(is_kf),
                          np.array(times), self.failures, self.resets,
                          np.array([k.segment for k in self.kfs], dtype=int),
                          np.concatenate(self.point_segments) if self.point_segments else np.zeros(0, int))


def replay(frames_dir: Path, frames: list[Frame], params: dict, K_full: np.ndarray,
           dist: np.ndarray | None, full_width: int) -> LiveResult:
    """Feed frames one at a time, as they would arrive on the device, timing each."""
    first = cv2.imread(str(Path(frames_dir) / frames[0].file))
    h, w = first.shape[:2]
    scale_to = min(ANALYSIS_WIDTH, w)
    K = scaled_intrinsics(K_full, full_width, scale_to)
    vo = LiveVO(K, dist, params, scale_to)
    latency, is_kf, times = [], [], []
    for fr in frames:
        img = cv2.imread(str(Path(frames_dir) / fr.file))
        if img.shape[1] != scale_to:
            img = cv2.resize(img, (scale_to, round(img.shape[0] * scale_to / img.shape[1])),
                             interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        kf = vo.process(fr.t, img)
        latency.append((time.perf_counter() - t0) * 1000.0)
        is_kf.append(kf)
        times.append(fr.t)
    return vo.result(latency, is_kf, times)


def latency_summary(res: LiveResult, ingest_fps: float, capture_fps: float = 30.0) -> dict:
    lat = res.latency_ms
    if len(lat) == 0:
        return {}
    return {
        "frames_processed": int(len(lat)),
        "keyframes": int(res.is_keyframe.sum()),
        "live_map_points": int(len(res.points)),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "latency_ms_max": round(float(lat.max()), 1),
        "budget_ms_at_ingest_fps": round(1000.0 / ingest_fps, 1),
        "pct_within_ingest_budget": round(100.0 * float(np.mean(lat <= 1000.0 / ingest_fps)), 1),
        "budget_ms_at_30fps": round(1000.0 / capture_fps, 1),
        "pct_within_30fps_budget": round(100.0 * float(np.mean(lat <= 1000.0 / capture_fps)), 1),
        "frames_not_tracked": int(res.n_tracking_failures),
        "map_resets": int(res.n_resets),
    }
