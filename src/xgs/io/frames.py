"""Frame tables, image-sequence/TUM ingest, and ground-truth trajectories."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Frame:
    idx: int
    t: float
    file: str


def write_frames_csv(path: Path, frames: list[Frame]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t", "file"])
        for fr in frames:
            w.writerow([fr.idx, f"{fr.t:.6f}", fr.file])


def read_frames_csv(path: Path) -> list[Frame]:
    with open(path) as f:
        return [Frame(int(r["idx"]), float(r["t"]), r["file"]) for r in csv.DictReader(f)]


def _save_resized(src: Path, dst: Path, max_width: int) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {src}")
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _subsample(times: np.ndarray, fps: float) -> list[int]:
    """Indices that thin a timestamped sequence to roughly `fps`."""
    keep, next_t = [], -np.inf
    for i, t in enumerate(times):
        if t >= next_t - 1e-6:
            keep.append(i)
            next_t = t + 1.0 / fps
    return keep


def ingest_image_folder(src: Path, out_dir: Path, fps: float, source_fps: float,
                        max_width: int) -> list[Frame]:
    """Sorted images in a folder, assumed captured at `source_fps`."""
    files = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    times = np.arange(len(files)) / source_fps
    return _copy_selected(files, times, out_dir, fps, max_width)


def ingest_tum(src: Path, out_dir: Path, fps: float, max_width: int,
               start_s: float = 0.0, duration_s: float | None = None) -> tuple[list[Frame], np.ndarray | None]:
    """TUM RGB-D sequence folder (rgb.txt, groundtruth.txt). Times are made relative to the first frame."""
    src = Path(src)
    rows = _read_tum_list(src / "rgb.txt")
    t_abs = np.array([float(r[0]) for r in rows])
    t0 = t_abs[0]
    times = t_abs - t0
    files = [src / r[1] for r in rows]
    sel = (times >= start_s) & ((times <= start_s + duration_s) if duration_s else True)
    files = [f for f, s in zip(files, sel, strict=True) if s]
    times = times[sel]
    frames = _copy_selected(files, times, out_dir, fps, max_width)
    gt = None
    gt_path = src / "groundtruth.txt"
    if gt_path.exists():
        g = np.array([[float(x) for x in r] for r in _read_tum_list(gt_path)])
        g[:, 0] -= t0
        gt = g  # columns: t, tx, ty, tz, qx, qy, qz, qw  (camera-to-world)
    return frames, gt


def _copy_selected(files: list[Path], times: np.ndarray, out_dir: Path, fps: float,
                   max_width: int) -> list[Frame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    frames = []
    for new_idx, i in enumerate(_subsample(times, fps)):
        name = f"{new_idx:06d}.jpg"
        _save_resized(files[i], out_dir / name, max_width)
        frames.append(Frame(new_idx, float(times[i]), name))
    return frames


def _read_tum_list(path: Path) -> list[list[str]]:
    with open(path) as f:
        return [line.split() for line in f if line.strip() and not line.startswith("#")]


def write_trajectory_csv(path: Path, traj: np.ndarray) -> None:
    """Rows: t, tx, ty, tz, qx, qy, qz, qw (camera-to-world)."""
    np.savetxt(path, traj, delimiter=",", fmt="%.9f", header="t,tx,ty,tz,qx,qy,qz,qw", comments="")


def read_trajectory_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
