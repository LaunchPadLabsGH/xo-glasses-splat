"""Camera poses and sparse structure with COLMAP (via pycolmap, BSD-3).

This is the base-station (post-mission) pose solver: SIFT features, sequential
matching suited to video, then incremental (robust) or global (GLOMAP, faster)
mapping. The largest reconstruction is kept, undistorted to a pinhole model and
handed to Gaussian Splatting training.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pycolmap


def _reader_options(params: dict, calibration: dict | None) -> pycolmap.ImageReaderOptions:
    opts = pycolmap.ImageReaderOptions()
    if calibration:
        opts.camera_model = calibration["model"]
        opts.camera_params = ",".join(str(v) for v in calibration["params"])
    else:
        opts.camera_model = params.get("camera_model", "SIMPLE_RADIAL")
    return opts


def run_colmap(image_dir: Path, image_names: list[str], out_dir: Path, params: dict,
               calibration: dict | None = None) -> pycolmap.Reconstruction:
    """Run feature extraction, matching and mapping. Returns the largest model."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    db = out_dir / "database.db"

    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = int(params["max_image_size"])
    ext.sift.max_num_features = int(params["max_num_features"])
    pycolmap.extract_features(
        db, image_dir, image_names=image_names, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=_reader_options(params, calibration), extraction_options=ext,
        device=pycolmap.Device.cpu,
    )

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = int(params["sequential_overlap"])
    pycolmap.match_sequential(db, pairing_options=pairing, device=pycolmap.Device.cpu)

    models_dir = out_dir / "models"
    models_dir.mkdir()
    if params.get("mapper", "incremental") == "global":
        models = pycolmap.global_mapping(db, image_dir, models_dir)
    else:
        opts = pycolmap.IncrementalPipelineOptions()
        if calibration:  # trust the supplied calibration
            opts.ba_refine_focal_length = False
            opts.ba_refine_principal_point = False
            opts.ba_refine_extra_params = False
        models = pycolmap.incremental_mapping(db, image_dir, models_dir, options=opts)
    if not models:
        raise RuntimeError("COLMAP produced no reconstruction (too little overlap or texture?)")
    best = max(models.values(), key=lambda r: r.num_reg_images())
    sparse = out_dir / "sparse" / "0"
    sparse.mkdir(parents=True)
    best.write(sparse)
    return best


def undistort(sparse_dir: Path, image_dir: Path, out_dir: Path) -> Path:
    """Undistort to PINHOLE (the camera model splat trainers expect). Returns the output dir."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pycolmap.undistort_images(output_path=out_dir, input_path=sparse_dir, image_path=image_dir)
    return out_dir


def reconstruction_metrics(rec: pycolmap.Reconstruction, n_input: int) -> dict:
    return {
        "images_input": n_input,
        "images_registered": rec.num_reg_images(),
        "registered_pct": round(100.0 * rec.num_reg_images() / max(1, n_input), 1),
        "points3d": rec.num_points3D(),
        "mean_reprojection_error_px": round(float(rec.compute_mean_reprojection_error()), 3),
        "mean_track_length": round(float(rec.compute_mean_track_length()), 2),
        "mean_observations_per_image": round(float(rec.compute_mean_observations_per_reg_image()), 1),
        "camera": {c.camera_id: {"model": c.model_name, "params": [round(float(p), 4) for p in c.params]}
                   for c in rec.cameras.values()},
    }


def camera_trajectory(rec: pycolmap.Reconstruction, name_to_t: dict[str, float]) -> np.ndarray:
    """Rows t, tx, ty, tz, qx, qy, qz, qw (camera-to-world), sorted by time."""
    rows = []
    for img in rec.images.values():
        if not img.has_pose or img.name not in name_to_t:
            continue
        world_from_cam = img.cam_from_world().inverse()
        q = world_from_cam.rotation.quat  # x, y, z, w
        c = world_from_cam.translation
        rows.append([name_to_t[img.name], *c, *q])
    rows.sort(key=lambda r: r[0])
    return np.array(rows, dtype=float).reshape(-1, 8)


def point_arrays(rec: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    """xyz, rgb, track length and reprojection error for every 3D point."""
    ids = list(rec.points3D.keys())
    pts = [rec.points3D[i] for i in ids]
    return {
        "ids": np.array(ids, dtype=np.int64),
        "xyz": np.array([p.xyz for p in pts], dtype=np.float64).reshape(-1, 3),
        "rgb": np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3),
        "track_len": np.array([p.track.length() for p in pts], dtype=np.int32),
        "error": np.array([p.error for p in pts], dtype=np.float32),
    }


def observing_centers(rec: pycolmap.Reconstruction) -> list[np.ndarray]:
    """For each 3D point (same order as point_arrays), the centres of the cameras that see it."""
    centers = {iid: img.projection_center() for iid, img in rec.images.items() if img.has_pose}
    out = []
    for pid in rec.points3D:
        els = rec.points3D[pid].track.elements
        out.append(np.array([centers[e.image_id] for e in els if e.image_id in centers]).reshape(-1, 3))
    return out
