"""Export a run as web assets for the browser viewer (docs/index.html).

Everything is moved into one viewer frame: centred on the camera path, +Y up
(estimated from the average camera orientation, assuming a roughly level head)
and, where ground truth exists, scaled to metres. Layers:
- refined:    trained Gaussian Splatting model (GPU stage), pruned for the web
- live:       the live map from the on-device replay (longest tracked segment)
- sparse:     SfM points as splats (training-free)
- confidence: SfM points coloured red (weak) to green (well observed)
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from xgs.config import REPO_ROOT
from xgs.io import frames as fio
from xgs.manifest import read_manifest
from xgs.pipeline import _say
from xgs.splat import gaussians as gs


def viewer_frame(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (world -> viewer) and centre from camera poses (rows t, xyz, qxyzw)."""
    R_wc = Rotation.from_quat(traj[:, 4:8]).as_matrix()
    up = -R_wc[:, :, 1].mean(axis=0)            # image y points down
    up /= np.linalg.norm(up)
    fwd = R_wc[:, :, 2].mean(axis=0)
    fwd -= up * (fwd @ up)
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.cross(up, [1.0, 0, 0])
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    R = np.stack([x, up, z])                     # rows: viewer axes in world coordinates
    return R, traj[:, 1:4].mean(axis=0)


def transform_gaussians(g: gs.Gaussians, R: np.ndarray, c: np.ndarray, s: float) -> gs.Gaussians:
    q_align = Rotation.from_matrix(R)
    q = Rotation.from_quat(g.quats[:, [1, 2, 3, 0]])  # wxyz -> xyzw
    q_new = (q_align * q).as_quat()[:, [3, 0, 1, 2]]
    # Higher-order SH coefficients would need rotating too; the web format keeps
    # only the DC (view-independent) colour, so they are dropped here.
    return gs.Gaussians(s * (g.means - c) @ R.T, g.log_scales + np.log(s), q_new, g.opacity_logit, g.sh0)


def _metrics(run: Path) -> dict:
    out = {}
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        m = read_manifest(d)
        if m and m.get("status") == "ok":
            out[d.name] = m.get("metrics", {})
    return out


def export_scene(run: Path, cfg: dict, name: str, web_root: Path | None = None) -> Path:
    from xgs.confidence import confidence_colors
    from xgs.sfm.scale import umeyama

    run = Path(run)
    out = Path(web_root or REPO_ROOT / cfg["export"]["web_dir"]) / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    metrics = _metrics(run)
    traj = fio.read_trajectory_csv(run / "sfm" / "trajectory.csv")
    R, c = viewer_frame(traj)
    sfm_ate = metrics.get("evaluate", {}).get("sfm_ate", {})
    s = float(sfm_ate.get("sim3_scale", 1.0)) if "ate_rmse_m" in sfm_ate else 1.0
    units = "metres (scaled by alignment to ground truth)" if s != 1.0 else \
        "arbitrary (monocular reconstruction has no absolute scale)"
    to_view = lambda x: s * (np.asarray(x) - c) @ R.T  # noqa: E731
    layers = []
    p = cfg["splat"]

    refined = run / "splat" / "scene.ply"
    if refined.exists():
        g = gs.prune(gs.read_ply(refined), p["prune_opacity"])
        n = gs.write_splat(out / "refined.splat", transform_gaussians(g, R, c, s), p["web_max_splats"])
        layers.append({"id": "refined", "label": "Refined (Gaussian Splatting)", "file": "refined.splat",
                       "count": n})

    lm_path = run / "live" / "live_map.npz"
    live_traj_view = []
    if lm_path.exists():
        lm = np.load(lm_path)
        lt = fio.read_trajectory_csv(run / "live" / "trajectory.csv")
        seg = np.loadtxt(run / "live" / "segments.csv", skiprows=1, ndmin=1).astype(int)
        if len(lt) >= 3 and len(lm["xyz"]):
            ids, counts = np.unique(seg, return_counts=True)
            best = ids[np.argmax(counts)]
            seg_traj = lt[seg == best]
            from xgs.eval.metrics import associate

            ie, ig = associate(seg_traj[:, 0], traj[:, 0], max_dt=0.05)
            if len(ie) >= 3:
                tf = umeyama(seg_traj[ie, 1:4], traj[ig, 1:4])
                mask = lm["segment"] == best
                pts = to_view(tf.apply(lm["xyz"][mask]))
                if len(pts) > 10:
                    g = gs.from_points(pts, lm["rgb"][mask])
                    n = gs.write_splat(out / "live.splat", g)
                    layers.append({"id": "live", "label": "Live map (on-device replay)", "file": "live.splat",
                                   "count": n})
                live_traj_view = [{"t": round(float(r[0]), 3), "p": [round(float(v), 4) for v in
                                   to_view(tf.apply(r[1:4][None]))[0]]} for r in seg_traj]

    pts = np.load(run / "sfm" / "points.npz")
    xyz_v = to_view(pts["xyz"])
    gs.write_splat(out / "sparse.splat", gs.from_points(xyz_v, pts["rgb"]))
    layers.append({"id": "sparse", "label": "Sparse map (SfM points)", "file": "sparse.splat",
                   "count": int(len(xyz_v))})
    conf_path = run / "confidence" / "confidence.npz"
    if conf_path.exists():
        conf = np.load(conf_path)["confidence"]
        gs.write_splat(out / "confidence.splat", gs.from_points(xyz_v, confidence_colors(conf)))
        layers.append({"id": "confidence", "label": "Confidence (red = weakly observed)",
                       "file": "confidence.splat", "count": int(len(xyz_v))})

    cams = to_view(traj[:, 1:4])
    R_wc = Rotation.from_quat(traj[:, 4:8]).as_matrix()
    look = (R @ R_wc[:, :, 2].T).T
    trajectory = [{"t": round(float(t), 3), "p": [round(float(v), 4) for v in p_],
                   "d": [round(float(v), 4) for v in d_]} for t, p_, d_ in zip(traj[:, 0], cams, look, strict=True)]
    (out / "trajectory.json").write_text(json.dumps({"sfm": trajectory, "live": live_traj_view}))

    ev_path = run / "audio" / "events.json"
    events = json.loads(ev_path.read_text()) if ev_path.exists() else []
    for e in events:
        e["position"] = [round(float(v), 4) for v in to_view(np.array(e["position"])[None])[0]]
    (out / "events.json").write_text(json.dumps(events))

    report_dir = run / "report"
    for png in sorted(report_dir.glob("*.png")) if report_dir.exists() else []:
        shutil.copy2(png, out / png.name)

    for i, img in enumerate(sorted((run / "splat").glob("heldout_*.jpg"))[:3] if (run / "splat").exists() else []):
        shutil.copy2(img, out / f"heldout_{i}.jpg")

    extent = float(np.percentile(np.linalg.norm(xyz_v, axis=1), 90)) if len(xyz_v) else 5.0
    first = trajectory[0]
    meta = {
        "name": name,
        "source": cfg["device_profile"].get("name"),
        "units": units,
        "extent": round(extent, 3),
        "layers": layers,
        "start": {"position": first["p"], "direction": first["d"]},
        "metrics": metrics,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "scene.json").write_text(json.dumps(meta, indent=1, default=str))
    _update_index(out.parent, name, cfg["device_profile"].get("name"))
    _say("export", f"{len(layers)} layers -> {out}")
    return out


def _update_index(web_dir: Path, name: str, source: str | None) -> None:
    """docs/assets/scenes.json lists the scenes the viewer can open."""
    idx_path = web_dir / "scenes.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"scenes": []}
    idx["scenes"] = [s for s in idx["scenes"] if s["name"] != name] + [{"name": name, "source": source}]
    idx["scenes"].sort(key=lambda s: s["name"])
    idx_path.write_text(json.dumps(idx, indent=1))
