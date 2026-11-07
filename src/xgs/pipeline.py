"""Pipeline stages. Each stage reads upstream outputs from the run directory,
writes its own sub-directory plus a manifest, and is skipped when cached.

    ingest -> select -> privacy -> sfm -> confidence --+-> preview -> export -> report
         \\-> live (on-device path replay) -------------/      ^
                                         splat (GPU) ----------/
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from rich.console import Console

from xgs.io import frames as fio
from xgs.manifest import Stage

console = Console()
CPU_STAGES = ["ingest", "select", "privacy", "sfm", "confidence", "live", "audio", "preview", "evaluate"]


def _say(stage: str, msg: str) -> None:
    console.print(f"[bold cyan]{stage:>10}[/] {msg}")


def _cached(stage: Stage, force: bool) -> bool:
    if stage.is_cached() and not force:
        _say(stage.name, "[dim]cached, skipping[/]")
        return True
    return False


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# --------------------------------------------------------------------------- ingest
def ingest(run: Path, cfg: dict, source: Path, force: bool = False) -> Stage:
    p = cfg["ingest"]
    dev = cfg["device_profile"]
    source = Path(source)
    src_id = {"path": str(source.resolve()), "size": source.stat().st_size if source.is_file() else None}
    st = Stage(run, "ingest", {"ingest": p, "device": dev, "source": src_id}, code=["xgs.io.frames", "xgs.io.video"])
    if _cached(st, force):
        return st
    with st.running():
        out = st.dir / "frames"
        gt = None
        has_audio = False
        if dev.get("kind") == "tum":
            frames, gt = fio.ingest_tum(source, out, p["fps"], p["max_width"], p["start_s"] or 0.0,
                                        p["duration_s"])
            kind = "tum"
        elif source.is_dir():
            frames = fio.ingest_image_folder(source, out, p["fps"], p.get("source_fps", 30.0), p["max_width"])
            kind = "images"
        else:
            from xgs.io.video import extract_audio, extract_frames, probe

            info = probe(source)
            raw = extract_frames(source, out, p["fps"], p["max_width"], p["start_s"] or 0.0, p["duration_s"])
            frames = [fio.Frame(i, t, f) for i, t, f in raw]
            if p.get("extract_audio", True):
                has_audio = extract_audio(source, st.dir / "audio.wav", p["start_s"] or 0.0, p["duration_s"])
            st.metrics["source_video"] = info
            kind = "video"
        if not frames:
            raise RuntimeError("no frames extracted")
        fio.write_frames_csv(st.dir / "frames.csv", frames)
        if gt is not None:
            fio.write_trajectory_csv(st.dir / "groundtruth.csv", gt)
        import cv2

        h, w = cv2.imread(str(out / frames[0].file)).shape[:2]
        st.metrics.update({"kind": kind, "frames": len(frames), "width": w, "height": h,
                           "duration_s": round(frames[-1].t - frames[0].t, 2), "fps": p["fps"],
                           "has_audio": has_audio, "has_groundtruth": gt is not None})
        _say("ingest", f"{len(frames)} frames at {w}x{h} ({kind})")
    return st


# --------------------------------------------------------------------------- select
def select(run: Path, cfg: dict, imu_path: Path | None = None, force: bool = False) -> Stage:
    from xgs.select.keyframes import scores_to_rows, select_keyframes, summarise

    p = cfg["select"]
    st = Stage(run, "select", {"select": p, "imu": str(imu_path) if imu_path else None}, ["ingest"],
               code=["xgs.select.keyframes", "xgs.io.imu"])
    if _cached(st, force):
        return st
    with st.running():
        frames = fio.read_frames_csv(run / "ingest" / "frames.csv")
        imu = None
        if imu_path:
            from xgs.io.imu import load_imu

            icfg = cfg["device_profile"].get("imu") or {}
            imu = load_imu(imu_path, icfg.get("format", "generic_csv"), icfg.get("gyro_units", "rad_s"))
        scores = select_keyframes(run / "ingest" / "frames", frames, p, imu)
        rows = scores_to_rows(scores)
        with open(st.dir / "keyframes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        st.metrics.update(summarise(scores))
        _say("select", f"{st.metrics['keyframes']} keyframes from {len(scores)} frames "
                       f"(dropped: {st.metrics['drop_reasons']})")
    return st


def _keyframes(run: Path) -> list[dict]:
    with open(run / "select" / "keyframes.csv") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "keyframe"]


# --------------------------------------------------------------------------- privacy
def privacy(run: Path, cfg: dict, force: bool = False) -> Stage:
    p = cfg["privacy"]
    st = Stage(run, "privacy", {"privacy": p}, ["select"], code=["xgs.privacy.faces"])
    if _cached(st, force):
        return st
    with st.running():
        out = st.dir / "images"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        kfs = _keyframes(run)
        src = run / "ingest" / "frames"
        faces_total, frames_with_faces = 0, 0
        if p["enabled"]:
            from xgs.privacy.faces import FaceBlurrer, ensure_model

            model = ensure_model(p["model_url"], Path.home() / ".cache" / "xgs", p.get("model_sha256"))
            blurrer = FaceBlurrer(model, p["score_threshold"], p["margin"])
            for k in kfs:
                n = blurrer.process(src / k["file"], out / k["file"])
                faces_total += n
                frames_with_faces += int(n > 0)
        else:
            for k in kfs:
                shutil.copy2(src / k["file"], out / k["file"])
        st.metrics.update({"enabled": p["enabled"], "images": len(kfs), "faces_blurred": faces_total,
                           "images_with_faces": frames_with_faces})
        _say("privacy", f"blurred {faces_total} faces in {frames_with_faces}/{len(kfs)} keyframes")
    return st


# --------------------------------------------------------------------------- sfm
def sfm(run: Path, cfg: dict, force: bool = False) -> Stage:
    from xgs.sfm import colmap

    p = cfg["sfm"]
    calib = cfg["device_profile"].get("calibration")
    st = Stage(run, "sfm", {"sfm": p, "calibration": calib}, ["privacy"], code=["xgs.sfm.colmap"])
    if _cached(st, force):
        return st
    with st.running():
        img_dir = run / "privacy" / "images"
        names = sorted(f.name for f in img_dir.glob("*.jpg"))
        _say("sfm", f"COLMAP on {len(names)} keyframes ({p['mapper']} mapper, CPU) ...")
        rec = colmap.run_colmap(img_dir, names, st.dir / "colmap", p, calib)
        colmap.undistort(st.dir / "colmap" / "sparse" / "0", img_dir, st.dir / "undistorted")
        name_to_t = {k["file"]: float(k["t"]) for k in _keyframes(run)}
        traj = colmap.camera_trajectory(rec, name_to_t)
        fio.write_trajectory_csv(st.dir / "trajectory.csv", traj)
        pts = colmap.point_arrays(rec)
        np.savez(st.dir / "points.npz", **pts)
        centers = colmap.observing_centers(rec)
        np.savez(st.dir / "observers.npz", *centers)
        st.metrics.update(colmap.reconstruction_metrics(rec, len(names)))
        _say("sfm", f"registered {st.metrics['images_registered']}/{len(names)} images, "
                    f"{st.metrics['points3d']} points, reproj {st.metrics['mean_reprojection_error_px']} px")
    return st


# --------------------------------------------------------------------------- confidence
def confidence(run: Path, cfg: dict, force: bool = False) -> Stage:
    from xgs.confidence import max_triangulation_angle_deg, point_confidence

    p = cfg["confidence"]
    st = Stage(run, "confidence", {"confidence": p}, ["sfm"], code=["xgs.confidence"])
    if _cached(st, force):
        return st
    with st.running():
        pts = np.load(run / "sfm" / "points.npz")
        obs = np.load(run / "sfm" / "observers.npz")
        centers = [obs[f"arr_{i}"] for i in range(len(obs.files))]
        angles = np.array([max_triangulation_angle_deg(x, c) for x, c in zip(pts["xyz"], centers, strict=True)])
        conf = point_confidence(pts["track_len"], angles, p["full_obs"], p["full_angle_deg"])
        np.savez(st.dir / "confidence.npz", confidence=conf, angle_deg=angles)
        low = conf < p["low_threshold"]
        st.metrics.update({
            "points": int(len(conf)),
            "low_confidence_pct": round(100.0 * float(low.mean()), 1) if len(conf) else None,
            "median_triangulation_angle_deg": round(float(np.median(angles)), 2) if len(angles) else None,
            "median_confidence": round(float(np.median(conf)), 3) if len(conf) else None,
        })
        _say("confidence", f"{st.metrics['low_confidence_pct']}% of points low-confidence")
    return st


# --------------------------------------------------------------------------- live
def _intrinsics(run: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray | None, int, str]:
    """Camera matrix for the live path: device calibration if known, else the SfM estimate."""
    calib = cfg["device_profile"].get("calibration")
    if calib:
        fx, fy, cx, cy, *d = calib["params"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        dist = np.array((list(d) + [0.0] * 5)[:5])  # OpenCV order: k1, k2, p1, p2, k3
        return K, dist, int(calib["width"]), "device calibration"
    m = json.loads((run / "sfm" / "manifest.json").read_text())["metrics"]
    cam = next(iter(m["camera"].values()))
    ing = json.loads((run / "ingest" / "manifest.json").read_text())["metrics"]
    prm = cam["params"]
    if cam["model"] in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        f, cx, cy = prm[:3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        dist = np.array([prm[3], 0, 0, 0, 0]) if cam["model"] == "SIMPLE_RADIAL" else None
    else:
        fx, fy, cx, cy = prm[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        dist = None
    return K, dist, int(ing["width"]), "SfM self-calibration (stand-in for factory calibration)"


def live(run: Path, cfg: dict, force: bool = False) -> Stage:
    from xgs.live.replay import latency_summary, replay

    p = cfg["live"]
    calibrated = bool(cfg["device_profile"].get("calibration"))
    st = Stage(run, "live", {"live": p}, ["ingest"] if calibrated else ["ingest", "sfm"],
               code=["xgs.live.replay"])
    if _cached(st, force):
        return st
    with st.running():
        frames = fio.read_frames_csv(run / "ingest" / "frames.csv")
        K, dist, width, k_source = _intrinsics(run, cfg)
        res = replay(run / "ingest" / "frames", frames, p, K, dist, width)
        fio.write_trajectory_csv(st.dir / "trajectory.csv", res.trajectory)
        np.savez(st.dir / "live_map.npz", xyz=res.points, rgb=res.colors, segment=res.point_segments)
        np.savetxt(st.dir / "segments.csv", res.segments, fmt="%d", header="segment", comments="")
        np.savetxt(st.dir / "latency.csv", np.c_[res.frame_t, res.latency_ms, res.is_keyframe],
                   delimiter=",", header="t,latency_ms,is_keyframe", comments="", fmt="%.6f")
        st.metrics.update(latency_summary(res, cfg["ingest"]["fps"]))
        st.metrics["intrinsics_source"] = k_source
        _say("live", f"p50 {st.metrics.get('latency_ms_p50')} ms/frame, {st.metrics.get('keyframes')} keyframes, "
                     f"{st.metrics.get('live_map_points')} map points")
    return st


# --------------------------------------------------------------------------- audio
def audio(run: Path, cfg: dict, force: bool = False) -> Stage:
    from xgs.audio.events import detect_events, pin_to_trajectory, read_wav_mono

    st = Stage(run, "audio", {"start_s": cfg["ingest"]["start_s"]}, ["ingest", "sfm"], code=["xgs.audio.events"])
    if _cached(st, force):
        return st
    with st.running():
        wav = run / "ingest" / "audio.wav"
        events: list[dict] = []
        if wav.exists():
            x, sr = read_wav_mono(wav)
            offset = float(cfg["ingest"]["start_s"] or 0.0)
            raw = [{**e, "t": round(e["t"] + offset, 2)} for e in detect_events(x, sr)]
            traj = fio.read_trajectory_csv(run / "sfm" / "trajectory.csv")
            events = pin_to_trajectory(raw, traj)
        _write_json(st.dir / "events.json", events)
        st.metrics.update({"has_audio": wav.exists(), "events": len(events)})
        _say("audio", f"{len(events)} audio events pinned to the map" if wav.exists() else "no audio track")
    return st


# --------------------------------------------------------------------------- preview
def preview(run: Path, cfg: dict, force: bool = False) -> Stage:
    """Training-free splats for the viewer: the live map, and SfM points coloured by confidence."""
    from xgs.confidence import confidence_colors
    from xgs.splat.gaussians import from_points, write_ply, write_splat

    st = Stage(run, "preview", {}, ["sfm", "confidence", "live"], code=["xgs.splat.gaussians"])
    if _cached(st, force):
        return st
    with st.running():
        pts = np.load(run / "sfm" / "points.npz")
        conf = np.load(run / "confidence" / "confidence.npz")["confidence"]
        g = from_points(pts["xyz"], pts["rgb"])
        write_ply(st.dir / "sparse.ply", g)
        write_splat(st.dir / "sparse.splat", g)
        gc = from_points(pts["xyz"], confidence_colors(conf))
        write_splat(st.dir / "confidence.splat", gc)
        n_live = 0
        lm_path = run / "live" / "live_map.npz"
        if lm_path.exists():
            lm = np.load(lm_path)
            n_live = int(len(lm["xyz"]))
            if n_live > 10:
                write_splat(st.dir / "live.splat", from_points(lm["xyz"], lm["rgb"]))
        st.metrics.update({"sparse_splats": len(g), "live_splats": n_live})
        _say("preview", f"sparse map {len(g)} splats, live map {n_live} splats")
    return st


# --------------------------------------------------------------------------- evaluate
def evaluate(run: Path, cfg: dict, force: bool = False) -> Stage:
    from xgs.eval.metrics import ate

    st = Stage(run, "evaluate", {}, ["ingest", "sfm", "live"], code=["xgs.eval.metrics"])
    if _cached(st, force):
        return st
    with st.running():
        gt_path = run / "ingest" / "groundtruth.csv"
        if gt_path.exists():
            gt = fio.read_trajectory_csv(gt_path)
            for name in ("sfm", "live"):
                if not (run / name / "trajectory.csv").exists():
                    continue
                est = fio.read_trajectory_csv(run / name / "trajectory.csv")
                st.metrics[f"{name}_ate"] = ate(est, gt) if len(est) >= 3 else {"error": "too few poses"}
            seg_path = run / "live" / "segments.csv"
            if seg_path.exists():
                seg = np.loadtxt(seg_path, skiprows=1, ndmin=1).astype(int)
                est = fio.read_trajectory_csv(run / "live" / "trajectory.csv")
                ids, counts = np.unique(seg, return_counts=True)
                longest = ids[np.argmax(counts)]
                m = ate(est[seg == longest], gt)
                m["poses"] = int(counts.max())
                m["segments"] = int(len(ids))
                st.metrics["live_ate_longest_segment"] = m
            _say("evaluate", "ATE " + ", ".join(
                f"{k} {v.get('ate_rmse_m')} m" for k, v in st.metrics.items() if "_ate" in k))
        else:
            st.metrics["note"] = "no ground truth for this sequence"
            _say("evaluate", "no ground truth: trajectory accuracy not computed")
    return st


def run_cpu(run: Path, cfg: dict, source: Path, imu: Path | None = None, until: str | None = None,
            force: bool = False) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    steps = {
        "ingest": lambda: ingest(run, cfg, source, force),
        "select": lambda: select(run, cfg, imu, force),
        "privacy": lambda: privacy(run, cfg, force),
        "sfm": lambda: sfm(run, cfg, force),
        "confidence": lambda: confidence(run, cfg, force),
        "live": lambda: live(run, cfg, force) if cfg["live"]["enabled"] else None,
        "audio": lambda: audio(run, cfg, force),
        "preview": lambda: preview(run, cfg, force),
        "evaluate": lambda: evaluate(run, cfg, force),
    }
    for name in CPU_STAGES:
        steps[name]()
        if name == until:
            break
