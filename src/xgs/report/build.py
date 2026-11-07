"""Collect a run's stage manifests into results.json / results.md and plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from xgs.io import frames as fio
from xgs.manifest import read_manifest
from xgs.pipeline import _say

STAGES = ["ingest", "select", "privacy", "sfm", "confidence", "live", "audio", "preview", "evaluate", "splat"]


def collect(run: Path) -> dict:
    out = {}
    for s in STAGES:
        m = read_manifest(run / s)
        if m and m.get("status") == "ok":
            out[s] = {"metrics": m["metrics"], "duration_s": m["duration_s"], "versions": m.get("versions", {})}
    return out


def _row(label: str, value) -> str:
    return f"| {label} | {value} |"


def markdown(run: Path, res: dict) -> str:
    g = lambda s, k, d="–": res.get(s, {}).get("metrics", {}).get(k, d)  # noqa: E731
    lines = [f"# Results: `{run.name}`", "", "| Measure | Value |", "|---|---|"]
    ing = res.get("ingest", {}).get("metrics", {})
    lines.append(_row("Input", f"{ing.get('frames', '–')} frames at {ing.get('width')}x{ing.get('height')}, "
                               f"{ing.get('duration_s')} s ({ing.get('kind')})"))
    lines.append(_row("Keyframes kept", f"{g('select', 'keyframes')} of {g('select', 'frames_in')} "
                                        f"(dropped: {g('select', 'drop_reasons', {})})"))
    lines.append(_row("Face detections blurred", g("privacy", "faces_blurred")))
    lines.append(_row("Images posed by SfM", f"{g('sfm', 'images_registered')} ({g('sfm', 'registered_pct')}%)"))
    lines.append(_row("Sparse 3D points", g("sfm", "points3d")))
    lines.append(_row("Mean reprojection error", f"{g('sfm', 'mean_reprojection_error_px')} px"))
    lines.append(_row("Mean track length", g("sfm", "mean_track_length")))
    lines.append(_row("Low-confidence points", f"{g('confidence', 'low_confidence_pct')}%"))
    if "live" in res:
        lines.append(_row("Live front end latency p50 / p95",
                          f"{g('live', 'latency_ms_p50')} / {g('live', 'latency_ms_p95')} ms (CPU proxy)"))
        lines.append(_row("Live frames within 30 fps budget", f"{g('live', 'pct_within_30fps_budget')}%"))
        lines.append(_row("Live map resets (tracking lost)", g("live", "map_resets")))
    ev = res.get("evaluate", {}).get("metrics", {})
    if "sfm_ate" in ev and "ate_rmse_m" in ev["sfm_ate"]:
        a = ev["sfm_ate"]
        lines.append(_row("Post-mission pose error (ATE RMSE)",
                          f"{a['ate_rmse_m'] * 100:.1f} cm over a {a['gt_path_length_m']} m path "
                          f"({a['ate_pct_of_path']}%)"))
    if "live_ate_longest_segment" in ev and "ate_rmse_m" in ev["live_ate_longest_segment"]:
        a = ev["live_ate_longest_segment"]
        lines.append(_row("Live pose drift, longest tracked segment",
                          f"{a['ate_rmse_m'] * 100:.1f} cm over {a['gt_path_length_m']} m ({a['ate_pct_of_path']}%)"))
    if "audio" in res:
        lines.append(_row("Audio events pinned to map", g("audio", "events")))
    if "splat" in res:
        lines.append(_row("Gaussian Splatting PSNR / SSIM (held-out views)",
                          f"{g('splat', 'test_psnr_db')} dB / {g('splat', 'test_ssim')}"))
        lines.append(_row("Splat training", f"{g('splat', 'iterations')} iterations, {g('splat', 'train_time_s')} s "
                                            f"on {g('splat', 'gpu')}, {g('splat', 'gaussians'):,} Gaussians"
                          if isinstance(g("splat", "gaussians"), int) else "–"))
    lines += ["", "Stage timings (s): " + ", ".join(f"{k} {v['duration_s']}" for k, v in res.items()), ""]
    return "\n".join(lines)


def plots(run: Path, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from xgs.eval.metrics import associate
    from xgs.sfm.scale import umeyama

    made = []
    gt_path = run / "ingest" / "groundtruth.csv"
    if gt_path.exists() and (run / "sfm" / "trajectory.csv").exists():
        gt = fio.read_trajectory_csv(gt_path)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(gt[:, 1], gt[:, 2], color="#222", lw=1.5, label="Ground truth (motion capture)")
        colors = {"sfm": "#1f77b4", "live": "#d62728"}
        labels = {"sfm": "Post-mission (SfM)", "live": "Live front end (longest segment)"}
        for name in ("sfm", "live"):
            p = run / name / "trajectory.csv"
            if not p.exists():
                continue
            est = fio.read_trajectory_csv(p)
            if name == "live" and (run / "live" / "segments.csv").exists():
                seg = np.loadtxt(run / "live" / "segments.csv", skiprows=1, ndmin=1).astype(int)
                ids, counts = np.unique(seg, return_counts=True)
                est = est[seg == ids[np.argmax(counts)]]
            ie, ig = associate(est[:, 0], gt[:, 0])
            if len(ie) < 3:
                continue
            a = umeyama(est[ie, 1:4], gt[ig, 1:4]).apply(est[ie, 1:4])
            ax.plot(a[:, 0], a[:, 1], ".-", ms=3, lw=1, color=colors[name], label=labels[name])
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Camera path vs ground truth (top view)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(out / "trajectory.png", dpi=150)
        plt.close(fig)
        made.append("trajectory.png")

    lat = run / "live" / "latency.csv"
    if lat.exists():
        d = np.loadtxt(lat, delimiter=",", skiprows=1, ndmin=2)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(d[:, 1], bins=40, color="#4c78a8")
        ax.axvline(1000 / 30, color="#d62728", ls="--", lw=1, label="30 fps budget (33 ms)")
        ax.set_xlabel("Live front-end processing time per frame (ms, CPU)")
        ax.set_ylabel("Frames")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "latency.png", dpi=150)
        plt.close(fig)
        made.append("latency.png")

    kf = run / "select" / "keyframes.csv"
    if kf.exists():
        with open(kf) as f:
            rows = list(csv.DictReader(f))
        t = np.array([float(r["t"]) for r in rows])
        s = np.array([float(r["sharpness"]) for r in rows])
        st = [r["status"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 3))
        for status, color, label in (("skipped", "#bbb", "Skipped (redundant)"), ("rejected", "#d62728", "Rejected"),
                                     ("keyframe", "#2ca02c", "Keyframe")):
            m = np.array([x == status for x in st])
            ax.scatter(t[m], s[m], s=8, color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Sharpness")
        ax.set_title("Keyframe selection")
        ax.legend(fontsize=8, markerscale=2)
        fig.tight_layout()
        fig.savefig(out / "keyframes.png", dpi=150)
        plt.close(fig)
        made.append("keyframes.png")
    return made


def build_report(run: Path) -> Path:
    run = Path(run)
    out = run / "report"
    out.mkdir(exist_ok=True)
    res = collect(run)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "results.md").write_text(markdown(run, res))
    made = plots(run, out)
    _say("report", f"results.md, results.json, {', '.join(made)} -> {out}")
    return out
