"""`xgs train`: the GPU stage, run on the local RTX machine after the CPU stages."""

from __future__ import annotations

from pathlib import Path

from xgs.manifest import Stage
from xgs.pipeline import _cached, _say


def train_stage(run: Path, cfg: dict, force: bool = False) -> Stage:
    from xgs.splat.train import train

    run = Path(run)
    p = cfg["splat"]
    st = Stage(run, "splat", {"splat": p}, ["sfm"], code=["xgs.splat.train", "xgs.splat.gaussians"])
    if _cached(st, force):
        return st
    with st.running():
        st.metrics.update(train(run / "sfm" / "undistorted", st.dir, p, log=lambda m: _say("splat", m)))
        st.outputs = ["scene.ply", "train_metrics.json"]
        _say("splat", f"PSNR {st.metrics['test_psnr_db']} dB on held-out views, "
                      f"{st.metrics['gaussians']:,} Gaussians, {st.metrics['train_time_s']} s")
    return st
