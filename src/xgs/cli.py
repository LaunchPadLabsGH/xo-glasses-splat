"""Command-line interface: `xgs run ...` for the CPU pipeline, `xgs train` / `xgs export` for the GPU path."""

from __future__ import annotations

from pathlib import Path

import typer

from xgs import pipeline
from xgs.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Smart-glasses footage to live maps and Gaussian Splatting models.")

RunOpt = typer.Option(..., "--run", "-r", help="Run directory, e.g. runs/demo")
ConfigOpt = typer.Option(None, "--config", "-c", help="Extra YAML config merged over configs/default.yaml")
DeviceOpt = typer.Option(None, "--device", "-d", help="Device profile name (configs/devices/*.yaml)")
SetOpt = typer.Option([], "--set", "-s", help="Override, e.g. --set select.max_keyframes=200")


def _cfg(config: Path | None, device: str | None, overrides: list[str]) -> dict:
    return load_config(config, device, overrides)


@app.command()
def run(
    source: Path = typer.Argument(..., help="Video file, image folder, or TUM sequence folder"),
    run_dir: Path = RunOpt,
    config: Path | None = ConfigOpt,
    device: str | None = DeviceOpt,
    imu: Path | None = typer.Option(None, help="Optional IMU CSV (see xgs.io.imu)"),
    until: str | None = typer.Option(None, help=f"Stop after this stage: {', '.join(pipeline.CPU_STAGES)}"),
    force: bool = typer.Option(False, help="Ignore cached stages"),
    overrides: list[str] = SetOpt,
) -> None:
    """Run all CPU stages: ingest, keyframes, privacy, SfM, confidence, live replay, audio, preview, evaluate."""
    pipeline.run_cpu(run_dir, _cfg(config, device, overrides), source, imu, until, force)


@app.command()
def train(
    run_dir: Path = RunOpt,
    config: Path | None = ConfigOpt,
    device: str | None = DeviceOpt,
    force: bool = typer.Option(False, help="Retrain even if cached"),
    overrides: list[str] = SetOpt,
) -> None:
    """Train the Gaussian Splatting model on a CUDA GPU (post-mission, local hardware)."""
    from xgs.splat.stage import train_stage

    train_stage(run_dir, _cfg(config, device, overrides), force)


@app.command()
def export(
    run_dir: Path = RunOpt,
    name: str = typer.Option(..., "--name", "-n", help="Scene name under docs/assets/"),
    config: Path | None = ConfigOpt,
    device: str | None = DeviceOpt,
    overrides: list[str] = SetOpt,
) -> None:
    """Export web assets (splats, trajectory, events, metrics) for the browser viewer."""
    from xgs.report.export import export_scene

    export_scene(run_dir, _cfg(config, device, overrides), name)


@app.command()
def report(run_dir: Path = RunOpt) -> None:
    """Write results.json, results.md and plots for a run."""
    from xgs.report.build import build_report

    build_report(run_dir)


if __name__ == "__main__":
    app()
