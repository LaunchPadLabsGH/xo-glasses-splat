"""Frame and audio extraction from video files using ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} not found on PATH; install ffmpeg (e.g. `sudo apt install ffmpeg`)")


def probe(path: Path) -> dict:
    """Return basic stream info: width, height, fps, duration, has_audio."""
    require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    video = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    num, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0.0)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration_s": duration,
        "codec": video.get("codec_name"),
        "has_audio": any(s.get("codec_type") == "audio" for s in info["streams"]),
    }


def extract_frames(
    src: Path, out_dir: Path, fps: float, max_width: int, start_s: float = 0.0,
    duration_s: float | None = None,
) -> list[tuple[int, float, str]]:
    """Extract frames at `fps` as JPEGs. Returns (index, timestamp_s, filename) per frame.

    Timestamps are relative to the start of the source video.
    """
    require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start_s:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(src)]
    if duration_s:
        cmd += ["-t", f"{duration_s:.3f}"]
    vf = f"fps={fps},scale='min({max_width},iw)':-2"
    cmd += ["-vf", vf, "-q:v", "2", "-start_number", "0", str(out_dir / "%06d.jpg")]
    subprocess.run(cmd, check=True)
    files = sorted(out_dir.glob("*.jpg"))
    return [(i, start_s + i / fps, f.name) for i, f in enumerate(files)]


def extract_audio(src: Path, out_wav: Path, start_s: float = 0.0, duration_s: float | None = None,
                  sample_rate: int = 16000) -> bool:
    """Extract mono 16 kHz audio. Returns False if the source has no audio."""
    require_ffmpeg()
    if not probe(src)["has_audio"]:
        return False
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start_s:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(src)]
    if duration_s:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), str(out_wav)]
    subprocess.run(cmd, check=True)
    return out_wav.exists()
