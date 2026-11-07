"""Audio events pinned to the map (supports the audio desirable).

Detects short, loud events (onsets well above the background level) in the
glasses' audio track and places each one at the camera position at that time,
so an analyst can see where on the route something was heard. Speech is never
transcribed; only timing and loudness are kept.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        ch = w.getnchannels()
    dtype = {2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64) / np.iinfo(dtype).max
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def detect_events(x: np.ndarray, sr: int, hop_s: float = 0.05, bg_window_s: float = 5.0,
                  threshold_db: float = 12.0, min_gap_s: float = 1.0) -> list[dict]:
    """Frames whose level exceeds the rolling-median background by `threshold_db`."""
    hop = max(1, int(sr * hop_s))
    n = len(x) // hop
    if n == 0:
        return []
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    half = max(1, int(bg_window_s / hop_s / 2))
    bg = np.array([np.median(db[max(0, i - half): i + half + 1]) for i in range(n)])
    above = db - bg
    events: list[dict] = []
    for i in np.argsort(-above):
        if above[i] < threshold_db:
            break
        t = i * hop_s
        if all(abs(t - e["t"]) >= min_gap_s for e in events):
            events.append({"t": round(float(t), 2), "level_db_above_bg": round(float(above[i]), 1)})
    return sorted(events, key=lambda e: e["t"])


def pin_to_trajectory(events: list[dict], traj: np.ndarray, max_dt: float = 1.0) -> list[dict]:
    """Attach the nearest camera position (rows t, tx, ty, tz, ...) to each event."""
    if len(traj) == 0:
        return []
    out = []
    for e in events:
        j = int(np.argmin(np.abs(traj[:, 0] - e["t"])))
        if abs(traj[j, 0] - e["t"]) <= max_dt:
            out.append({**e, "position": [round(float(v), 4) for v in traj[j, 1:4]]})
    return out
