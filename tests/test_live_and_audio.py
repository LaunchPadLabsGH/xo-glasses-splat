import wave

import cv2
import numpy as np

from xgs.audio.events import detect_events, pin_to_trajectory, read_wav_mono
from xgs.config import load_config
from xgs.io.frames import Frame
from xgs.live.replay import latency_summary, replay


def _render_points(K, R_cw, t_cw, pts, colors, size):
    """Draw 3D points as small coloured discs: a synthetic textured scene."""
    img = np.zeros((size[1], size[0], 3), np.uint8)
    pc = pts @ R_cw.T + t_cw
    ok = pc[:, 2] > 0.1
    uv = (pc[ok] @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]
    for (u, v), c in zip(uv.astype(int), colors[ok], strict=True):
        if 0 <= u < size[0] and 0 <= v < size[1]:
            cv2.circle(img, (int(u), int(v)), 3, tuple(int(x) for x in c), -1)
    return img


def test_live_replay_tracks_sideways_motion(tmp_path, rng):
    K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
    pts = np.c_[rng.uniform(-6, 6, 4000), rng.uniform(-3, 3, 4000), rng.uniform(4, 9, 4000)]
    colors = rng.integers(40, 255, size=(4000, 3))
    frames = []
    for i in range(25):
        img = _render_points(K, np.eye(3), np.array([-0.08 * i, 0, 0]), pts, colors, (640, 480))
        name = f"{i:06d}.png"
        cv2.imwrite(str(tmp_path / name), img)
        frames.append(Frame(i, i / 10, name))
    params = {**load_config()["live"], "min_inliers": 40, "min_triangulated": 20}
    res = replay(tmp_path, frames, params, K, None, 640)
    assert len(res.trajectory) >= 3 and len(res.points) > 100
    x = res.trajectory[:, 1]
    assert np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)       # consistent direction of travel
    lat = latency_summary(res, ingest_fps=10)
    assert lat["frames_processed"] == 25 and lat["latency_ms_p50"] > 0


def test_audio_events_detected_and_pinned(tmp_path):
    sr = 16000
    x = 0.01 * np.random.default_rng(0).normal(size=sr * 10)
    x[sr * 4: sr * 4 + 1600] += 0.8 * np.sin(np.arange(1600) / 3)     # loud event at 4 s
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    y, sr2 = read_wav_mono(p)
    ev = detect_events(y, sr2)
    assert len(ev) == 1 and abs(ev[0]["t"] - 4.0) < 0.1
    traj = np.array([[3.9, 1, 2, 3, 0, 0, 0, 1], [8, 0, 0, 0, 0, 0, 0, 1]])
    assert pin_to_trajectory(ev, traj)[0]["position"] == [1.0, 2.0, 3.0]
