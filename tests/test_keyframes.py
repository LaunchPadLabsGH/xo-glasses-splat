import cv2
import numpy as np
from conftest import textured

from xgs.config import load_config
from xgs.io.frames import Frame
from xgs.io.imu import ImuData
from xgs.select.keyframes import rolling_median, select_keyframes, sharpness, summarise


def test_sharpness_drops_with_blur():
    img = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    assert sharpness(cv2.GaussianBlur(img, (15, 15), 5)) < 0.2 * sharpness(img)


def test_rolling_median():
    assert rolling_median(np.array([1, 100, 1, 1, 1.0]), 1).tolist() == [50.5, 1, 1, 1, 1]


def _sequence(tmp_path, n=30, blur_at=(10,), shift_px=6):
    """A camera panning across a large textured canvas; some frames blurred."""
    canvas = textured(480, 2000, seed=3)
    frames = []
    for i in range(n):
        img = canvas[100:340, i * shift_px: i * shift_px + 320].copy()
        if i in blur_at:
            img = cv2.blur(img, (25, 1))
        name = f"{i:06d}.jpg"
        cv2.imwrite(str(tmp_path / name), img)
        frames.append(Frame(i, i / 10.0, name))
    return frames


def test_select_rejects_blur_and_keeps_spaced_keyframes(tmp_path):
    frames = _sequence(tmp_path)
    params = load_config()["select"]
    scores = select_keyframes(tmp_path, frames, params)
    by_idx = {s.idx: s for s in scores}
    assert by_idx[10].status == "rejected" and by_idx[10].reason == "blur"
    s = summarise(scores)
    assert 2 <= s["keyframes"] < len(frames)
    assert s["frames_in"] == len(frames)
    # every frame has an explicit outcome
    assert all(x.status in {"keyframe", "rejected", "skipped"} for x in scores)


def test_select_respects_budget(tmp_path):
    frames = _sequence(tmp_path, n=40, blur_at=())
    params = {**load_config()["select"], "min_parallax_frac": 0.001, "max_keyframes": 5}
    assert summarise(select_keyframes(tmp_path, frames, params))["keyframes"] == 5


def test_imu_rejects_fast_head_turns(tmp_path):
    frames = _sequence(tmp_path, n=20, blur_at=())
    t = np.linspace(0, 2, 400)
    gyro = np.zeros((400, 3))
    gyro[(t > 0.95) & (t < 1.05), 1] = np.radians(300)   # 300 deg/s spike around frame 10
    scores = select_keyframes(tmp_path, frames, load_config()["select"], ImuData(t, gyro))
    assert {s.idx: s for s in scores}[10].reason == "fast_head_turn"
