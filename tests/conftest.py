import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def textured(h=240, w=320, seed=0):
    """A random, high-texture test image (uint8 BGR)."""
    import cv2

    r = np.random.default_rng(seed)
    img = (r.random((h // 8, w // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (3, 3), 0)
