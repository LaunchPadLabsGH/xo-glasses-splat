"""Face anonymisation before any reconstruction.

Faces are detected with OpenCV's YuNet model (MIT licence, OpenCV Zoo) and
blurred in the keyframes that feed SfM and splat training, so no identifiable
face can end up in a 3D model. Vehicle number plates are NOT handled in this
proof of concept (documented limitation; see docs/PRIVACY.md).
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = "face_detection_yunet_2023mar.onnx"


def ensure_model(url: str, cache_dir: Path, sha256: str | None = None) -> Path:
    """Download the detector once into the cache, verifying its checksum."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / MODEL_NAME
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "xo-glasses-splat"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if sha256 and hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError(f"checksum mismatch for {url}")
        path.write_bytes(data)
    return path


class FaceBlurrer:
    def __init__(self, model_path: Path, score_threshold: float = 0.6, margin: float = 0.25):
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)
        self.margin = margin

    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return np.zeros((0, 4)) if faces is None else faces[:, :4]

    def blur(self, img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = img.shape[:2]
        for x, y, bw, bh in boxes:
            mx, my = bw * self.margin, bh * self.margin
            x0, y0 = int(max(0, x - mx)), int(max(0, y - my))
            x1, y1 = int(min(w, x + bw + mx)), int(min(h, y + bh + my))
            if x1 <= x0 or y1 <= y0:
                continue
            roi = out[y0:y1, x0:x1]
            # Pixelate then blur: irreversible and visually obvious.
            small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)),
                               interpolation=cv2.INTER_LINEAR)
            roi = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            k = max(3, ((x1 - x0) // 4) | 1)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        return out

    def process(self, src: Path, dst: Path) -> int:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        boxes = self.detect(img)
        cv2.imwrite(str(dst), self.blur(img, boxes) if len(boxes) else img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return len(boxes)
