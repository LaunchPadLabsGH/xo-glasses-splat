"""CPU checks of the GPU trainer's pieces (the rasteriser itself needs CUDA)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xgs.splat.train import init_params, photometric_loss, scene_scale, split_views, ssim  # noqa: E402


def test_loss_and_ssim():
    a = torch.rand(32, 40, 3)
    assert photometric_loss(a, a).item() < 1e-6
    assert photometric_loss(a, torch.rand(32, 40, 3)).item() > 0.1
    assert abs(ssim(a.permute(2, 0, 1)[None], a.permute(2, 0, 1)[None]).item() - 1) < 1e-5


def test_split_and_scale():
    views = [{"viewmat": np.eye(4, dtype=np.float32)} for _ in range(16)]
    for i, v in enumerate(views):
        v["viewmat"][0, 3] = -float(i)
    train, test = split_views(views, 8)
    assert len(test) == 2 and len(train) == 14
    assert scene_scale(views) > 7


def test_init_params_shapes_and_strategy():
    gsplat = pytest.importorskip("gsplat")
    pts = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    params, opts = init_params(pts, np.full((100, 3), 0.5, np.float32), 3, 1.0, "cpu")
    assert params["shN"].shape == (100, 15, 3) and params["sh0"].shape == (100, 1, 3)
    gsplat.strategy.DefaultStrategy().check_sanity(params, opts)
