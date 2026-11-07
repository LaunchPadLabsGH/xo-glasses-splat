"""Gaussian Splatting training on a local CUDA GPU, using gsplat (Apache-2.0).

This is X-O Labs' own minimal trainer built on gsplat's rasteriser and
densification strategy. It deliberately does not use the original Inria 3DGS
code, which is licensed for non-commercial research only.

Inputs: the undistorted (PINHOLE) COLMAP model from the SfM stage.
Outputs: a standard 3DGS .ply, held-out PSNR/SSIM, and side-by-side renders.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


def _torch():
    import torch

    return torch


def gaussian_window(size: int = 11, sigma: float = 1.5):
    torch = _torch()
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(img1, img2, window=None):
    """SSIM for (B, C, H, W) tensors in [0, 1] (Wang et al. 2004, Gaussian window)."""
    torch = _torch()
    import torch.nn.functional as F

    c = img1.shape[1]
    if window is None:
        window = gaussian_window().to(img1.device)
    w = window.expand(c, 1, -1, -1)
    pad = w.shape[-1] // 2
    mu1 = F.conv2d(img1, w, padding=pad, groups=c)
    mu2 = F.conv2d(img2, w, padding=pad, groups=c)
    s11 = F.conv2d(img1 * img1, w, padding=pad, groups=c) - mu1 ** 2
    s22 = F.conv2d(img2 * img2, w, padding=pad, groups=c) - mu2 ** 2
    s12 = F.conv2d(img1 * img2, w, padding=pad, groups=c) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2)
    return torch.mean(num / den)


def photometric_loss(render, target, ssim_lambda: float = 0.2):
    """(1 - lambda) * L1 + lambda * (1 - SSIM), on (H, W, 3) images."""
    torch = _torch()
    l1 = torch.abs(render - target).mean()
    s = ssim(render.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
    return (1 - ssim_lambda) * l1 + ssim_lambda * (1 - s)


def load_views(undistorted_dir: Path, downscale: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Registered views: image (H, W, 3) float32, K (3, 3), viewmat (4, 4) world-to-camera."""
    import cv2
    import pycolmap

    rec = pycolmap.Reconstruction(str(Path(undistorted_dir) / "sparse"))
    views = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        if not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        if cam.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"expected undistorted pinhole cameras, got {cam.model_name}")
        bgr = cv2.imread(str(Path(undistorted_dir) / "images" / img.name), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        if downscale > 1:
            w2, h2 = w // downscale, h // downscale
            bgr = cv2.resize(bgr, (w2, h2), interpolation=cv2.INTER_AREA)
        sx, sy = bgr.shape[1] / w, bgr.shape[0] / h
        K = cam.calibration_matrix().copy()
        K[0] *= sx
        K[1] *= sy
        vm = np.eye(4)
        vm[:3, :4] = img.cam_from_world().matrix()
        views.append({"name": img.name, "image": bgr[:, :, ::-1].astype(np.float32) / 255.0,
                      "K": K.astype(np.float32), "viewmat": vm.astype(np.float32)})
    points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    colors = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0
    return views, points, colors


def split_views(views: list[dict], test_every: int) -> tuple[list[dict], list[dict]]:
    if test_every <= 0:
        return views, []
    test = [v for i, v in enumerate(views) if i % test_every == 0]
    train = [v for i, v in enumerate(views) if i % test_every != 0]
    return train, test


def scene_scale(views: list[dict]) -> float:
    centers = np.array([np.linalg.inv(v["viewmat"])[:3, 3] for v in views])
    return float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)) * 1.1) or 1.0


def init_params(points: np.ndarray, colors: np.ndarray, sh_degree: int, scale: float, device: str):
    torch = _torch()
    from scipy.spatial import cKDTree

    from xgs.splat.gaussians import rgb_to_sh0

    n = len(points)
    k = min(4, n)
    d, _ = cKDTree(points).query(points, k=k)
    d = np.asarray(d).reshape(n, -1)[:, 1:] if k > 1 else np.ones((n, 1))
    dist = np.sqrt(np.mean(d ** 2, axis=1)).clip(1e-7)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    n_sh = (sh_degree + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(t(points)),
        "scales": torch.nn.Parameter(t(np.log(np.repeat(dist[:, None], 3, axis=1)))),
        "quats": torch.nn.Parameter(t(np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)))),
        "opacities": torch.nn.Parameter(t(np.full(n, math.log(0.1 / 0.9)))),
        "sh0": torch.nn.Parameter(t(rgb_to_sh0(colors))[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros((n, n_sh - 1, 3), device=device)),
    })
    lrs = {"means": 1.6e-4 * scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    return params, optimizers


def render(params, view_K, view_vm, width: int, height: int, sh_degree: int, background=None):
    torch = _torch()
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=view_vm[None],
        Ks=view_K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        backgrounds=background,
    )
    return out[0].clamp(0, 1), alpha[0], info


def to_gaussians(params):
    from xgs.splat.gaussians import Gaussians

    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    q = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    return Gaussians(p["means"], p["scales"], q, p["opacities"], p["sh0"][:, 0, :], p["shN"])


def train(undistorted_dir: Path, out_dir: Path, cfg: dict, log=print) -> dict:
    """Train a splat model; returns metrics. Requires CUDA."""
    torch = _torch()
    from gsplat.strategy import DefaultStrategy

    from xgs.splat.gaussians import write_ply

    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available: run `xgs train` on the RTX machine "
                           "(see scripts/setup_wsl_gpu.sh)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    views, points, colors = load_views(undistorted_dir, int(cfg["downscale"]))
    train_v, test_v = split_views(views, int(cfg["test_every"]))
    scale = scene_scale(views)
    log(f"{len(train_v)} train / {len(test_v)} test views, {len(points)} init points, scene scale {scale:.3f}")

    iters = int(cfg["iterations"])
    sh_degree = int(cfg["sh_degree"])
    params, optimizers = init_params(points, colors, sh_degree, scale, device)
    strategy = DefaultStrategy(refine_stop_iter=min(15000, int(0.8 * iters)), reset_every=3000,
                               refine_start_iter=500, refine_every=100, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / iters))

    tv = [{"img": torch.tensor(v["image"], device=device), "K": torch.tensor(v["K"], device=device),
           "vm": torch.tensor(v["viewmat"], device=device)} for v in train_v]
    t0 = time.time()
    order = rng.permutation(len(tv))
    for step in range(iters):
        if step % len(tv) == 0 and step:
            order = rng.permutation(len(tv))
        v = tv[order[step % len(tv)]]
        h, w = v["img"].shape[:2]
        bg = torch.rand(1, 3, device=device) if cfg.get("random_background") else None
        img, _, info = render(params, v["K"], v["vm"], w, h, min(step // 1000, sh_degree), bg)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss = photometric_loss(img, v["img"])
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 500 == 0 or step == iters - 1:
            log(f"step {step:>6}/{iters}  loss {loss.item():.4f}  gaussians {len(params['means']):,}")
    train_s = time.time() - t0

    # Evaluate on held-out views and save side-by-side renders.
    import cv2

    psnrs, ssims = [], []
    with torch.no_grad():
        for i, v in enumerate(test_v):
            gt = torch.tensor(v["image"], device=device)
            h, w = gt.shape[:2]
            img, _, _ = render(params, torch.tensor(v["K"], device=device),
                               torch.tensor(v["viewmat"], device=device), w, h, sh_degree)
            mse = torch.mean((img - gt) ** 2).item()
            psnrs.append(10 * math.log10(1.0 / max(mse, 1e-10)))
            ssims.append(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item())
            if i < 6:
                pair = np.concatenate([v["image"], img.cpu().numpy()], axis=1)
                cv2.imwrite(str(out_dir / f"heldout_{i:02d}.jpg"),
                            (pair[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8))
    write_ply(out_dir / "scene.ply", to_gaussians(params))
    metrics = {
        "iterations": iters,
        "train_time_s": round(train_s, 1),
        "gaussians": int(len(params["means"])),
        "train_views": len(train_v),
        "test_views": len(test_v),
        "test_psnr_db": round(float(np.mean(psnrs)), 2) if psnrs else None,
        "test_ssim": round(float(np.mean(ssims)), 4) if ssims else None,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                            if device.startswith("cuda") else None),
        "downscale": int(cfg["downscale"]),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
