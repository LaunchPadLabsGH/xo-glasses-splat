import numpy as np

from xgs.splat.gaussians import (
    Gaussians,
    from_points,
    prune,
    read_ply,
    read_splat,
    rgb_to_sh0,
    sh0_to_rgb,
    sigmoid,
    write_ply,
    write_splat,
)


def _g(rng, n=50, sh=True):
    return Gaussians(
        means=rng.normal(size=(n, 3)), log_scales=rng.normal(-3, 0.2, size=(n, 3)),
        quats=np.tile([1.0, 0, 0, 0], (n, 1)), opacity_logit=rng.normal(size=n),
        sh0=rng.normal(size=(n, 3)) * 0.1, shN=rng.normal(size=(n, 15, 3)) if sh else None)


def test_sh0_roundtrip():
    rgb = np.array([[0.2, 0.5, 0.9]])
    assert np.allclose(sh0_to_rgb(rgb_to_sh0(rgb)), rgb)


def test_ply_roundtrip_with_sh(tmp_path, rng):
    g = _g(rng)
    write_ply(tmp_path / "s.ply", g)
    h = read_ply(tmp_path / "s.ply")
    for k in ("means", "log_scales", "quats", "opacity_logit", "sh0", "shN"):
        assert np.allclose(getattr(g, k), getattr(h, k), atol=1e-6), k


def test_splat_format(tmp_path, rng):
    g = _g(rng, sh=False)
    n = write_splat(tmp_path / "s.splat", g, max_splats=20)
    assert n == 20 and (tmp_path / "s.splat").stat().st_size == 20 * 32
    d = read_splat(tmp_path / "s.splat")
    imp = np.prod(np.exp(g.log_scales), axis=1) * sigmoid(g.opacity_logit)
    assert np.allclose(d["pos"][0], g.means[np.argmax(imp)], atol=1e-6)   # most important first


def test_from_points_and_prune(rng):
    xyz = rng.normal(size=(200, 3))
    xyz[0] = [1000, 0, 0]                                              # far outlier
    g = from_points(xyz, np.full((200, 3), 128, np.uint8))
    assert len(g) == 200 and np.allclose(sh0_to_rgb(g.sh0), 128 / 255)
    p = prune(g, min_opacity=0.5, crop_pct=99.0)
    assert len(p) < 200 and not np.any(np.all(p.means == [1000, 0, 0], axis=1))
