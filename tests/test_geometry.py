import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from xgs.confidence import confidence_colors, max_triangulation_angle_deg, nearest_confidence, point_confidence
from xgs.eval.metrics import associate, ate, psnr
from xgs.sfm.scale import detect_markers, marker_scale, umeyama


def test_umeyama_recovers_sim3(rng):
    src = rng.normal(size=(50, 3))
    R = Rotation.from_euler("xyz", [10, -20, 35], degrees=True).as_matrix()
    dst = 2.5 * src @ R.T + np.array([1, -2, 3])
    tf = umeyama(src, dst)
    assert np.isclose(tf.s, 2.5) and np.allclose(tf.R, R) and np.allclose(tf.apply(src), dst)


def test_ate_is_zero_for_scaled_copy(rng):
    t = np.linspace(0, 10, 100)
    gt = np.c_[t, np.cumsum(rng.normal(size=(100, 3)), axis=0), np.zeros((100, 4))]
    est = gt.copy()
    est[:, 1:4] = 0.3 * est[:, 1:4] + 5
    m = ate(est, gt)
    assert m["matched"] == 100 and m["ate_rmse_m"] < 1e-6 and np.isclose(m["sim3_scale"], 1 / 0.3, rtol=1e-4)


def test_associate_respects_max_dt():
    ie, ig = associate(np.array([0.0, 1.0, 2.5]), np.array([0.0, 1.01, 2.0]), max_dt=0.02)
    assert ie.tolist() == [0, 1] and ig.tolist() == [0, 1]


def test_marker_scale_and_error():
    true = {0: np.array([0, 0, 0.0]), 1: np.array([2.0, 0, 0]), 2: np.array([0, 3.0, 0])}
    model = {k: v / 4.0 for k, v in true.items()}
    m = marker_scale(model, true)
    assert np.isclose(m["scale"], 4.0) and m["mean_abs_error_m"] < 1e-9


def test_detect_synthetic_aruco_marker():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(d, 7, 120)
    img = np.full((300, 300), 255, np.uint8)
    img[90:210, 90:210] = tag
    found = detect_markers(img)
    assert list(found) == [7]
    assert np.allclose(found[7].mean(axis=0), [149.5, 149.5], atol=2)


def test_confidence_behaviour():
    p = np.zeros(3)
    assert max_triangulation_angle_deg(p, np.array([[1, 0, 10.0]])) == 0.0
    wide = max_triangulation_angle_deg(p, np.array([[-5, 0, 10.0], [5, 0, 10.0]]))
    assert 50 < wide < 54
    c = point_confidence(np.array([2, 6, 20]), np.array([1.0, 10.0, 30.0]))
    assert c[0] < 0.2 and np.isclose(c[1], 1.0) and np.isclose(c[2], 1.0)
    rgb = confidence_colors(np.array([0.0, 1.0]))
    assert rgb[0, 0] > rgb[0, 1] and rgb[1, 1] > rgb[1, 0]       # red low, green high
    q = nearest_confidence(np.array([[0, 0, 0.1]]), np.array([[0, 0, 0.0], [9, 9, 9.0]]), np.array([0.2, 0.9]), k=1)
    assert np.isclose(q[0], 0.2)


def test_psnr():
    a = np.zeros((4, 4, 3))
    assert psnr(a, a) == float("inf") and np.isclose(psnr(a, a + 0.1), 20.0)
