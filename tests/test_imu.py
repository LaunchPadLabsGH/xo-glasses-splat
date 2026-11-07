import numpy as np

from xgs.io.imu import load_imu


def test_generic_csv_deg_s(tmp_path):
    p = tmp_path / "imu.csv"
    p.write_text("t_s,gx,gy,gz,ax,ay,az\n0.1,0,90,0,0,0,9.8\n0.0,0,0,0,0,0,9.8\n")
    imu = load_imu(p, "generic_csv", gyro_units="deg_s")
    assert imu.t.tolist() == [0.0, 0.1]                 # sorted by time
    assert np.isclose(imu.gyro[1, 1], np.pi / 2)
    assert imu.accel.shape == (2, 3)
    assert np.isclose(imu.angular_speed_deg_s(np.array([0.1]), 0.01)[0], 90)


def test_euroc(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n1000000000,0.1,0,0,0,0,9.8\n1005000000,0.2,0,0,0,0,9.8\n")
    imu = load_imu(p, "euroc")
    assert np.allclose(imu.t, [0, 0.005])
    assert np.allclose(imu.gyro[:, 0], [0.1, 0.2])
