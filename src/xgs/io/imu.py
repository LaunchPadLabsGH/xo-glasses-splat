"""IMU loaders. IMU input is optional: without it, motion is estimated from optical flow.

Supported formats
- generic_csv: header row; columns t_s, gx, gy, gz [, ax, ay, az]. Gyro in rad/s
  (or deg/s with gyro_units: deg_s). Timestamps in seconds on the video clock.
- euroc: EuRoC/ASL `imu0/data.csv`: timestamp [ns], w_x, w_y, w_z [rad/s], a_x, a_y, a_z.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ImuData:
    t: np.ndarray          # (N,) seconds
    gyro: np.ndarray       # (N, 3) rad/s
    accel: np.ndarray | None = None  # (N, 3) m/s^2

    def angular_speed_deg_s(self, times: np.ndarray, window_s: float = 0.05) -> np.ndarray:
        """Peak gyro magnitude within +/- window_s of each query time (deg/s)."""
        mag = np.degrees(np.linalg.norm(self.gyro, axis=1))
        out = np.zeros(len(times))
        lo = np.searchsorted(self.t, times - window_s, side="left")
        hi = np.searchsorted(self.t, times + window_s, side="right")
        for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
            if b > a:
                out[i] = mag[a:b].max()
            else:  # no sample in window: nearest sample
                j = int(np.clip(np.searchsorted(self.t, times[i]), 0, len(self.t) - 1))
                out[i] = mag[j]
        return out


def load_imu(path: Path, fmt: str = "generic_csv", gyro_units: str = "rad_s",
             time_offset_s: float = 0.0) -> ImuData:
    path = Path(path)
    if fmt == "euroc":
        data = np.loadtxt(path, delimiter=",", comments="#")
        t = data[:, 0] * 1e-9
        t = t - t[0]
        gyro = data[:, 1:4]
        accel = data[:, 4:7]
    elif fmt == "generic_csv":
        data = np.genfromtxt(path, delimiter=",", names=True)
        names = data.dtype.names or ()
        t = np.asarray(data[names[0]], dtype=float)
        gyro = np.stack([data[n] for n in ("gx", "gy", "gz")], axis=1).astype(float)
        accel = (np.stack([data[n] for n in ("ax", "ay", "az")], axis=1).astype(float)
                 if all(n in names for n in ("ax", "ay", "az")) else None)
    else:
        raise ValueError(f"unknown IMU format {fmt!r}")
    if gyro_units == "deg_s":
        gyro = np.radians(gyro)
    order = np.argsort(t)
    return ImuData(t=t[order] + time_offset_s, gyro=gyro[order],
                   accel=None if accel is None else accel[order])
