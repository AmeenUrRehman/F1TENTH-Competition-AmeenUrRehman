"""Pure Pursuit tracking of a precomputed racing line (known-map stages only).

The racing line is a CSV of ``x, y, speed`` rows produced offline by
``tools/make_raceline.py`` from the officially released map. Stage 3 forbids
track-specific coordinates, so this module must not be used there.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PursuitParams:
    wheelbase: float = 0.33          # m, lf + lr of the F1TENTH model
    max_steer: float = 0.4189        # rad
    lookahead_min: float = 0.6       # m
    lookahead_gain: float = 0.18     # s, lookahead grows with speed
    lookahead_max: float = 2.0       # m
    speed_scale: float = 1.0         # multiply the profile, quick global tune
    max_speed: float = 8.0           # m/s hard cap
    search_window: int = 60          # waypoints searched around last index


def load_raceline(path):
    """Read ``x, y, v`` rows; lines starting with ``#`` are ignored."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path}: expected rows of x, y, speed")
    return data[:, :3]


class PurePursuit:
    def __init__(self, raceline, params=None):
        self.p = params or PursuitParams()
        self.xy = raceline[:, :2]
        self.v = raceline[:, 2]
        self.n = len(self.xy)
        seg = np.linalg.norm(np.diff(self.xy, axis=0, append=self.xy[:1]), axis=1)
        self.seg = seg
        self.index = None

    def reset(self):
        self.index = None

    def _nearest(self, x, y):
        if self.index is None:
            d = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
            self.index = int(np.argmin(d))
            return self.index
        idx = (self.index + np.arange(-5, self.p.search_window)) % self.n
        d = np.hypot(self.xy[idx, 0] - x, self.xy[idx, 1] - y)
        self.index = int(idx[np.argmin(d)])
        return self.index

    def compute(self, x, y, yaw, speed):
        """Return (steering_angle, target_speed, debug dict).

        ``x, y, yaw`` is the rear-axle pose (``base_link`` in the odometry).
        """
        p = self.p
        i = self._nearest(x, y)
        lookahead = float(np.clip(p.lookahead_min + p.lookahead_gain * speed,
                                  p.lookahead_min, p.lookahead_max))

        # Walk forward along the line until the lookahead distance is reached.
        j = i
        travelled = 0.0
        while travelled < lookahead:
            travelled += self.seg[j]
            j = (j + 1) % self.n
            if j == i:
                break
        gx, gy = self.xy[j]

        dx, dy = gx - x, gy - y
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        dist2 = dx * dx + dy * dy
        curvature = 2.0 * local_y / max(dist2, 1e-6)
        steer = float(np.clip(math.atan(p.wheelbase * curvature),
                              -p.max_steer, p.max_steer))

        target = float(min(self.v[i] * p.speed_scale, p.max_speed))
        cross_track = float(np.hypot(self.xy[i, 0] - x, self.xy[i, 1] - y))
        return steer, target, {"index": i, "goal": (gx, gy),
                               "cross_track": cross_track}
