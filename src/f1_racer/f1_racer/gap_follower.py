"""Map-free reactive driving: a disparity-extender variant of Follow the Gap.

The planner only looks at the current LiDAR scan, so it works on tracks it has
never seen (Stage 3) and on the real car. It has no ROS dependency, which lets
the offline test harness call it directly.

Pipeline for every scan:
1. Clean the ranges (NaN/inf, clipping).
2. Extend every "disparity" (a sudden jump in range, i.e. the edge of a wall
   or obstacle) by half the car width plus a margin, so that a gap is only
   considered open if the whole car fits through it.
3. Steer towards the deepest direction of the extended scan.
4. Pick a speed that lets the car stop before the obstacle straight ahead and
   that drops as the steering angle grows.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GapParams:
    car_width: float = 0.31          # m, F1TENTH vehicle width in the sim
    safety_margin: float = 0.12      # m, extra clearance on each side
    disparity_threshold: float = 0.25  # m, range jump treated as an edge
    fov_deg: float = 100.0           # +/- degrees of the scan used for steering
    max_range: float = 10.0          # m, ranges are clipped to this
    side_clearance: float = 0.22     # m, never turn into a wall closer than this
    steer_gain: float = 1.0
    center_gain: float = 0.25        # rad per m of left/right imbalance
    max_steer: float = 0.4189        # rad, simulator steering limit
    steer_smoothing: float = 0.5     # 0 = no filter, closer to 1 = smoother
    max_speed: float = 6.0           # m/s
    min_speed: float = 1.0           # m/s
    brake_decel: float = 5.0         # m/s^2 assumed achievable braking
    stop_margin: float = 0.35        # m, distance to keep in front of the car
    steer_speed_gain: float = 1.6    # speed reduction per rad of steering
    forward_cone_deg: float = 6.0    # +/- degrees used to measure free distance


class GapFollower:
    """Compute (steering, speed) from one LiDAR scan."""

    def __init__(self, params=None):
        self.p = params or GapParams()
        self.prev_steer = 0.0

    def reset(self):
        self.prev_steer = 0.0

    def _extend_disparities(self, ranges, angle_inc):
        """Grow the closer side of each disparity by half the car width."""
        out = ranges.copy()
        half_width = self.p.car_width / 2.0 + self.p.safety_margin
        jumps = np.flatnonzero(
            np.abs(np.diff(ranges)) > self.p.disparity_threshold)
        n = len(ranges)
        for i in jumps:
            near, far = (i, i + 1) if ranges[i] < ranges[i + 1] else (i + 1, i)
            near_r = max(ranges[near], 0.05)
            count = int(math.ceil(math.atan2(half_width, near_r) / angle_inc))
            step = 1 if far > near else -1
            end = min(max(near + step * count, 0), n - 1)
            idx = np.arange(far, end + step, step)
            out[idx] = np.minimum(out[idx], near_r)
        return out

    def compute(self, ranges, angle_min, angle_inc, current_speed=0.0):
        """Return (steering_angle [rad], speed [m/s], debug dict)."""
        p = self.p
        r = np.asarray(ranges, dtype=np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=p.max_range, neginf=0.0)
        r = np.clip(r, 0.0, p.max_range)
        angles = angle_min + angle_inc * np.arange(len(r))

        extended = self._extend_disparities(r, angle_inc)

        # Steering: deepest direction inside the field of view. Among beams
        # nearly as deep as the deepest one, take the one closest to straight
        # ahead, which avoids zig-zagging down long corridors.
        fov = math.radians(p.fov_deg)
        window = np.flatnonzero(np.abs(angles) <= fov)
        sub = extended[window]
        best = sub.max()
        candidates = window[sub >= 0.95 * best]
        target = candidates[np.argmin(np.abs(angles[candidates]))]
        target_angle = angles[target]

        # A wall running alongside the car creates no disparity, so also push
        # away from whichever side is closer. Capped so it cannot override
        # a real turn.
        side = np.abs(np.abs(angles) - math.pi / 2) < math.radians(15)
        left_min = r[side & (angles > 0)].min(initial=p.max_range)
        right_min = r[side & (angles < 0)].min(initial=p.max_range)
        imbalance = np.clip(min(left_min, 2.0) - min(right_min, 2.0), -1.0, 1.0)
        steer = float(np.clip(p.steer_gain * target_angle + p.center_gain * imbalance,
                              -p.max_steer, p.max_steer))

        # Do not cut a corner into the wall right next to the car.
        limit = p.car_width / 2.0 + p.side_clearance
        if steer > 0 and left_min < limit:
            steer = 0.0
        elif steer < 0 and right_min < limit:
            steer = 0.0

        steer = p.steer_smoothing * self.prev_steer + (1 - p.steer_smoothing) * steer
        self.prev_steer = steer

        # Speed: stop-distance limit along the travel direction, then reduce
        # for sharp steering.
        cone = np.abs(angles - steer) <= math.radians(p.forward_cone_deg)
        free = float(extended[cone].min(initial=p.max_range))
        free_straight = float(
            r[np.abs(angles) <= math.radians(p.forward_cone_deg)].min(
                initial=p.max_range))
        free = min(free, max(free_straight, free))
        usable = max(free - p.stop_margin, 0.0)
        v_stop = math.sqrt(2.0 * p.brake_decel * usable)
        v_turn = p.max_speed / (1.0 + p.steer_speed_gain * abs(steer))
        speed = float(np.clip(min(v_stop, v_turn), p.min_speed, p.max_speed))
        if free < p.stop_margin:
            speed = 0.5  # crawl rather than freeze; steering keeps working

        return steer, speed, {"target_angle": target_angle, "free": free}
