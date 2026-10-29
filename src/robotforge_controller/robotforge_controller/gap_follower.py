"""Map-free reactive driving: Follow the Gap target + collision-checked arcs.

The planner only looks at the current LiDAR scan, so it works on tracks it has
never seen (Stage 3) and on the real car. It has no ROS dependency, which lets
the offline test harness call it directly.

Pipeline for every scan:
1. Clean the ranges and extend every disparity (sudden jump in range, i.e.
   an obstacle edge) by half the car width. The deepest direction of this
   extended scan is the *target*: where the car wants to go.
2. Roll out a fan of constant-steering arcs. Along each arc the car's real
   rectangle (plus a margin) is checked against every LiDAR point, giving how
   far that arc stays collision-free.
3. Among arcs that stay free for the whole horizon, take the one that ends
   closest to the target. This lets the car sidestep an obstacle first and
   then go straight, which a single straight-line check cannot express.
4. Speed is limited so the car can stop within the free length of the chosen
   arc, and drops as the steering angle grows.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GapParams:
    wheelbase: float = 0.33          # m
    car_width: float = 0.31          # m
    rear_overhang: float = 0.12      # m behind the rear axle
    front_reach: float = 0.46        # m from rear axle to front bumper
    lidar_x: float = 0.275           # m, LiDAR position ahead of the rear axle
    side_margin: float = 0.08        # m, clearance added on each side
    end_margin: float = 0.10         # m, clearance added front and back
    disparity_threshold: float = 0.25  # m, range jump treated as an edge
    target_fov_deg: float = 100.0    # +/- degrees searched for the target
    target_dist: float = 6.0         # m, target point is capped at this range
    near_best: float = 0.95          # target tie-break: prefer straight ahead
    max_range: float = 10.0          # m, ranges are clipped to this
    n_arcs: int = 31
    max_steer: float = 0.4189        # rad, simulator steering limit
    arc_step: float = 0.1            # m between footprint checks
    horizon_min: float = 1.5         # m, arc length at standstill
    horizon_per_speed: float = 0.5   # s, horizon grows with speed
    horizon_max: float = 4.0         # m
    smooth_weight: float = 0.3       # cost per rad of steering change
    steer_smoothing: float = 0.2     # 0 = no filter, closer to 1 = smoother
    max_speed: float = 6.0           # m/s
    min_speed: float = 0.8           # m/s
    brake_decel: float = 5.0         # m/s^2 assumed achievable braking
    stop_margin: float = 0.25        # m kept in front of the car
    lateral_accel: float = 6.0       # m/s^2, grip budget used in turns
    point_decimate: int = 2          # use every n-th beam for collision checks


class GapFollower:
    """Compute (steering, speed) from one LiDAR scan."""

    def __init__(self, params=None):
        self.p = params or GapParams()
        self.prev_steer = 0.0
        p = self.p
        self.steers = np.linspace(-p.max_steer, p.max_steer, p.n_arcs)
        self.curv = np.tan(self.steers) / p.wheelbase

    def reset(self):
        self.prev_steer = 0.0

    def _extend_disparities(self, ranges, angle_inc):
        """Grow the closer side of each disparity by half the car width."""
        out = ranges.copy()
        half_width = self.p.car_width / 2.0 + self.p.side_margin
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

    def _target(self, r, angles, angle_inc):
        """Deepest direction of the disparity-extended scan, as a point."""
        p = self.p
        ext = self._extend_disparities(r, angle_inc)
        window = np.flatnonzero(np.abs(angles) <= math.radians(p.target_fov_deg))
        sub = ext[window]
        candidates = window[sub >= p.near_best * sub.max()]
        i = candidates[np.argmin(np.abs(angles[candidates]))]
        d = min(ext[i], p.target_dist)
        # In the rear-axle frame, where the arcs start.
        return np.array([p.lidar_x + d * math.cos(angles[i]),
                         d * math.sin(angles[i])]), float(angles[i])

    def _arc_free_lengths(self, px, py, horizon):
        """Collision-free length of each arc for the car footprint."""
        p = self.p
        s = np.arange(p.arc_step, horizon + 1e-9, p.arc_step)          # (S,)
        k = self.curv[:, None]                                          # (A,1)
        ks = k * s[None, :]
        small = np.abs(k) < 1e-6
        safe_k = np.where(small, 1.0, k)
        x = np.where(small, s[None, :], np.sin(ks) / safe_k)            # (A,S)
        y = np.where(small, 0.0, (1.0 - np.cos(ks)) / safe_k)
        th = ks
        c, sn = np.cos(th)[..., None], np.sin(th)[..., None]
        dx = px[None, None, :] - x[..., None]                           # (A,S,P)
        dy = py[None, None, :] - y[..., None]
        lx = c * dx + sn * dy
        ly = -sn * dx + c * dy
        half = p.car_width / 2.0 + p.side_margin
        inside = ((lx > -p.rear_overhang - p.end_margin)
                  & (lx < p.front_reach + p.end_margin)
                  & (np.abs(ly) < half)).any(axis=2)                    # (A,S)
        hit = inside.any(axis=1)
        first = np.where(hit, inside.argmax(axis=1), len(s))
        free = np.where(first == 0, 0.0, s[np.maximum(first - 1, 0)])
        free[~hit] = horizon
        ends = np.stack([x[:, -1], y[:, -1]], axis=1)
        return free, ends

    def compute(self, ranges, angle_min, angle_inc, current_speed=0.0):
        """Return (steering_angle [rad], speed [m/s], debug dict)."""
        p = self.p
        r = np.asarray(ranges, dtype=np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=p.max_range, neginf=0.0)
        r = np.clip(r, 0.0, p.max_range)
        angles = angle_min + angle_inc * np.arange(len(r))

        target, target_angle = self._target(r, angles, angle_inc)

        horizon = float(np.clip(p.horizon_min + p.horizon_per_speed * max(current_speed, 0.0),
                                p.horizon_min, p.horizon_max))
        sel = slice(None, None, p.point_decimate)
        rr, aa = r[sel], angles[sel]
        near = (rr > 0.05) & (rr < horizon + 1.0)
        px = p.lidar_x + rr[near] * np.cos(aa[near])
        py = rr[near] * np.sin(aa[near])
        free, ends = self._arc_free_lengths(px, py, horizon)

        clear = free >= horizon - 1e-9
        if clear.any():
            cost = np.linalg.norm(ends - target, axis=1) \
                + p.smooth_weight * np.abs(self.steers - self.prev_steer)
            cost[~clear] = np.inf
            best = int(np.argmin(cost))
        else:
            # Nothing is clear for the full horizon: take the longest escape,
            # breaking ties towards the target direction.
            best = int(np.argmax(free - 0.01 * np.abs(self.steers - target_angle)))

        steer = float(self.steers[best])
        steer = p.steer_smoothing * self.prev_steer + (1 - p.steer_smoothing) * steer
        self.prev_steer = steer

        usable = max(float(free[best]) - p.stop_margin, 0.0)
        v_stop = math.sqrt(2.0 * p.brake_decel * usable)
        if clear[best]:
            # The whole horizon is free; beyond it the straight-line depth to
            # the target still bounds how fast it is sensible to go.
            v_stop = max(v_stop, math.sqrt(2.0 * p.brake_decel * max(
                np.linalg.norm(target) - p.lidar_x - p.stop_margin, 0.0)))
        # Grip limit for the curvature being driven (v^2 * k <= a_lat).
        curvature = abs(math.tan(steer)) / p.wheelbase
        v_turn = math.sqrt(p.lateral_accel / max(curvature, 1e-6))
        speed = float(np.clip(min(v_stop, v_turn), p.min_speed, p.max_speed))
        if not clear.any():
            # Boxed in: creep only while the escape arc is still clearly free,
            # otherwise stop. Any contact voids an attempt (and on the real
            # car it is the one thing that must not happen).
            speed = min(speed, 0.5) if free[best] > 2 * p.stop_margin else 0.0

        return steer, speed, {"target_angle": target_angle, "free": float(free[best]),
                              "arc_free": free, "horizon": horizon}

    def stop_speed(self, free_length):
        """Highest speed that still stops within ``free_length``."""
        usable = max(free_length - self.p.stop_margin, 0.0)
        return math.sqrt(2.0 * self.p.brake_decel * usable)
