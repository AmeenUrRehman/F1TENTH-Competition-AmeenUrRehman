"""ROS-free decision layer shared by the node and the offline test harness."""

import math

import numpy as np

from f1_racer.gap_follower import GapFollower, GapParams
from f1_racer.pure_pursuit import PurePursuit, PursuitParams

LIDAR_TO_FRONT = 0.19   # m, LiDAR (0.275 m ahead of rear axle) to front bumper
CORRIDOR_HALF = 0.22    # m, half car width plus a small margin


def corridor_ttc(ranges, angle_min, angle_inc, speed):
    """Time until the car's straight-ahead footprint reaches an obstacle.

    Only beams that land inside the car's own corridor count, so a wall
    running alongside the car does not cause false braking.
    """
    if speed <= 0.1:
        return math.inf
    r = np.nan_to_num(np.asarray(ranges, dtype=np.float64), nan=0.0, posinf=0.0)
    a = angle_min + angle_inc * np.arange(len(r))
    valid = (r > 0.05) & (np.cos(a) > 0.0)
    lateral = np.abs(r * np.sin(a))
    ahead = r * np.cos(a) - LIDAR_TO_FRONT
    hit = valid & (lateral < CORRIDOR_HALF)
    if not hit.any():
        return math.inf
    return max(float(ahead[hit].min()), 0.0) / speed


class Racer:
    """Pick a planner for the stage and add the shared safety layer."""

    def __init__(self, mode, max_speed, raceline=None, speed_scale=1.0,
                 ttc_brake=0.2, gap_params=None, pursuit_params=None):
        if mode not in ('gap', 'pursuit'):
            raise ValueError(f"mode must be 'gap' or 'pursuit', got {mode!r}")
        if mode == 'pursuit' and raceline is None:
            raise ValueError('pursuit mode needs a raceline')
        self.mode = mode
        self.ttc_brake = ttc_brake
        gp = gap_params or GapParams()
        gp.max_speed = max_speed
        self.gap = GapFollower(gp)
        self.pursuit = None
        if mode == 'pursuit':
            pp = pursuit_params or PursuitParams()
            pp.max_speed = max_speed
            pp.speed_scale = speed_scale
            self.pursuit = PurePursuit(raceline, pp)

    def reset(self):
        self.gap.reset()
        if self.pursuit:
            self.pursuit.reset()

    def compute(self, ranges, angle_min, angle_inc, speed, pose=None):
        """Return (steer, speed). ``pose`` is (x, y, yaw) of the rear axle."""
        gap_steer, gap_speed, dbg = self.gap.compute(
            ranges, angle_min, angle_inc, speed)
        if self.mode == 'pursuit':
            if pose is None:
                return 0.0, 0.0
            steer, target, _ = self.pursuit.compute(*pose, speed)
            # The scan still caps speed if something unexpected is close ahead.
            if dbg['free'] < 2.0:
                target = min(target, max(gap_speed, 1.0))
        else:
            steer, target = gap_steer, gap_speed

        if corridor_ttc(ranges, angle_min, angle_inc, speed) < self.ttc_brake:
            target = 0.0
        return steer, target
