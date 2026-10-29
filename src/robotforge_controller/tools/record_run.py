#!/usr/bin/env python3
"""Record a run in the official simulator for the README video/screenshots.

Listens only (never publishes to /drive): car pose and speed from
/ego_racecar/odom, a thinned LiDAR scan, the controller's /drive command and
the official race monitor topics. Optionally resets the car to the start pose
first (same as RViz "2D Pose Estimate", which also resets the race monitor).
Stops after --laps timed laps (or --max-time seconds) and saves an .npz.

Run with the system Python and ROS sourced, while the simulator is running:
  python3 tools/record_run.py --out run.npz --reset --laps 3
"""

import argparse
import math
import time

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray, Int32, String

SCAN_EVERY = 4          # keep every 4th beam
SAMPLE_PERIOD = 0.05    # s, 20 Hz recording


class Recorder(Node):
    def __init__(self):
        super().__init__('robotforge_recorder')
        self.odom = None
        self.scan = None
        self.drive = (0.0, 0.0)
        self.mon = {'laps': 0, 'collisions': 0, 'cur': 0.0, 'status': '', 'history': []}
        self.rows, self.scans = [], []
        self.scan_meta = None
        self.t0 = None
        self.create_subscription(Odometry, '/ego_racecar/odom', self.on_odom, 20)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(AckermannDriveStamped, '/drive', self.on_drive, 20)
        self.create_subscription(Int32, '/competition/lap_count',
                                 lambda m: self.mon.__setitem__('laps', m.data), 10)
        self.create_subscription(Int32, '/competition/collision_count',
                                 lambda m: self.mon.__setitem__('collisions', m.data), 10)
        self.create_subscription(Float32, '/competition/current_lap_time',
                                 lambda m: self.mon.__setitem__('cur', m.data), 10)
        self.create_subscription(String, '/competition/race_status',
                                 lambda m: self.mon.__setitem__('status', m.data), 10)
        self.create_subscription(Float32MultiArray, '/competition/lap_history',
                                 lambda m: self.mon.__setitem__('history', list(m.data)), 10)
        self.reset_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_timer(SAMPLE_PERIOD, self.sample)

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom = (m.pose.pose.position.x, m.pose.pose.position.y, yaw,
                     m.twist.twist.linear.x)

    def on_scan(self, m):
        self.scan = np.asarray(m.ranges[::SCAN_EVERY], dtype=np.float32)
        self.scan_meta = (m.angle_min, m.angle_increment * SCAN_EVERY)

    def on_drive(self, m):
        self.drive = (m.drive.steering_angle, m.drive.speed)

    def reset(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        self.reset_pub.publish(msg)

    def sample(self):
        if self.odom is None or self.scan is None:
            return
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        m = self.mon
        self.rows.append((now - self.t0, *self.odom, *self.drive,
                          m['laps'], m['collisions'], m['cur']))
        self.scans.append(self.scan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--laps', type=int, default=3)
    ap.add_argument('--max-time', type=float, default=120.0)
    ap.add_argument('--reset', action='store_true', help='reset car to the start pose first')
    ap.add_argument('--start', type=float, nargs=3, default=(-12.0, 0.0, 0.0))
    ap.add_argument('--label', default='', help='free text stored with the run')
    args = ap.parse_args()

    rclpy.init()
    rec = Recorder()
    if args.reset:
        end = time.monotonic() + 1.0
        while time.monotonic() < end:     # let discovery finish, then reset
            rclpy.spin_once(rec, timeout_sec=0.05)
        for _ in range(3):
            rec.reset(*args.start)
            rclpy.spin_once(rec, timeout_sec=0.1)
        rec.rows.clear()
        rec.scans.clear()
        rec.t0 = None
    print('recording...', flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(rec, timeout_sec=0.05)
            t = rec.rows[-1][0] if rec.rows else 0.0
            if rec.mon['laps'] >= args.laps or t > args.max_time:
                end = time.monotonic() + 1.0          # a little extra after the line
                while time.monotonic() < end:
                    rclpy.spin_once(rec, timeout_sec=0.05)
                break
    except KeyboardInterrupt:
        pass
    np.savez_compressed(
        args.out, rows=np.array(rec.rows, dtype=np.float64),
        columns=np.array(['t', 'x', 'y', 'yaw', 'v', 'steer_cmd', 'speed_cmd',
                          'laps', 'collisions', 'cur_lap']),
        scans=np.array(rec.scans, dtype=np.float32),
        scan_meta=np.array(rec.scan_meta or (0.0, 0.0)),
        lap_history=np.array(rec.mon['history'], dtype=np.float64),
        label=np.array(args.label))
    print(f"saved {len(rec.rows)} samples, laps={rec.mon['laps']}, "
          f"collisions={rec.mon['collisions']}, history={rec.mon['history']}", flush=True)
    rec.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
