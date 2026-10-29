"""Competition controller node: /scan + /ego_racecar/odom in, /drive out.

Organizers run ``ros2 run robotforge_controller controller`` with no arguments, so the
defaults for each stage live in STAGE_PRESETS below. Before submitting for a
stage, set STAGE to that stage and commit.
"""

import math
import os

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from robotforge_controller.pure_pursuit import load_raceline
from robotforge_controller.racer import Racer

# Which preset the submitted build uses: 'stage1', 'stage2', 'stage3', 'final'.
STAGE = 'stage1'

STAGE_PRESETS = {
    # Qualification only needs a clean run, so leave a large safety margin.
    'stage1': {'mode': 'pursuit', 'raceline': 'levine_raceline.csv',
               'speed_scale': 0.8, 'max_speed': 6.0},
    'stage2': {'mode': 'pursuit', 'raceline': 'levine_blocked_raceline.csv',
               'speed_scale': 1.0, 'max_speed': 8.0},
    # Hidden track: map-free only.
    'stage3': {'mode': 'gap', 'max_speed': 5.0},
    # Real car: organizers announce the speed cap; keep this at or below it.
    'final': {'mode': 'gap', 'max_speed': 2.0},
}

SCAN_TIMEOUT = 0.25    # s without a scan before the car is stopped
CONTROL_PERIOD = 0.02  # s, 50 Hz command rate


class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        preset = STAGE_PRESETS[STAGE]
        self.declare_parameter('mode', preset['mode'])
        self.declare_parameter('raceline', preset.get('raceline', ''))
        self.declare_parameter('max_speed', float(preset['max_speed']))
        self.declare_parameter('speed_scale', float(preset.get('speed_scale', 1.0)))

        mode = self.get_parameter('mode').value
        max_speed = float(self.get_parameter('max_speed').value)
        raceline = None
        if mode == 'pursuit':
            name = self.get_parameter('raceline').value
            path = name if os.path.isabs(name) else os.path.join(
                get_package_share_directory('robotforge_controller'), 'racelines', name)
            raceline = load_raceline(path)
            self.get_logger().info(f'Racing line: {path}')
        self.racer = Racer(mode, max_speed, raceline,
                           float(self.get_parameter('speed_scale').value))

        self.scan = None
        self.scan_time = None
        self.pose = None
        self.speed = 0.0

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/ego_racecar/odom',
                                 self.odom_callback, 10)
        self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.get_logger().info(
            f'{STAGE}: mode={mode}, max_speed={max_speed:.1f} m/s')

    def scan_callback(self, msg):
        self.scan = msg
        self.scan_time = self.get_clock().now()

    def odom_callback(self, msg):
        p = msg.pose.pose
        q = p.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (p.position.x, p.position.y, yaw)
        self.speed = msg.twist.twist.linear.x

    def publish(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def control_loop(self):
        now = self.get_clock().now()
        if self.scan is None or (now - self.scan_time).nanoseconds * 1e-9 > SCAN_TIMEOUT:
            self.publish(0.0, 0.0)  # no fresh sensor data: stop
            return
        s = self.scan
        steer, speed = self.racer.compute(
            s.ranges, s.angle_min, s.angle_increment, self.speed, self.pose)
        self.publish(steer, speed)


def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
