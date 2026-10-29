# robotforge_controller — Team RobotForge, CIE-KFUPM F1TENTH competition controller

Team: **RobotForge** · Member: Ameen Ur Rehman (ID 202527430)

ROS 2 Jazzy `ament_python` package. One executable:

```bash
ros2 run robotforge_controller controller
```

## Interface

| Topic | Type | Direction |
|---|---|---|
| `/scan` | `sensor_msgs/msg/LaserScan` (sensor-data QoS) | subscribed |
| `/ego_racecar/odom` | `nav_msgs/msg/Odometry` | subscribed |
| `/drive` | `ackermann_msgs/msg/AckermannDriveStamped` | published at 50 Hz, stamped, radians and m/s |

If no scan has arrived for 0.25 s the node publishes zero speed. On shutdown it
publishes one zero-speed command.

## Dependencies

`rclpy`, `sensor_msgs`, `nav_msgs`, `ackermann_msgs`, `ament_index_python`,
`python3-numpy`, all declared in `package.xml`. Nothing else is needed at runtime.
(`tools/` scripts are development-only and are not installed.)

## Build

```bash
source /opt/ros/jazzy/setup.bash
source <competition-repo>/starter_ws/.venv/bin/activate
source <competition-repo>/starter_ws/install/setup.bash
cd <workspace>
python /usr/bin/colcon build --symlink-install
source install/setup.bash
ros2 run robotforge_controller controller
```

## How it drives

* **`gap` mode (map-free):** disparity-extender Follow the Gap. Steers to the
  deepest direction after widening every obstacle edge by half the car width
  plus a margin; speed limited by stopping distance and steering angle. Uses
  `/scan` only. This is the mode for the hidden-track stage and the real car.
* **`pursuit` mode (known maps only):** Pure Pursuit on a racing line in
  `racelines/`, generated offline from the officially released map and
  centerline by `tools/make_raceline.py`. The LiDAR still limits speed.
* **Both modes:** a corridor time-to-collision brake sets speed to zero if the
  car's own straight-ahead footprint would reach an obstacle within 0.2 s.

## Stage selection and parameters

`STAGE` at the top of `robotforge_controller/controller_node.py` selects the preset used
when the node is started with no arguments:

| STAGE | mode | racing line | max speed requested |
|---|---|---|---|
| `stage1` | pursuit | `levine_raceline.csv` | 6.0 m/s |
| `stage2` | pursuit | `levine_blocked_raceline.csv` | 8.0 m/s |
| `stage3` | gap | none (no track-specific data) | 5.0 m/s |
| `final` | gap | none | 2.0 m/s (kept at or below the official cap) |

ROS parameters can override a preset for testing: `mode` (`gap`/`pursuit`),
`raceline` (file name in `racelines/` or absolute path), `max_speed` (m/s),
`speed_scale` (multiplies the racing-line speed profile).

**Maximum speed the controller may request:** the `max_speed` of the active
preset (8.0 m/s at most).
