#!/usr/bin/env python3
"""Drive the robot through the waypoints in config/route.yaml (world frame).

Turn in place towards the next waypoint, then drive to it with a proportional
heading correction. A LiDAR safety check stops the robot when something is
inside a box in front of it. Exits when the route is finished.
"""
import math
import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from diffbot_sim.lidar_recorder import wrap, yaw_from_quaternion

CONTROL_PERIOD = 0.05     # s
WP_TOLERANCE = 0.30       # m: switch to next waypoint
FINAL_TOLERANCE = 0.20    # m
TURN_IN_PLACE = 0.5       # rad: rotate first if heading error is larger
MAX_TURN = 0.9            # rad/s
FRONT_BOX_X = 0.95        # m ahead of the LiDAR (body front face is at 0.5)
FRONT_BOX_HALF_Y = 0.62   # m (body half width 0.5 + wheels)
BLOCKED_SKIP_S = 8.0      # give up on a waypoint after being blocked this long


class WaypointDriver(Node):

    def __init__(self):
        super().__init__('waypoint_driver')
        default = os.path.join(get_package_share_directory('diffbot_sim'), 'config', 'route.yaml')
        self.declare_parameter('route_file', default)
        self.declare_parameter('speed', 0.0)   # 0 -> use the value in the route file
        route = yaml.safe_load(open(self.get_parameter('route_file').value))
        self.waypoints = route['waypoints'][1:]   # first entry is the spawn point
        self.speed = self.get_parameter('speed').value or float(route.get('speed', 0.5))
        self.idx = 0
        self.pose = None
        self.scan = None
        self.blocked_for = 0.0
        self.done = False
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_timer(CONTROL_PERIOD, self.control)
        self.get_logger().info(f'{len(self.waypoints)} waypoints, speed {self.speed} m/s')

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

    def scan_cb(self, msg):
        self.scan = msg

    def front_blocked(self):
        s = self.scan
        if s is None:
            return False
        for i, r in enumerate(s.ranges):
            if not math.isfinite(r):
                continue
            a = s.angle_min + i * s.angle_increment
            x, y = r * math.cos(a), r * math.sin(a)
            if 0.0 < x < FRONT_BOX_X and abs(y) < FRONT_BOX_HALF_Y:
                return True
        return False

    def control(self):
        if self.done or self.pose is None or self.scan is None:
            return
        x, y, th = self.pose
        gx, gy = self.waypoints[self.idx]
        dist = math.hypot(gx - x, gy - y)
        last = self.idx == len(self.waypoints) - 1
        if dist < (FINAL_TOLERANCE if last else WP_TOLERANCE):
            self.get_logger().info(f'reached waypoint {self.idx + 1}/{len(self.waypoints)} ({gx}, {gy})')
            self.idx += 1
            self.blocked_for = 0.0
            if self.idx >= len(self.waypoints):
                self.pub.publish(Twist())
                self.get_logger().info('route finished')
                self.done = True
            return

        err = wrap(math.atan2(gy - y, gx - x) - th)
        cmd = Twist()
        if abs(err) > TURN_IN_PLACE:
            cmd.angular.z = math.copysign(max(0.3, min(MAX_TURN, 1.5 * abs(err))), err)
        else:
            cmd.linear.x = min(self.speed, 0.8 * dist + 0.15) * max(0.3, math.cos(err))
            cmd.angular.z = max(-MAX_TURN, min(MAX_TURN, 1.8 * err))
            if self.front_blocked():
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.blocked_for += CONTROL_PERIOD
                if self.blocked_for > BLOCKED_SKIP_S:
                    self.get_logger().warn(f'blocked, skipping waypoint {self.idx + 1}')
                    self.idx = min(self.idx + 1, len(self.waypoints) - 1)
                    self.blocked_for = 0.0
            else:
                self.blocked_for = 0.0
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointDriver()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
