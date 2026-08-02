#!/usr/bin/env python3
"""Reactive obstacle-avoidance driver for diffbot using the 2D LiDAR scan."""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


FORWARD_SPEED = 0.5      # m/s
TURN_SPEED = 0.8         # rad/s
FRONT_HALF_ANGLE = math.radians(30)   # +/- cone in front considered "ahead"
SIDE_HALF_ANGLE = math.radians(70)    # cone used to compare left vs right space
STOP_DISTANCE = 1.0      # m: start turning when something is closer than this
CONTROL_PERIOD = 0.1     # s


def sector_min_range(scan: LaserScan, center_angle: float, half_width: float) -> float:
    lo = center_angle - half_width
    hi = center_angle + half_width
    i_lo = max(0, int((lo - scan.angle_min) / scan.angle_increment))
    i_hi = min(len(scan.ranges) - 1, int((hi - scan.angle_min) / scan.angle_increment))
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo
    best = float('inf')
    for r in scan.ranges[i_lo:i_hi + 1]:
        if math.isfinite(r) and r > 0.0:
            best = min(best, r)
    return best


class MoveRobot(Node):

    def __init__(self):
        super().__init__('move_robot')
        self.scan = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)
        self.turn_bias = 1.0  # remembers which way we were turning
        self.get_logger().info('move_robot started: driving diffbot around using /scan')

    def scan_cb(self, msg: LaserScan):
        self.scan = msg

    def control_loop(self):
        cmd = Twist()
        if self.scan is None:
            self.cmd_pub.publish(cmd)
            return

        front = sector_min_range(self.scan, 0.0, FRONT_HALF_ANGLE)
        left = sector_min_range(self.scan, SIDE_HALF_ANGLE, FRONT_HALF_ANGLE)
        right = sector_min_range(self.scan, -SIDE_HALF_ANGLE, FRONT_HALF_ANGLE)

        if front < STOP_DISTANCE:
            cmd.linear.x = 0.0
            self.turn_bias = 1.0 if left > right else -1.0
            cmd.angular.z = TURN_SPEED * self.turn_bias
            self.get_logger().info(
                f'obstacle ahead ({front:.2f} m) -> turning {"left" if self.turn_bias > 0 else "right"}',
                throttle_duration_sec=1.0)
        else:
            cmd.linear.x = FORWARD_SPEED
            # gentle steering away from the closer side to avoid grazing walls
            cmd.angular.z = 0.4 * (left - right) / max(left, right, 0.01)
            cmd.angular.z = max(min(cmd.angular.z, TURN_SPEED), -TURN_SPEED)

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MoveRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
