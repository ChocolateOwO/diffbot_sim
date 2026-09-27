#!/usr/bin/env python3
"""Record 2D LiDAR scans + ground-truth pose to data.csv (Gazebo version of
save_laser_show_pointcloud.py).

One row per recorded pose:
  col 1 = x, col 2 = y, col 3 = yaw [rad]  (robot base pose in the world frame)
  col 4+2(i-1) = range of ray i [m]  (nan = no return)
  col 5+2(i-1) = angle of ray i [rad] relative to the sensor x axis
"""
import csv
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

DEFAULT_OUTPUT = '/mnt/e/งาน/271411/ocgm_part2/data.csv'


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LidarRecorder(Node):

    def __init__(self):
        super().__init__('lidar_recorder')
        self.declare_parameter('output_path', DEFAULT_OUTPUT)
        self.declare_parameter('min_distance', 0.25)     # m moved between rows
        self.declare_parameter('min_rotation_deg', 5.0)  # deg turned between rows
        self.min_dist = self.get_parameter('min_distance').value
        self.min_rot = math.radians(self.get_parameter('min_rotation_deg').value)
        path = self.get_parameter('output_path').value

        self.file = open(path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.pose = None
        self.last = None
        self.rows = 0
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.get_logger().info(f'recording to {path}')

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

    def scan_cb(self, scan):
        if self.pose is None:
            return
        x, y, th = self.pose
        if self.last is not None:
            lx, ly, lth = self.last
            if (math.hypot(x - lx, y - ly) < self.min_dist
                    and abs(wrap(th - lth)) < self.min_rot):
                return
        row = [x, y, th]
        for i, r in enumerate(scan.ranges):
            ok = math.isfinite(r) and scan.range_min <= r < scan.range_max
            row.append(r if ok else float('nan'))
            row.append(scan.angle_min + i * scan.angle_increment)
        self.writer.writerow(row)
        self.file.flush()
        self.last = (x, y, th)
        self.rows += 1
        if self.rows % 50 == 0:
            self.get_logger().info(f'{self.rows} rows recorded')

    def close(self):
        self.get_logger().info(f'closing file, {self.rows} rows recorded')
        self.file.close()


def main(args=None):
    rclpy.init(args=args)
    node = LidarRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
