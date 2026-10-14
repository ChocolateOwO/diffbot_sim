#!/usr/bin/env python3
"""Drive the robot through the RRT* nodes of plan.json in Gazebo.

This is the Gazebo version of the CoppeliaSim zmqRemoteApi program:

  CoppeliaSim (zmqRemoteApi)                    Gazebo (ROS 2 / rclpy)
  ------------------------------------------    ------------------------------------------------
  RemoteAPIClient() / sim.startSimulation()     Gazebo is already running (ros2 launch ...)
  sim.getObjectPosition / getObjectOrientation  subscribe /odom  (ground truth pose in the world)
  sim.setJointTargetVelocity(motor, speed)      left / right wheel speed [rad/s] -> /cmd_vel;
                                                gazebo_ros_diff_drive turns it into the target
                                                velocity of the 4 wheel joints

What the program does:
  1. read plan.json  (start pose, goal pose, ordered RRT* nodes)
  2. interpolate between the nodes (a point every 0.1 m)
  3. speed + steering: pure-pursuit style heading control towards a look-ahead point on the
     interpolated path, converted to left / right wheel speeds
  4. when the position of the goal is reached, rotate in place to the goal heading theta
  5. save the driven trajectory (trajectory.csv) and print the final errors

Run (Gazebo must already be running with the robot spawned on the start pose of the plan,
rrt_nav.launch.py does both):
    python3 follow_path_gazebo_660610829.py --plan plan.json
"""
import argparse
import csv
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# robot data (diffbot.urdf.xacro)
WHEEL_RADIUS = 0.15          # [m]
WHEEL_SEPARATION = 1.08      # [m] left wheel centre <-> right wheel centre
MAX_WHEEL_SPEED = 6.0        # [rad/s]  = 0.9 m/s

# controller settings
CONTROL_DT = 0.05            # [s]
INTERP_STEP = 0.1            # [m] spacing of the interpolated path points
POS_TOL = 0.10               # [m] goal position reached
ANG_TOL = 0.05               # [rad] goal heading reached (~3 deg)
ROT_ENTER, ROT_EXIT = 0.8, 0.25   # [rad] start / stop turning in place
K_HEADING = 1.8              # [1/s]
MAX_YAW_RATE = 0.9           # [rad/s]
# obstacle safety box in front of the LiDAR (x forward, y left), body front face is at x = 0.5
SAFE_X, SAFE_HALF_Y = 0.70, 0.52
BLOCKED_ABORT_S = 10.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def interpolate_path(nodes, step=INTERP_STEP):
    """Straight-line interpolation between consecutive RRT* nodes."""
    pts = []
    for (x0, y0), (x1, y1) in zip(nodes[:-1], nodes[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        pts.extend((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n))
    pts.append(tuple(nodes[-1]))
    return pts


def wheel_speeds(v, w):
    """Body speed (v [m/s], w [rad/s]) -> left / right wheel joint speed [rad/s]."""
    wl = (v - w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    wr = (v + w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    big = max(abs(wl), abs(wr))
    if big > MAX_WHEEL_SPEED:                 # keep the ratio (= the steering), only slow down
        wl, wr = wl * MAX_WHEEL_SPEED / big, wr * MAX_WHEEL_SPEED / big
    return wl, wr


def body_speed(wl, wr):
    """Left / right wheel joint speed [rad/s] -> body speed (v [m/s], w [rad/s])."""
    return (WHEEL_RADIUS * (wl + wr) / 2.0,
            WHEEL_RADIUS * (wr - wl) / WHEEL_SEPARATION)


class PathFollower(Node):

    def __init__(self, plan, args):
        super().__init__('path_follower')
        self.args = args
        self.goal = plan['goal']                              # x, y, theta
        self.nodes = plan['path']
        self.path = interpolate_path(self.nodes)
        self.arc = [0.0]
        for a, b in zip(self.path[:-1], self.path[1:]):
            self.arc.append(self.arc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.get_logger().info(
            f'{len(self.nodes)} RRT* nodes -> {len(self.path)} interpolated points, '
            f'{self.arc[-1]:.1f} m, max speed {args.speed} m/s')

        self.pose = None
        self.scan = None
        self.idx = 0
        self.mode = 'TRACK'
        self.turning = False
        self.settled = 0
        self.blocked_for = 0.0
        self.t_start = None
        self.last_log = 0.0
        self.done = False
        self.exit_code = 0
        self.rows = []
        # the trajectory is written while driving (one row per control step) so it can be
        # watched live next to Gazebo (plot_result_660610829.py --live)
        self.traj_file = open(args.traj_out, 'w', newline='')
        self.traj_writer = csv.writer(self.traj_file)
        self.traj_writer.writerow(['t', 'x', 'y', 'yaw', 'v_cmd', 'w_cmd', 'wheel_left', 'wheel_right', 'mode'])
        self.traj_file.flush()

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_timer(CONTROL_DT, self.control)

    # ---- sensors ------------------------------------------------------------
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
            if math.isfinite(r) and s.range_min <= r < s.range_max:
                a = s.angle_min + i * s.angle_increment
                if 0.0 < r * math.cos(a) < SAFE_X and abs(r * math.sin(a)) < SAFE_HALF_Y:
                    return True
        return False

    # ---- helpers --------------------------------------------------------------
    def send(self, v, w):
        """Wheel joint speeds (what setJointTargetVelocity would get) -> /cmd_vel."""
        wl, wr = wheel_speeds(v, w)
        v_cmd, w_cmd = body_speed(wl, wr)
        msg = Twist()
        msg.linear.x, msg.angular.z = v_cmd, w_cmd
        self.pub.publish(msg)
        return v_cmd, w_cmd, wl, wr

    def turn_ahead(self):
        """Heading change of the path in the next 1.5 m (used to slow down before corners)."""
        j0 = self.idx
        j1 = j0
        while j1 < len(self.path) - 1 and self.arc[j1] - self.arc[j0] < 1.5:
            j1 += 1
        if j1 - j0 < 3:
            return 0.0
        mid = (j0 + j1) // 2
        a = math.atan2(self.path[mid][1] - self.path[j0][1], self.path[mid][0] - self.path[j0][0])
        b = math.atan2(self.path[j1][1] - self.path[mid][1], self.path[j1][0] - self.path[mid][0])
        return abs(wrap(b - a))

    def finish(self, ok, text):
        self.send(0.0, 0.0)
        x, y, yaw = self.pose
        gx, gy, gth = self.goal
        pos_err = math.hypot(gx - x, gy - y)
        ang_err = wrap(gth - yaw)
        # how far the driven trajectory stayed from the planned path
        cross = 0.0
        for r in self.rows:
            cross = max(cross, min(math.hypot(r[1] - px, r[2] - py) for px, py in self.path[::3]))
        elapsed = time.time() - self.t_start
        self.get_logger().info(text)
        banner = 'SUCCESS: goal reached' if ok else f'FAILED: {text}'
        self.get_logger().info('*' * 8 + f'  {banner}  ' + '*' * 8)
        self.get_logger().info(
            f'final pose ({x:.2f}, {y:.2f}, {yaw:.2f}) goal ({gx:.2f}, {gy:.2f}, {gth:.2f}) | '
            f'position error {pos_err:.3f} m, heading error {math.degrees(ang_err):.1f} deg | '
            f'max distance from the planned path {cross:.2f} m | {elapsed:.0f} s')
        self.traj_file.close()
        with open(self.args.traj_out.rsplit('.', 1)[0] + '_result.json', 'w') as f:
            json.dump({'success': ok, 'message': text, 'final_pose': [x, y, yaw], 'goal': self.goal,
                       'position_error_m': pos_err, 'heading_error_deg': math.degrees(ang_err),
                       'max_distance_from_plan_m': cross, 'time_s': elapsed}, f, indent=1)
        self.exit_code = 0 if ok else 2
        self.done = True

    # ---- control loop ---------------------------------------------------------
    def control(self):
        if self.pose is None or self.done:
            return
        now = time.time()
        if self.t_start is None:
            self.t_start = now
            self.get_logger().info(
                f'start pose ({self.pose[0]:.2f}, {self.pose[1]:.2f}, {self.pose[2]:.2f}) '
                f'-> goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f})')
        if now - self.t_start > self.args.timeout:
            return self.finish(False, f'timeout after {self.args.timeout:.0f} s')

        x, y, yaw = self.pose
        gx, gy, gth = self.goal
        v = w = 0.0

        if self.mode == 'TRACK':
            # closest path point (only look forward, 6 m window)
            best, best_d = self.idx, float('inf')
            for j in range(self.idx, min(self.idx + 60, len(self.path))):
                d = math.hypot(self.path[j][0] - x, self.path[j][1] - y)
                if d < best_d:
                    best, best_d = j, d
            self.idx = best
            # look-ahead point
            j = self.idx
            while j < len(self.path) - 1 and self.arc[j] - self.arc[self.idx] < self.args.lookahead:
                j += 1
            tx, ty = self.path[j]
            dist_goal = math.hypot(gx - x, gy - y)

            if dist_goal < POS_TOL:
                self.mode = 'FINAL_ROTATE'
                self.get_logger().info('goal position reached, rotating to the goal heading')
            else:
                err = wrap(math.atan2(ty - y, tx - x) - yaw)
                if self.turning and abs(err) < ROT_EXIT:
                    self.turning = False
                if not self.turning and abs(err) > ROT_ENTER:
                    self.turning = True
                if self.turning:
                    w = clip(K_HEADING * err, -MAX_YAW_RATE, MAX_YAW_RATE)
                    if abs(w) < 0.25:
                        w = math.copysign(0.25, err)
                else:
                    v = self.args.speed * max(0.25, 1.0 - 0.8 * abs(err) / ROT_ENTER)
                    v *= max(0.45, 1.0 - 0.6 * self.turn_ahead() / (math.pi / 2))
                    v = min(v, max(0.12, 0.7 * dist_goal))
                    w = clip(K_HEADING * err, -MAX_YAW_RATE, MAX_YAW_RATE)
                    if self.front_blocked():
                        v = 0.0
                        self.blocked_for += CONTROL_DT
                        if self.blocked_for > BLOCKED_ABORT_S:
                            return self.finish(False, 'something is blocking the way, abort')
                    else:
                        self.blocked_for = 0.0

        elif self.mode == 'FINAL_ROTATE':
            err = wrap(gth - yaw)
            if abs(err) < ANG_TOL:
                self.settled += 1
                if self.settled >= 6:
                    self.mode = 'DONE'
                    return self.finish(True, 'goal pose reached')
            else:
                self.settled = 0
                w = clip(1.5 * err, -0.8, 0.8)
                if abs(w) < 0.15:
                    w = math.copysign(0.15, err)

        v_cmd, w_cmd, wl, wr = self.send(v, w)
        self.rows.append([round(now - self.t_start, 3), x, y, yaw, v_cmd, w_cmd, wl, wr, self.mode])
        self.traj_writer.writerow(self.rows[-1])
        self.traj_file.flush()
        if now - self.last_log > 3.0:
            self.last_log = now
            self.get_logger().info(
                f'[{self.mode}] pos ({x:.1f}, {y:.1f}) yaw {math.degrees(yaw):.0f} deg | '
                f'{self.arc[self.idx]:.1f}/{self.arc[-1]:.1f} m of path | '
                f'wheels L {wl:.1f} R {wr:.1f} rad/s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='plan.json')
    ap.add_argument('--traj-out', default='trajectory.csv')
    ap.add_argument('--speed', type=float, default=0.7, help='max forward speed [m/s]')
    ap.add_argument('--lookahead', type=float, default=0.6, help='look-ahead distance [m]')
    ap.add_argument('--timeout', type=float, default=900.0, help='give up after this many seconds')
    args, _ = ap.parse_known_args()
    with open(args.plan) as f:
        plan = json.load(f)

    rclpy.init()
    node = PathFollower(plan, args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info('interrupted')
        node.exit_code = 130
    finally:
        if rclpy.ok():
            node.send(0.0, 0.0)
        code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
