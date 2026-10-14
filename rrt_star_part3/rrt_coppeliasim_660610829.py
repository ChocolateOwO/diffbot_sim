# RRT* path planning on the occupancy grid map + driving the robot in Gazebo (ROS 2)
# plan : random start / goal, RRT*, writes plan.json
# drive: follows plan.json, left / right wheel speed -> /cmd_vel

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node as RosNode
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False
    RosNode = object

STUDENT_ID = '660610829'
HERE = os.path.dirname(os.path.abspath(__file__))

LIDAR_Z = 0.685  # lidar height [m]
ROBOT_BOTTOM, ROBOT_TOP = 0.02, 0.72  # robot height range [m]
ROBOT_HALF_DIAG = 0.76  # half diagonal of the robot [m]


# occupancy grid and obstacle map
class GridMap:
    def __init__(self, prob, extent=(-16.0, 16.0, -16.0, 16.0)):
        self.prob = prob
        self.rows, self.cols = prob.shape
        self.x_min, self.x_max, self.y_min, self.y_max = extent
        self.res = (self.x_max - self.x_min) / self.cols
        self.blocked = np.zeros(prob.shape, dtype=bool)
        self.inflated = None

    def to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.x_min) / self.res).astype(int)
        row = np.floor((self.y_max - np.asarray(y)) / self.res).astype(int)
        return row, col

    def cell_center(self, row, col):
        return self.x_min + (col + 0.5) * self.res, self.y_max - (row + 0.5) * self.res

    # unknown cells count as obstacles too
    def block_from_ogm(self, free_thresh):
        self.blocked |= self.prob >= free_thresh

    # draw object shapes into the map
    def block_footprints(self, objs):
        for o in objs:
            r = math.hypot(o.size[0] / 2, o.size[1] / 2) if o.kind == 'box' else o.size[0]
            r0, c0 = self.to_cell(o.x - r, o.y + r)
            r1, c1 = self.to_cell(o.x + r, o.y - r)
            r0, c0 = max(int(r0), 0), max(int(c0), 0)
            r1, c1 = min(int(r1), self.rows - 1), min(int(c1), self.cols - 1)
            if r1 < r0 or c1 < c0:
                continue
            rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            gx, gy = self.cell_center(rr, cc)
            if o.kind == 'box':
                c, s = math.cos(o.yaw), math.sin(o.yaw)
                dx, dy = gx - o.x, gy - o.y
                lx, ly = dx * c + dy * s, -dx * s + dy * c
                hit = (np.abs(lx) <= o.size[0] / 2) & (np.abs(ly) <= o.size[1] / 2)
            else:
                hit = np.hypot(gx - o.x, gy - o.y) <= o.size[0]
            self.blocked[r0:r1 + 1, c0:c1 + 1] |= hit

    # grow obstacles by the robot radius
    def inflate(self, radius):
        n = radius / self.res
        k = int(math.ceil(n))
        dys, dxs = np.mgrid[-k:k + 1, -k:k + 1]
        offsets = [(int(dy), int(dx)) for dy, dx in zip(dys.ravel(), dxs.ravel())
                   if dy * dy + dx * dx <= n * n]
        padded = np.pad(self.blocked, k, constant_values=True)
        out = np.zeros_like(self.blocked)
        for dy, dx in offsets:
            out |= padded[k + dy:k + dy + self.rows, k + dx:k + dx + self.cols]
        self.inflated = out
        return out

    def points_free(self, xs, ys):
        r, c = self.to_cell(xs, ys)
        inside = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
        ok = np.zeros(np.shape(r), dtype=bool)
        ok[inside] = ~self.inflated[r[inside], c[inside]]
        return ok

    # biggest connected free area
    def largest_free_region(self):
        free = ~self.inflated
        seen = np.zeros(free.shape, dtype=bool)
        best = None
        for r0, c0 in np.argwhere(free):
            if seen[r0, c0]:
                continue
            comp = [(r0, c0)]
            seen[r0, c0] = True
            q = deque(comp)
            while q:
                r, c = q.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.rows and 0 <= cc < self.cols and free[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            comp.append((rr, cc))
                            q.append((rr, cc))
            if best is None or len(comp) > len(best):
                best = comp
        mask = np.zeros(free.shape, dtype=bool)
        idx = np.array(best)
        mask[idx[:, 0], idx[:, 1]] = True
        return mask


# one box / cylinder / sphere from the world file
class Obj:
    def __init__(self, kind, x, y, z, yaw, size, rgba, collision):
        self.kind, self.x, self.y, self.z, self.yaw = kind, x, y, z, yaw
        self.size, self.rgba, self.collision = size, rgba, collision

    @property
    def z_range(self):
        h = self.size[2] if self.kind == 'box' else self.size[1] if self.kind == 'cyl' else 2 * self.size[0]
        return self.z - h / 2, self.z + h / 2

    def seen_by_lidar(self):
        lo, hi = self.z_range
        return self.collision and lo <= LIDAR_Z <= hi

    def blocks_robot_body(self):
        lo, hi = self.z_range
        return self.collision and lo < ROBOT_TOP and hi > ROBOT_BOTTOM

    def corners(self):
        sx, sy = self.size[0] / 2, self.size[1] / 2
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return [(self.x + px * c - py * s, self.y + px * s + py * c)
                for px, py in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))]

    def distance(self, px, py):
        if self.kind == 'box':
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            dx, dy = px - self.x, py - self.y
            lx, ly = dx * c + dy * s, -dx * s + dy * c
            qx, qy = abs(lx) - self.size[0] / 2, abs(ly) - self.size[1] / 2
            return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0)
        return math.hypot(px - self.x, py - self.y) - self.size[0]


# read the .world file
def load_world(path):
    floats = lambda t: [float(v) for v in t.split()]
    objs = []
    for model in ET.parse(path).getroot().iter('model'):
        pose = model.find('pose')
        if pose is None:
            continue
        x, y, z, _, _, yaw = floats(pose.text)
        for link in model.iter('link'):
            col, vis = link.find('collision'), link.find('visual')
            geom = (vis if vis is not None else col).find('geometry')
            if geom is None:
                continue
            box, cyl, sph = geom.find('box'), geom.find('cylinder'), geom.find('sphere')
            if box is not None:
                kind, size = 'box', tuple(floats(box.find('size').text))
            elif cyl is not None:
                kind, size = 'cyl', (float(cyl.find('radius').text), float(cyl.find('length').text))
            elif sph is not None:
                kind, size = 'sphere', (float(sph.find('radius').text),)
            else:
                continue
            rgba = (0.7, 0.7, 0.7, 1.0)
            if vis is not None and vis.find('material/ambient') is not None:
                rgba = tuple(floats(vis.find('material/ambient').text))
            objs.append(Obj(kind, x, y, z, yaw, size, rgba, col is not None))
    return objs


# rrt* node, same fields as the earlier rrt file
class Node:
    def __init__(self, x, y, index=0):
        self.index = index
        self.position_x = x
        self.position_y = y
        self.parent_nodes = None
        self.distance_to_parent = 0.0
        self.distance_from_start = 0.0
        self.path_x = np.array([x])
        self.path_y = np.array([y])


# rrt* on the grid map
class RRTStar:
    def __init__(self, start, goal, grid, sample_cells, expand_dis=1.0, goal_sample_rate=10,
                 max_iter=5000, path_resolution=0.1, rng=None):
        self.start = Node(start[0], start[1], index=0)
        self.end = Node(goal[0], goal[1], index=-1)
        self.grid = grid
        self.sample_cells = sample_cells
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.path_resolution = path_resolution
        self.rng = rng or random.Random()
        self.node_list = [self.start]
        self.children = {0: []}
        cap = max_iter + 16
        self._xs = np.empty(cap)
        self._ys = np.empty(cap)
        self._xs[0], self._ys[0] = start
        self.iterations_done = 0

    def _register(self, node):
        node.index = len(self.node_list)
        if node.index >= len(self._xs):
            self._xs = np.concatenate([self._xs, np.empty(len(self._xs))])
            self._ys = np.concatenate([self._ys, np.empty(len(self._ys))])
        self.node_list.append(node)
        self._xs[node.index], self._ys[node.index] = node.position_x, node.position_y
        self.children[node.index] = []
        if node.parent_nodes is not None:
            self.children[node.parent_nodes.index].append(node.index)

    def check_obstacle(self, node):
        return node is not None and bool(self.grid.points_free(node.path_x, node.path_y).all())

    # random free cell, sometimes the goal
    def populate_node(self):
        if self.rng.randint(0, 100) > self.goal_sample_rate:
            r, c = self.sample_cells[self.rng.randrange(len(self.sample_cells))]
            x, y = self.grid.cell_center(r, c)
            res = self.grid.res
            return x + self.rng.uniform(-res / 2, res / 2), y + self.rng.uniform(-res / 2, res / 2)
        return self.end.position_x, self.end.position_y

    def find_node_nearest(self, x, y):
        n = len(self.node_list)
        d2 = (self._xs[:n] - x) ** 2 + (self._ys[:n] - y) ** 2
        return self.node_list[int(np.argmin(d2))]

    # move from from_node toward the target, max expand_dis
    def steer(self, from_node, to_x, to_y, extend_length=float('inf')):
        d = math.hypot(to_x - from_node.position_x, to_y - from_node.position_y)
        theta = math.atan2(to_y - from_node.position_y, to_x - from_node.position_x)
        length = min(extend_length, d)
        n_expand = int(math.floor(length / self.path_resolution))
        k = np.arange(n_expand + 1) * self.path_resolution
        px = from_node.position_x + k * math.cos(theta)
        py = from_node.position_y + k * math.sin(theta)
        if math.hypot(px[-1] - to_x, py[-1] - to_y) <= self.path_resolution:
            px, py = np.append(px, to_x), np.append(py, to_y)
            x_new, y_new = to_x, to_y
        else:
            x_new, y_new = float(px[-1]), float(py[-1])
        node = Node(x_new, y_new)
        node.path_x, node.path_y = px, py
        node.parent_nodes = from_node
        node.distance_to_parent = math.hypot(x_new - from_node.position_x, y_new - from_node.position_y)
        node.distance_from_start = from_node.distance_from_start + node.distance_to_parent
        return node

    def find_node_within_radius(self, x, y):
        n = len(self.node_list) + 1
        r = min(50.0 * math.sqrt(math.log(n) / n), self.expand_dis * 5.0)
        m = len(self.node_list)
        d2 = (self._xs[:m] - x) ** 2 + (self._ys[:m] - y) ** 2
        return np.nonzero(d2 <= r * r)[0].tolist()

    # cheapest parent without collision
    def choose_parent(self, new_node, near_inds):
        costs = []
        for i in near_inds:
            nb = self.node_list[i]
            d = math.hypot(new_node.position_x - nb.position_x, new_node.position_y - nb.position_y)
            costs.append((nb.distance_from_start + d, i))
        for _, i in sorted(costs):
            t_node = self.steer(self.node_list[i], new_node.position_x, new_node.position_y)
            if self.check_obstacle(t_node):
                return t_node
        return new_node

    # reconnect near nodes if the new node gives a shorter way
    def rewire(self, new_node, near_inds):
        for i in near_inds:
            nb = self.node_list[i]
            if nb is new_node.parent_nodes:
                continue
            d = math.hypot(nb.position_x - new_node.position_x, nb.position_y - new_node.position_y)
            if nb.distance_from_start <= new_node.distance_from_start + d + 1e-9:
                continue
            edge = self.steer(new_node, nb.position_x, nb.position_y)
            if not self.check_obstacle(edge):
                continue
            self.children[nb.parent_nodes.index].remove(nb.index)
            nb.parent_nodes = new_node
            self.children[new_node.index].append(nb.index)
            nb.distance_to_parent = edge.distance_to_parent
            nb.distance_from_start = edge.distance_from_start
            nb.path_x, nb.path_y = edge.path_x, edge.path_y
            self.propagate_cost_to_leaves(nb)

    # update the cost of all children
    def propagate_cost_to_leaves(self, parent_node):
        stack = [parent_node]
        while stack:
            p = stack.pop()
            for ci in self.children[p.index]:
                c = self.node_list[ci]
                c.distance_to_parent = math.hypot(c.position_x - p.position_x, c.position_y - p.position_y)
                c.distance_from_start = p.distance_from_start + c.distance_to_parent
                stack.append(c)

    # best node that can reach the goal
    def search_best_goal_node(self):
        m = len(self.node_list)
        d = np.hypot(self._xs[:m] - self.end.position_x, self._ys[:m] - self.end.position_y)
        best, best_cost = None, float('inf')
        for i in np.nonzero(d <= self.expand_dis)[0]:
            t_node = self.steer(self.node_list[int(i)], self.end.position_x, self.end.position_y)
            if self.check_obstacle(t_node) and t_node.distance_from_start < best_cost:
                best, best_cost = int(i), t_node.distance_from_start
        return best

    # nodes from start to goal
    def generate_final_course(self, goal_ind):
        path = [[self.end.position_x, self.end.position_y]]
        node = self.node_list[goal_ind]
        while node is not None:
            path.append([node.position_x, node.position_y])
            node = node.parent_nodes
        return path[::-1]

    # keep going in batches until a path is found
    def planning(self, batch=2000, max_total=30000):
        total = 0
        todo = self.max_iter
        while True:
            for _ in range(todo):
                total += 1
                x, y = self.populate_node()
                nearest = self.find_node_nearest(x, y)
                new_node = self.steer(nearest, x, y, self.expand_dis)
                if not self.check_obstacle(new_node):
                    continue
                near_inds = self.find_node_within_radius(new_node.position_x, new_node.position_y)
                new_node = self.choose_parent(new_node, near_inds)
                self._register(new_node)
                self.rewire(new_node, near_inds)
            self.iterations_done = total
            last = self.search_best_goal_node()
            if last is not None:
                final = self.steer(self.node_list[last], self.end.position_x, self.end.position_y)
                final.index = -1
                self.end.parent_nodes = self.node_list[last]
                self.end.distance_from_start = final.distance_from_start
                return self.generate_final_course(last)
            if total >= max_total:
                return None
            todo = batch
            if len(self.node_list) + batch >= len(self._xs):
                self._xs = np.concatenate([self._xs, np.empty(batch + 16)])
                self._ys = np.concatenate([self._ys, np.empty(batch + 16)])


# random pose on a free cell
def random_pose(rng, grid, region_cells):
    r, c = region_cells[rng.randrange(len(region_cells))]
    x, y = grid.cell_center(r, c)
    return [x + rng.uniform(-grid.res / 2, grid.res / 2),
            y + rng.uniform(-grid.res / 2, grid.res / 2),
            rng.uniform(-math.pi, math.pi)]


def path_length(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


# distance from the path to the real obstacle shapes
def world_clearance(path, objs, step=0.1):
    obst = [o for o in objs if o.blocks_robot_body()]
    worst, where = float('inf'), None
    for i in range(len(path) - 1):
        (x0, y0), (x1, y1) = path[i], path[i + 1]
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
        for t in np.linspace(0, 1, n):
            px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            for o in obst:
                d = o.distance(px, py)
                if d < worst:
                    worst, where = d, (px, py)
    return worst, where


def draw_topview(ax, objs, extent):
    from matplotlib.patches import Circle, Polygon
    for o in sorted(objs, key=lambda o: o.z_range[1]):
        alpha = 1.0 if o.seen_by_lidar() else 0.45
        if o.kind == 'box':
            ax.add_patch(Polygon(o.corners(), closed=True, fc=o.rgba[:3], ec='k', lw=0.3, alpha=alpha))
        else:
            ax.add_patch(Circle((o.x, o.y), o.size[0], fc=o.rgba[:3], ec='k', lw=0.3, alpha=alpha))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')


def save_plot(png, grid, planner, path, start, goal, objs, title):
    import matplotlib.pyplot as plt
    ext = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8))

    a1.imshow(grid.prob, cmap='gray_r', vmin=0, vmax=1, origin='upper', extent=ext, interpolation='nearest')
    shade = np.ma.masked_where(~grid.inflated, np.ones(grid.inflated.shape))
    a1.imshow(shade, cmap='autumn', vmin=0, vmax=2, alpha=0.25, origin='upper', extent=ext,
              interpolation='nearest')
    for nd in planner.node_list:
        if nd.parent_nodes is not None:
            a1.plot(nd.path_x, nd.path_y, '-', color='c', lw=0.4, alpha=0.6)
    p = np.array(path)
    a1.plot(p[:, 0], p[:, 1], 'b-', lw=2.5, label='RRT* path')
    a1.plot(p[:, 0], p[:, 1], 'b.', ms=6)
    for pose, col, name in ((start, 'g', 'start'), (goal, 'r', 'goal')):
        a1.plot(pose[0], pose[1], 'o', color=col, ms=9, label=name)
        a1.arrow(pose[0], pose[1], 1.4 * math.cos(pose[2]), 1.4 * math.sin(pose[2]), color=col,
                 width=0.12, head_width=0.5, length_includes_head=True, zorder=6)
    a1.set_title(title + '\n(grey = OGM, orange = cells the robot centre can not use)')
    a1.legend(loc='upper right', fontsize=8)

    draw_topview(a2, objs, ext)
    a2.plot(p[:, 0], p[:, 1], 'b-', lw=2.5)
    for pose, col in ((start, 'g'), (goal, 'r')):
        a2.plot(pose[0], pose[1], 'o', color=col, ms=9)
        a2.arrow(pose[0], pose[1], 1.4 * math.cos(pose[2]), 1.4 * math.sin(pose[2]), color=col,
                 width=0.12, head_width=0.5, length_includes_head=True, zorder=6)
    a2.set_title('Top view of the Gazebo world with the same path')
    for a in (a1, a2):
        a.set_xlabel('x [m]')
        a.set_ylabel('y [m]')
        a.set_xlim(ext[0], ext[1])
        a.set_ylim(ext[2], ext[3])
        a.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    return fig


WHEEL_RADIUS = 0.15  # [m]
WHEEL_SEPARATION = 1.08  # [m]
MAX_WHEEL_SPEED = 6.0  # [rad/s]

CONTROL_DT = 0.05
INTERP_STEP = 0.1
POS_TOL = 0.10  # goal position tolerance [m]
ANG_TOL = 0.05  # goal heading tolerance [rad]
ROT_ENTER, ROT_EXIT = 0.8, 0.25  # turn in place above / below this heading error [rad]
K_HEADING = 1.8
MAX_YAW_RATE = 0.9
SAFE_X, SAFE_HALF_Y = 0.70, 0.52  # stop box in front of the lidar [m]
BLOCKED_ABORT_S = 10.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# points every 0.1 m between the nodes
def interpolate_path(nodes, step=INTERP_STEP):
    pts = []
    for (x0, y0), (x1, y1) in zip(nodes[:-1], nodes[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        pts.extend((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n))
    pts.append(tuple(nodes[-1]))
    return pts


# body speed -> left / right wheel speed [rad/s]
def wheel_speeds(v, w):
    wl = (v - w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    wr = (v + w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    big = max(abs(wl), abs(wr))
    if big > MAX_WHEEL_SPEED:
        wl, wr = wl * MAX_WHEEL_SPEED / big, wr * MAX_WHEEL_SPEED / big
    return wl, wr


# wheel speeds -> body speed
def body_speed(wl, wr):
    return (WHEEL_RADIUS * (wl + wr) / 2.0,
            WHEEL_RADIUS * (wr - wl) / WHEEL_SEPARATION)


# follow the path in Gazebo
class PathFollower(RosNode):
    def __init__(self, plan, args):
        super().__init__('path_follower')
        self.args = args
        self.goal = plan['goal']
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
        self.traj_file = open(args.traj_out, 'w', newline='')
        self.traj_writer = csv.writer(self.traj_file)
        self.traj_writer.writerow(['t', 'x', 'y', 'yaw', 'v_cmd', 'w_cmd', 'wheel_left', 'wheel_right', 'mode'])
        self.traj_file.flush()

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_timer(CONTROL_DT, self.control)

    def odom_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

    def scan_cb(self, msg):
        self.scan = msg

    # something close in front of the robot
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

    def send(self, v, w):
        wl, wr = wheel_speeds(v, w)
        v_cmd, w_cmd = body_speed(wl, wr)
        msg = Twist()
        msg.linear.x, msg.angular.z = v_cmd, w_cmd
        self.pub.publish(msg)
        return v_cmd, w_cmd, wl, wr

    # how much the path turns in the next 1.5 m
    def turn_ahead(self):
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

    # TRACK: follow the path, FINAL_ROTATE: turn to the goal heading
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
            best, best_d = self.idx, float('inf')
            for j in range(self.idx, min(self.idx + 60, len(self.path))):
                d = math.hypot(self.path[j][0] - x, self.path[j][1] - y)
                if d < best_d:
                    best, best_d = j, d
            self.idx = best
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


# step 1: plan
def plan_main():
    ap = argparse.ArgumentParser(prog='rrt_coppeliasim_660610829.py plan')
    ap.add_argument('--map', default=os.path.join(HERE, f'ocgm_{STUDENT_ID}.csv'))
    ap.add_argument('--world', default=os.path.join(HERE, f'rrt_coppeliasim_{STUDENT_ID}.world'))
    ap.add_argument('--out', default=os.path.join(HERE, 'plan.json'))
    ap.add_argument('--png', default=os.path.join(HERE, 'plan.png'))
    ap.add_argument('--nodes-csv', default=os.path.join(HERE, 'path_nodes.csv'))
    ap.add_argument('--seed', type=int, default=None, help='random seed (default: new random one)')
    ap.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'TH'), help='fixed start pose')
    ap.add_argument('--goal', type=float, nargs=3, metavar=('X', 'Y', 'TH'), help='fixed goal pose')
    ap.add_argument('--min-dist', type=float, default=10.0, help='min start-goal distance [m]')
    ap.add_argument('--robot-radius', type=float, default=0.85, help='obstacle growing radius [m]')
    ap.add_argument('--free-thresh', type=float, default=0.45,
                    help='cell is free when occupancy probability is below this')
    ap.add_argument('--expand-dis', type=float, default=1.0, help='max edge length [m]')
    ap.add_argument('--goal-sample-rate', type=int, default=10, help='percent of samples that are the goal')
    ap.add_argument('--max-iter', type=int, default=5000)
    ap.add_argument('--ogm-only', action='store_true', help='do not add blind obstacles from the world file')
    ap.add_argument('--no-show', action='store_true', help='save files only, do not open the plot window')
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use('Agg' if args.no_show else 'TkAgg')
        plotting = True
    except Exception as e:
        plotting = False
        print(f'(no plots: matplotlib not usable here: {e}; '
              'use ~/ocgm_venv/bin/python for the picture)')

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 10 ** 6)
    rng = random.Random(seed)
    print(f'seed = {seed}')

    prob = np.loadtxt(args.map, delimiter=',')
    grid = GridMap(prob)
    print(f'loaded {os.path.basename(args.map)}: {prob.shape[0]} x {prob.shape[1]} cells, '
          f'{grid.res} m/cell, x {grid.x_min}..{grid.x_max}, y {grid.y_min}..{grid.y_max}')

    grid.block_from_ogm(args.free_thresh)
    objs = load_world(args.world) if os.path.exists(args.world) else []
    if not args.ogm_only and objs:
        blind = [o for o in objs if o.blocks_robot_body() and not o.seen_by_lidar()]
        grid.block_footprints(blind)
        print(f'added {len(blind)} objects the 2D LiDAR can not see (sofa, cones, seats, car bodies ...)')
    grid.inflate(args.robot_radius)
    region = grid.largest_free_region()
    cells = np.argwhere(region)
    print(f'robot radius {args.robot_radius} m -> {int(region.sum())} usable cells '
          f'({region.sum() * grid.res ** 2:.0f} m^2)')

    def valid(p):
        return bool(grid.points_free(np.array([p[0]]), np.array([p[1]]))[0] and
                    region[grid.to_cell(p[0], p[1])])

    if args.start:
        start = list(args.start)
        if not valid(start):
            raise SystemExit(f'start pose {start} is on a cell the robot can not use')
    else:
        start = random_pose(rng, grid, cells)
    if args.goal:
        goal = list(args.goal)
        if not valid(goal):
            raise SystemExit(f'goal pose {goal} is on a cell the robot can not use')
    else:
        for _ in range(1000):
            goal = random_pose(rng, grid, cells)
            if math.hypot(goal[0] - start[0], goal[1] - start[1]) >= args.min_dist:
                break
    print(f'start = ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f} rad)')
    print(f'goal  = ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f} rad)')

    planner = RRTStar(start[:2], goal[:2], grid, cells, expand_dis=args.expand_dis,
                      goal_sample_rate=args.goal_sample_rate, max_iter=args.max_iter, rng=rng)
    t0 = time.time()
    path = planner.planning()
    dt = time.time() - t0
    if path is None:
        raise SystemExit(f'RRT* found no path after {planner.iterations_done} iterations')
    length = path_length(path)
    straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
    print(f'RRT*: {planner.iterations_done} iterations, {len(planner.node_list)} nodes in the tree, '
          f'{dt:.1f} s')
    print(f'path: {len(path)} nodes, length {length:.1f} m (straight line {straight:.1f} m)')

    if objs:
        clr, at = world_clearance(path, objs)
        verdict = 'OK' if clr >= ROBOT_HALF_DIAG else 'TOO CLOSE'
        print(f'check against the world file: closest obstacle {clr:.2f} m from the path '
              f'at ({at[0]:.1f}, {at[1]:.1f})  [{verdict}, robot half diagonal {ROBOT_HALF_DIAG} m]')

    plan = {
        'student_id': STUDENT_ID, 'seed': seed, 'map': os.path.basename(args.map),
        'extent': [grid.x_min, grid.x_max, grid.y_min, grid.y_max], 'resolution': grid.res,
        'robot_radius': args.robot_radius, 'free_thresh': args.free_thresh,
        'blind_obstacles_added': (not args.ogm_only) and bool(objs),
        'start': start, 'goal': goal,
        'expand_dis': args.expand_dis, 'iterations': planner.iterations_done,
        'tree_nodes': len(planner.node_list), 'planning_time_s': round(dt, 2),
        'path_length_m': round(length, 3), 'path': path,
    }
    with open(args.out, 'w') as f:
        json.dump(plan, f, indent=1)
    with open(args.nodes_csv, 'w') as f:
        f.write('index,x,y\n')
        for i, (x, y) in enumerate(path):
            f.write(f'{i},{x:.4f},{y:.4f}\n')
    title = (f'RRT* on the OGM - seed {seed}, {len(path)} nodes, {length:.1f} m, '
             f'{planner.iterations_done} iterations')
    if plotting:
        fig = save_plot(args.png, grid, planner, path, start, goal, objs, title)
        print(f'saved {args.out}, {args.nodes_csv}, {args.png}')
    else:
        print(f'saved {args.out}, {args.nodes_csv}')
    print('\nnode  x [m]    y [m]')
    for i, (x, y) in enumerate(path):
        print(f'{i:4d}  {x:7.2f}  {y:7.2f}')

    if plotting and not args.no_show:
        import matplotlib.pyplot as plt
        print('showing the plan (close the window to exit)')
        plt.show()


# step 2: drive
def drive_main():
    if not HAVE_ROS:
        raise SystemExit('ROS 2 python (rclpy) not found: run  source /opt/ros/humble/setup.bash  '
                         'and use the system python3')
    ap = argparse.ArgumentParser(prog='rrt_coppeliasim_660610829.py drive')
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


# usage: python rrt_coppeliasim_660610829.py plan | drive
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('plan', 'drive'):
        print('usage: python rrt_coppeliasim_660610829.py plan  [options]   # RRT* planning (offline)\n'
              '       python3 rrt_coppeliasim_660610829.py drive [options]   # drive the plan in Gazebo\n'
              '       add -h after plan / drive for the options')
        raise SystemExit(1)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    plan_main() if cmd == 'plan' else drive_main()


if __name__ == '__main__':
    main()
