"""RRT* path planning (offline) on the Occupancy Grid Map built from the Gazebo LiDAR data.

What it does (in this order):
  1. load the occupancy grid map  ocgm_660610829.csv   (0 = free, 1 = occupied, 0.5 = unknown)
  2. build the "robot can not be here" map: occupied + unknown cells, grown by the robot radius
  3. pick a RANDOM start pose (x, y, theta) and a RANDOM goal pose (x, y, theta) that are not on
     a cell the robot can not reach
  4. run RRT* and get the ordered list of nodes the robot should drive through
  5. save   plan.json (nodes + start + goal)   plan.png (picture)   path_nodes.csv

The class layout (Node / RRT / RRTStar, steer, choose_parent, rewire, search_best_goal_node,
propagate_cost_to_leaves) is the same as in rrt_660610829.py from the RRT / RRT* step; the
obstacle test now looks the cells up in the occupancy grid instead of using rectangles, and the
neighbour searches are done with numpy so a 32 x 32 m map with thousands of nodes stays fast.

Run:   python rrt_star_planner_660610829.py                 (random start / goal)
       python rrt_star_planner_660610829.py --seed 7        (same random poses again)
       python rrt_star_planner_660610829.py --start -10 -8 0.5 --goal 10 8 -1.2
       python rrt_star_planner_660610829.py --ogm-only      (plan on the OGM only, see below)

The 2D LiDAR plane is 0.685 m above the floor, so the OGM does not contain low things such as the
sofa, traffic cones, chair seats or car bodies, but the 1 m robot body would hit them. By default
those objects (read from myworld_660610829.world) are added to the "can not go" map as well.
--ogm-only switches this off and plans on the OGM alone.
"""
import argparse
import json
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

STUDENT_ID = '660610829'
HERE = os.path.dirname(os.path.abspath(__file__))

LIDAR_Z = 0.685                              # height of the LiDAR scan plane [m]
ROBOT_BOTTOM, ROBOT_TOP = 0.02, 0.72         # vertical extent of the robot [m]
ROBOT_HALF_DIAG = 0.76                       # half diagonal of the body incl. wheels [m]


# ==============================================================================
# 1. OCCUPANCY GRID + "can not go here" map
# ==============================================================================

class GridMap:
    """Occupancy grid. prob[row, col]; row 0 = top of the map (y max), col 0 = x min."""

    def __init__(self, prob, extent=(-16.0, 16.0, -16.0, 16.0)):
        self.prob = prob
        self.rows, self.cols = prob.shape
        self.x_min, self.x_max, self.y_min, self.y_max = extent
        self.res = (self.x_max - self.x_min) / self.cols
        self.blocked = np.zeros(prob.shape, dtype=bool)     # occupied / unknown / blind obstacles
        self.inflated = None                                # blocked, grown by the robot radius

    # ---- coordinates --------------------------------------------------------
    def to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.x_min) / self.res).astype(int)
        row = np.floor((self.y_max - np.asarray(y)) / self.res).astype(int)
        return row, col

    def cell_center(self, row, col):
        return self.x_min + (col + 0.5) * self.res, self.y_max - (row + 0.5) * self.res

    # ---- building the map ---------------------------------------------------
    def block_from_ogm(self, free_thresh):
        """A cell is only 'free' if the LiDAR has seen it free (prob < free_thresh).
        Occupied cells and never-observed cells (prob = 0.5) are both treated as obstacles."""
        self.blocked |= self.prob >= free_thresh

    def block_footprints(self, objs):
        """Rasterise obstacle footprints (boxes / circles) into the blocked map."""
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

    def inflate(self, radius):
        """Grow every blocked cell by the robot radius (the map border counts as blocked)."""
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

    # ---- queries ------------------------------------------------------------
    def points_free(self, xs, ys):
        """True for every point that lies on a cell the robot centre may occupy."""
        r, c = self.to_cell(xs, ys)
        inside = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
        ok = np.zeros(np.shape(r), dtype=bool)
        ok[inside] = ~self.inflated[r[inside], c[inside]]
        return ok

    def largest_free_region(self):
        """Boolean mask of the biggest 8-connected free region (so start and goal can connect)."""
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


# ==============================================================================
# 2. WORLD FILE (only used for the blind obstacles, the top view and the final check)
# ==============================================================================

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
        """Distance from a point to the footprint (negative inside)."""
        if self.kind == 'box':
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            dx, dy = px - self.x, py - self.y
            lx, ly = dx * c + dy * s, -dx * s + dy * c
            qx, qy = abs(lx) - self.size[0] / 2, abs(ly) - self.size[1] / 2
            return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0)
        return math.hypot(px - self.x, py - self.y) - self.size[0]


def load_world(path):
    floats = lambda t: [float(v) for v in t.split()]          # noqa: E731
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


# ==============================================================================
# 3. RRT / RRT*  (same structure as rrt_660610829.py)
# ==============================================================================

class Node:
    """Node variables as in the slides."""

    def __init__(self, x, y, index=0):
        self.index = index
        self.position_x = x
        self.position_y = y
        self.parent_nodes = None
        self.distance_to_parent = 0.0
        self.distance_from_start = 0.0
        self.path_x = np.array([x])        # points of the edge parent -> this node
        self.path_y = np.array([y])


class RRTStar:
    def __init__(self, start, goal, grid, sample_cells, expand_dis=1.0, goal_sample_rate=10,
                 max_iter=5000, path_resolution=0.1, rng=None):
        self.start = Node(start[0], start[1], index=0)
        self.end = Node(goal[0], goal[1], index=-1)
        self.grid = grid
        self.sample_cells = sample_cells            # (N, 2) rows/cols of the free region
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

    # ---- helpers -----------------------------------------------------------
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
        """True when every point of the edge lies on a free cell."""
        return node is not None and bool(self.grid.points_free(node.path_x, node.path_y).all())

    # ---- RRT steps ---------------------------------------------------------
    def populate_node(self):
        """Random node: a random free cell (goal with probability goal_sample_rate %)."""
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

    def steer(self, from_node, to_x, to_y, extend_length=float('inf')):
        d = math.hypot(to_x - from_node.position_x, to_y - from_node.position_y)
        theta = math.atan2(to_y - from_node.position_y, to_x - from_node.position_x)
        length = min(extend_length, d)
        n_expand = int(math.floor(length / self.path_resolution))
        k = np.arange(n_expand + 1) * self.path_resolution
        px = from_node.position_x + k * math.cos(theta)
        py = from_node.position_y + k * math.sin(theta)
        if math.hypot(px[-1] - to_x, py[-1] - to_y) <= self.path_resolution:
            px, py = np.append(px, to_x), np.append(py, to_y)       # reached the target point
            x_new, y_new = to_x, to_y
        else:
            x_new, y_new = float(px[-1]), float(py[-1])
        node = Node(x_new, y_new)
        node.path_x, node.path_y = px, py
        node.parent_nodes = from_node
        node.distance_to_parent = math.hypot(x_new - from_node.position_x, y_new - from_node.position_y)
        node.distance_from_start = from_node.distance_from_start + node.distance_to_parent
        return node

    # ---- RRT* extras -------------------------------------------------------
    def find_node_within_radius(self, x, y):
        n = len(self.node_list) + 1
        r = min(50.0 * math.sqrt(math.log(n) / n), self.expand_dis * 5.0)
        m = len(self.node_list)
        d2 = (self._xs[:m] - x) ** 2 + (self._ys[:m] - y) ** 2
        return np.nonzero(d2 <= r * r)[0].tolist()

    def choose_parent(self, new_node, near_inds):
        """Cheapest collision-free parent among the near nodes (checked cheapest first)."""
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

    def propagate_cost_to_leaves(self, parent_node):
        stack = [parent_node]
        while stack:
            p = stack.pop()
            for ci in self.children[p.index]:
                c = self.node_list[ci]
                c.distance_to_parent = math.hypot(c.position_x - p.position_x, c.position_y - p.position_y)
                c.distance_from_start = p.distance_from_start + c.distance_to_parent
                stack.append(c)

    def search_best_goal_node(self):
        m = len(self.node_list)
        d = np.hypot(self._xs[:m] - self.end.position_x, self._ys[:m] - self.end.position_y)
        best, best_cost = None, float('inf')
        for i in np.nonzero(d <= self.expand_dis)[0]:
            t_node = self.steer(self.node_list[int(i)], self.end.position_x, self.end.position_y)
            if self.check_obstacle(t_node) and t_node.distance_from_start < best_cost:
                best, best_cost = int(i), t_node.distance_from_start
        return best

    def generate_final_course(self, goal_ind):
        """Node positions in driving order: start ... goal."""
        path = [[self.end.position_x, self.end.position_y]]
        node = self.node_list[goal_ind]
        while node is not None:
            path.append([node.position_x, node.position_y])
            node = node.parent_nodes
        return path[::-1]

    # ---- main loop ---------------------------------------------------------
    def planning(self, batch=2000, max_total=30000):
        """Run RRT*; if no path to the goal exists yet after max_iter, keep going in batches."""
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


# ==============================================================================
# 4. RANDOM POSES, CHECKS, PLOTS
# ==============================================================================

def random_pose(rng, grid, region_cells):
    r, c = region_cells[rng.randrange(len(region_cells))]
    x, y = grid.cell_center(r, c)
    return [x + rng.uniform(-grid.res / 2, grid.res / 2),
            y + rng.uniform(-grid.res / 2, grid.res / 2),
            rng.uniform(-math.pi, math.pi)]


def path_length(path):
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def world_clearance(path, objs, step=0.1):
    """Independent check against the real world geometry: smallest distance between the path
    and any obstacle that touches the robot's height range."""
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


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', default=os.path.join(HERE, f'ocgm_{STUDENT_ID}.csv'))
    ap.add_argument('--world', default=os.path.join(HERE, f'myworld_{STUDENT_ID}.world'))
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

    import matplotlib
    matplotlib.use('Agg' if args.no_show else 'TkAgg')

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 10 ** 6)
    rng = random.Random(seed)
    print(f'seed = {seed}')

    # --- 1. load the OGM ---------------------------------------------------------
    prob = np.loadtxt(args.map, delimiter=',')
    grid = GridMap(prob)
    print(f'loaded {os.path.basename(args.map)}: {prob.shape[0]} x {prob.shape[1]} cells, '
          f'{grid.res} m/cell, x {grid.x_min}..{grid.x_max}, y {grid.y_min}..{grid.y_max}')

    # --- 2. cells the robot can not use ------------------------------------------
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

    # --- 3. random start / goal pose ---------------------------------------------
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

    # --- 4. RRT* -------------------------------------------------------------------
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

    # --- 5. save ---------------------------------------------------------------------
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
    fig = save_plot(args.png, grid, planner, path, start, goal, objs, title)
    print(f'saved {args.out}, {args.nodes_csv}, {args.png}')
    print('\nnode  x [m]    y [m]')
    for i, (x, y) in enumerate(path):
        print(f'{i:4d}  {x:7.2f}  {y:7.2f}')

    if not args.no_show:
        import matplotlib.pyplot as plt
        print('showing the plan (close the window to exit)')
        plt.show()


if __name__ == '__main__':
    main()
