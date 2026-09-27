"""Top-view renderer / route checker for the Gazebo office world (office_world.world).

Parses the SDF world (box / cylinder / sphere models) into 2D footprints so that the
same geometry can be used to
  * draw a Top View of the scene (compared with the occupancy grid map), and
  * check that a waypoint route keeps a safe clearance from every obstacle.

Usage:
  python world_topview.py --out topview.png
  python world_topview.py --check-route route.yaml --out route_check.png
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_WORLD = os.path.expanduser('~/ros2_ws/src/271411/diffbot_sim/worlds/office_world.world')
LIDAR_Z = 0.685        # height of the LiDAR scan plane above the floor [m]
ROBOT_HALF_DIAG = 0.75  # half diagonal of the 1.0 x 1.0 m body (+ wheels) [m]


class Obj:
    """One primitive: kind in {'box','cyl','sphere'}, pose (x, y, z, yaw) and size."""

    def __init__(self, name, kind, x, y, z, yaw, size, rgba, collision):
        self.name, self.kind = name, kind
        self.x, self.y, self.z, self.yaw = x, y, z, yaw
        self.size = size            # box: (sx,sy,sz)  cyl: (r,len)  sphere: (r,)
        self.rgba = rgba
        self.collision = collision

    @property
    def z_range(self):
        if self.kind == 'box':
            h = self.size[2]
        elif self.kind == 'cyl':
            h = self.size[1]
        else:
            h = 2 * self.size[0]
        return self.z - h / 2, self.z + h / 2

    def seen_by_lidar(self):
        lo, hi = self.z_range
        return self.collision and lo <= LIDAR_Z <= hi

    def corners(self):
        sx, sy = self.size[0] / 2, self.size[1] / 2
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pts = [(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)]
        return [(self.x + px * c - py * s, self.y + px * s + py * c) for px, py in pts]

    def distance(self, px, py):
        """Distance from point to the footprint (negative inside)."""
        if self.kind == 'box':
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            dx, dy = px - self.x, py - self.y
            lx, ly = dx * c + dy * s, -dx * s + dy * c
            qx, qy = abs(lx) - self.size[0] / 2, abs(ly) - self.size[1] / 2
            return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0)
        return math.hypot(px - self.x, py - self.y) - self.size[0]


def _floats(text):
    return [float(v) for v in text.split()]


def _rgba(vis):
    if vis is not None:
        amb = vis.find('material/ambient')
        if amb is not None:
            return tuple(_floats(amb.text))
    return (0.7, 0.7, 0.7, 1.0)


def load_world(path=DEFAULT_WORLD):
    root = ET.parse(path).getroot()
    objs = []
    for model in root.iter('model'):
        pose = model.find('pose')
        if pose is None:
            continue
        x, y, z, _, _, yaw = _floats(pose.text)
        for link in model.iter('link'):
            col = link.find('collision')
            vis = link.find('visual')
            geom_el = (vis if vis is not None else col).find('geometry')
            if geom_el is None:
                continue
            box, cyl, sph = geom_el.find('box'), geom_el.find('cylinder'), geom_el.find('sphere')
            if box is not None:
                kind, size = 'box', tuple(_floats(box.find('size').text))
            elif cyl is not None:
                kind = 'cyl'
                size = (float(cyl.find('radius').text), float(cyl.find('length').text))
            elif sph is not None:
                kind, size = 'sphere', (float(sph.find('radius').text),)
            else:
                continue
            objs.append(Obj(model.get('name'), kind, x, y, z, yaw, size, _rgba(vis), col is not None))
    return objs


def draw_topview(ax, objs, extent=(-16, 16, -16, 16), show_hidden=True):
    """Draw the scene from above. Objects the 2D LiDAR cannot see are drawn lighter."""
    from matplotlib.patches import Circle, Polygon
    ax.set_facecolor('white')
    for o in sorted(objs, key=lambda o: o.z_range[1]):
        hidden = not o.seen_by_lidar()
        if hidden and not show_hidden:
            continue
        color = o.rgba[:3]
        alpha = 0.35 if hidden else 1.0
        if o.kind == 'box':
            ax.add_patch(Polygon(o.corners(), closed=True, fc=color, ec='k', lw=0.3, alpha=alpha))
        else:
            ax.add_patch(Circle((o.x, o.y), o.size[0], fc=color, ec='k', lw=0.3, alpha=alpha))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')


def min_clearance(objs, x0, y0, x1, y1, step=0.05):
    """Smallest distance between the segment and every collision footprint."""
    obst = [o for o in objs if o.collision]
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
    worst, at = float('inf'), None
    for t in np.linspace(0, 1, n):
        px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        for o in obst:
            d = o.distance(px, py)
            if d < worst:
                worst, at = d, (px, py, o.name)
    return worst, at


def check_route(objs, waypoints, wp_clear=ROBOT_HALF_DIAG, seg_clear=0.65):
    ok = True
    for i, (x, y) in enumerate(waypoints):
        d, at = min_clearance(objs, x, y, x, y)
        if d < wp_clear:
            ok = False
            print(f'  waypoint {i} ({x:.2f},{y:.2f}): clearance {d:.2f} < {wp_clear} near {at[2]}')
    for i in range(len(waypoints) - 1):
        (x0, y0), (x1, y1) = waypoints[i], waypoints[i + 1]
        d, at = min_clearance(objs, x0, y0, x1, y1)
        if d < seg_clear:
            ok = False
            print(f'  segment {i}->{i + 1}: clearance {d:.2f} < {seg_clear} at ({at[0]:.2f},{at[1]:.2f}) near {at[2]}')
    return ok


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default=DEFAULT_WORLD)
    ap.add_argument('--check-route', metavar='ROUTE_YAML')
    ap.add_argument('--out', default='topview.png')
    args = ap.parse_args()

    objs = load_world(args.world)
    print(f'{len(objs)} primitives loaded from {args.world}')
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_topview(ax, objs)
    status = 0
    if args.check_route:
        wps = yaml.safe_load(open(args.check_route))['waypoints']
        ok = check_route(objs, wps)
        print('route OK' if ok else 'route has problems')
        status = 0 if ok else 1
        xs, ys = zip(*wps)
        ax.plot(xs, ys, 'r.-', lw=1, ms=5, zorder=10)
        ax.plot(xs[0], ys[0], 'go', ms=8, zorder=11)
    ax.set_title('Top view of the Gazebo scene (faded = not visible to the 2D LiDAR)')
    fig.savefig(args.out, dpi=130)
    print('saved', args.out)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
