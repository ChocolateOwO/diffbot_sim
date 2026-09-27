import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

STUDENT_ID = '660610829'
HERE = os.path.dirname(os.path.abspath(__file__))
WORLD_FILE = os.path.join(HERE, f'myworld_{STUDENT_ID}.world')
LIDAR_Z = 0.685          # height of the LiDAR scan plane above the floor [m]

# ==============================================================================
# 1. BRESENHAM'S LINE ALGORITHM
# ==============================================================================


def plotLineLow(x0, y0, x1, y1):
    points = []
    dx, dy = x1 - x0, y1 - y0
    yi = 1
    if dy < 0:
        yi, dy = -1, -dy
    D = 2 * dy - dx
    y = y0
    for x in range(x0, x1 + 1):
        points.append((x, y))
        if D > 0:
            y += yi
            D += 2 * (dy - dx)
        else:
            D += 2 * dy
    return points


def plotLineHigh(x0, y0, x1, y1):
    points = []
    dx, dy = x1 - x0, y1 - y0
    xi = 1
    if dx < 0:
        xi, dx = -1, -dx
    D = 2 * dx - dy
    x = x0
    for y in range(y0, y1 + 1):
        points.append((x, y))
        if D > 0:
            x += xi
            D += 2 * (dx - dy)
        else:
            D += 2 * dx
    return points


def bresenham(x0, y0, x1, y1):
    """Cells from (x0, y0) to (x1, y1), always ordered from the start to the end."""
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            pts = plotLineLow(x1, y1, x0, y0)
            pts.reverse()
        else:
            pts = plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            pts = plotLineHigh(x1, y1, x0, y0)
            pts.reverse()
        else:
            pts = plotLineHigh(x0, y0, x1, y1)
    return pts


# ==============================================================================
# 2. OCCUPANCY GRID
# ==============================================================================


class OccupancyGridMap:
    L_OCC, L_FREE, L_CLIP = 0.85, -0.40, 6.0

    def __init__(self, extent=(-16.0, 16.0, -16.0, 16.0), resolution=0.1, range_max=12.0):
        self.x_min, self.x_max, self.y_min, self.y_max = extent
        self.res = resolution
        self.range_max = range_max
        self.cols = int(round((self.x_max - self.x_min) / resolution))
        self.rows = int(round((self.y_max - self.y_min) / resolution))
        self.logodds = np.zeros((self.cols, self.rows))     # indexed [ix, iy]

    def world_to_grid(self, x, y):
        return (int(math.floor((x - self.x_min) / self.res)),
                int(math.floor((y - self.y_min) / self.res)))

    def update_ray(self, rx, ry, rth, length, angle):
        """Integrate one beam. length = nan means 'no return within range_max'."""
        hit = not np.isnan(length)
        z = length if hit else self.range_max
        beam = rth + angle
        i0, j0 = self.world_to_grid(rx, ry)
        i1, j1 = self.world_to_grid(rx + z * math.cos(beam), ry + z * math.sin(beam))
        cells = np.array(bresenham(i0, j0, i1, j1))
        ok = (cells[:, 0] >= 0) & (cells[:, 0] < self.cols) & (cells[:, 1] >= 0) & (cells[:, 1] < self.rows)
        end_inside = bool(ok[-1])
        cells = cells[ok]
        if len(cells) == 0:
            return
        n_free = len(cells) - 1 if (hit and end_inside) else len(cells)
        if n_free > 0:
            c = cells[:n_free]
            self.logodds[c[:, 0], c[:, 1]] = np.maximum(self.logodds[c[:, 0], c[:, 1]] + self.L_FREE, -self.L_CLIP)
        if hit and end_inside:
            i, j = cells[-1]
            self.logodds[i, j] = min(self.logodds[i, j] + self.L_OCC, self.L_CLIP)

    def update_scan(self, rx, ry, rth, lengths, angles):
        for le, an in zip(lengths, angles):
            self.update_ray(rx, ry, rth, le, an)

    def probability(self):
        """Occupancy probability as an image: row 0 = y max, column 0 = x min."""
        p = 1.0 / (1.0 + np.exp(-self.logodds))
        return np.flipud(p.T)


# ==============================================================================
# 3. TOP VIEW OF myworld_660610829.world (for the comparison picture)
# ==============================================================================


class Obj:
    """One scene primitive: kind in {'box','cyl','sphere'}, pose (x, y, z, yaw) and size."""

    def __init__(self, kind, x, y, z, yaw, size, rgba, collision):
        self.kind = kind
        self.x, self.y, self.z, self.yaw = x, y, z, yaw
        self.size = size            # box: (sx,sy,sz)  cyl: (r,len)  sphere: (r,)
        self.rgba = rgba
        self.collision = collision

    @property
    def z_range(self):
        h = self.size[2] if self.kind == 'box' else self.size[1] if self.kind == 'cyl' else 2 * self.size[0]
        return self.z - h / 2, self.z + h / 2

    def seen_by_lidar(self):
        lo, hi = self.z_range
        return self.collision and lo <= LIDAR_Z <= hi

    def corners(self):
        sx, sy = self.size[0] / 2, self.size[1] / 2
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pts = [(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)]
        return [(self.x + px * c - py * s, self.y + px * s + py * c) for px, py in pts]


def _floats(text):
    return [float(v) for v in text.split()]


def load_world(path=WORLD_FILE):
    """Read the Gazebo .world (SDF) file into a flat list of box/cylinder/sphere primitives."""
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
                kind, size = 'cyl', (float(cyl.find('radius').text), float(cyl.find('length').text))
            elif sph is not None:
                kind, size = 'sphere', (float(sph.find('radius').text),)
            else:
                continue
            rgba = (0.7, 0.7, 0.7, 1.0)
            if vis is not None:
                amb = vis.find('material/ambient')
                if amb is not None:
                    rgba = tuple(_floats(amb.text))
            objs.append(Obj(kind, x, y, z, yaw, size, rgba, col is not None))
    return objs


def draw_topview(ax, objs, extent):
    """Draw the scene from above. Objects the 2D LiDAR cannot see (too low/too high) are faded."""
    from matplotlib.patches import Circle, Polygon
    ax.set_facecolor('white')
    for o in sorted(objs, key=lambda o: o.z_range[1]):
        color, alpha = o.rgba[:3], 1.0 if o.seen_by_lidar() else 0.35
        if o.kind == 'box':
            ax.add_patch(Polygon(o.corners(), closed=True, fc=color, ec='k', lw=0.3, alpha=alpha))
        else:
            ax.add_patch(Circle((o.x, o.y), o.size[0], fc=color, ec='k', lw=0.3, alpha=alpha))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')


# ==============================================================================
# 4. DATA + PLOTS
# ==============================================================================


def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = data[None, :]
    n_ray = (data.shape[1] - 3) // 2
    X, Y, TH = data[:, 0], data[:, 1], data[:, 2]
    length = data[:, 3:3 + 2 * n_ray:2]
    angle = data[:, 4:4 + 2 * n_ray:2]
    return X, Y, TH, length, angle


def ground_truth_grid(ogm, objs):
    """Cells covered by obstacles the 2D LiDAR can see (same grid as the map) -- used only to
    print an accuracy check against the scene, not part of the saved output."""
    xs = ogm.x_min + (np.arange(ogm.cols) + 0.5) * ogm.res
    ys = ogm.y_min + (np.arange(ogm.rows) + 0.5) * ogm.res
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    gt = np.zeros_like(gx, dtype=bool)
    for o in objs:
        if not o.seen_by_lidar():
            continue
        if o.kind == 'box':
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            dx, dy = gx - o.x, gy - o.y
            lx, ly = dx * c + dy * s, -dx * s + dy * c
            gt |= (np.abs(lx) <= o.size[0] / 2) & (np.abs(ly) <= o.size[1] / 2)
        else:
            gt |= np.hypot(gx - o.x, gy - o.y) <= o.size[0]
    return np.flipud(gt.T)


def dilate(mask, r=1):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def save_comparison(png, ogm, prob, X, Y, objs):
    import matplotlib.pyplot as plt
    ext = (ogm.x_min, ogm.x_max, ogm.y_min, ogm.y_max)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.6))
    a1.imshow(prob, cmap='gray_r', vmin=0, vmax=1, origin='upper', extent=ext, interpolation='nearest')
    a1.plot(X, Y, 'c-', lw=0.6, alpha=0.7, label='robot path')
    a1.set_title(f'Occupancy Grid Map (student {STUDENT_ID}, {ogm.res} m/cell)\n'
                 'white = free, black = occupied, grey = unknown')
    a1.legend(loc='upper right', fontsize=8)
    draw_topview(a2, objs, ext)
    a2.set_title('Top View of myworld_660610829.world\n(faded objects are above/below the LiDAR plane)')
    for a in (a1, a2):
        a.set_xlabel('x [m]')
        a.set_ylabel('y [m]')
        a.set_xlim(ext[0], ext[1])
        a.set_ylim(ext[2], ext[3])
        a.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    return fig


def animate(ogm, X, Y, TH, L, A, every=3):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ext = (ogm.x_min, ogm.x_max, ogm.y_min, ogm.y_max)
    im = ax.imshow(ogm.probability(), cmap='gray_r', vmin=0, vmax=1, origin='upper', extent=ext)
    robot = Circle((X[0], Y[0]), 0.6, fill=False, ec='b', lw=1.5)
    ax.add_patch(robot)
    head, = ax.plot([], [], 'b-')
    plt.ion()
    plt.show()
    for k in range(len(X)):
        ogm.update_scan(X[k], Y[k], TH[k], L[k], A[k])
        if k % every == 0 or k == len(X) - 1:
            im.set_data(ogm.probability())
            robot.center = (X[k], Y[k])
            head.set_data([X[k], X[k] + 1.0 * math.cos(TH[k])], [Y[k], Y[k] + 1.0 * math.sin(TH[k])])
            ax.set_title(f'pose {k + 1}/{len(X)}')
            plt.pause(0.001)
        if not plt.fignum_exists(fig.number):
            break
    plt.ioff()
    plt.show()


def live(ogm, path, args, objs):
    """Follow data.csv while a robot is still driving in Gazebo and update the map live.

    Needs the ROS/Gazebo side of this project running and writing to `path` -- not something
    this standalone zip can do by itself, kept here so the same script also works that way
    when placed back next to the running simulation.
    """
    import time
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    IDLE_S, RAY_STEP = 8.0, 24
    ext = (ogm.x_min, ogm.x_max, ogm.y_min, ogm.y_max)
    fig, a1 = plt.subplots(figsize=(7.5, 7.5))
    im = a1.imshow(ogm.probability(), cmap='gray_r', vmin=0, vmax=1, origin='upper', extent=ext,
                   interpolation='nearest')
    path_line, = a1.plot([], [], 'c-', lw=0.8)
    rays = LineCollection([], colors='r', linewidths=0.5, alpha=0.6)
    a1.add_collection(rays)
    robot = Circle((0, 0), 0.6, fill=False, ec='b', lw=1.5)
    a1.add_patch(robot)
    head, = a1.plot([], [], 'b-', lw=1.5)
    a1.set_xlabel('x [m]')
    a1.set_ylabel('y [m]')
    a1.set_xlim(ext[0], ext[1])
    a1.set_ylim(ext[2], ext[3])
    a1.set_aspect('equal')
    fig.tight_layout()
    try:
        fig.canvas.manager.window.wm_geometry('750x790+40+40')
    except Exception:
        pass
    plt.ion()
    plt.show()

    offset, n, saved = 0, 0, False
    xs, ys = [], []
    last_new = time.time()
    a1.set_title('waiting for data.csv ...')
    while plt.fignum_exists(fig.number):
        rows = []
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size < offset:
                ogm.logodds[:] = 0
                offset, n, saved = 0, 0, False
                xs.clear()
                ys.clear()
            if size > offset:
                with open(path, 'rb') as f:
                    f.seek(offset)
                    chunk = f.read()
                end = chunk.rfind(b'\n') + 1
                offset += end
                for line in chunk[:end].decode().splitlines():
                    if line.strip():
                        rows.append([float(v) for v in line.split(',')])
        for row in rows:
            x, y, th = row[0], row[1], row[2]
            lengths, angles = np.array(row[3::2]), np.array(row[4::2])
            ogm.update_scan(x, y, th, lengths, angles)
            xs.append(x)
            ys.append(y)
            n += 1
        if rows:
            last_new = time.time()
            im.set_data(ogm.probability())
            path_line.set_data(xs, ys)
            robot.center = (x, y)
            head.set_data([x, x + math.cos(th)], [y, y + math.sin(th)])
            segs = []
            for le, an in zip(lengths[::RAY_STEP], angles[::RAY_STEP]):
                z = ogm.range_max if np.isnan(le) else le
                segs.append([(x, y), (x + z * math.cos(th + an), y + z * math.sin(th + an))])
            rays.set_segments(segs)
            a1.set_title(f'Occupancy Grid Map - pose {n}  x={x:.1f} y={y:.1f}  (white=free, black=occupied)')
        elif n > 0 and not saved and time.time() - last_new > IDLE_S:
            prob = ogm.probability()
            np.savetxt(args.out, prob, delimiter=',', fmt='%.4f')
            f2 = save_comparison(args.png, ogm, prob, xs, ys, objs)
            plt.close(f2)
            saved = True
            a1.set_title(f'Finished - {n} poses. Saved {os.path.basename(args.out)}')
            print(f'run finished, saved {args.out} and {args.png}')
        plt.pause(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data.csv'))
    ap.add_argument('--world', default=WORLD_FILE)
    ap.add_argument('--out', default=os.path.join(HERE, f'ocgm_{STUDENT_ID}.csv'))
    ap.add_argument('--png', default=os.path.join(HERE, f'ocgm_compare_{STUDENT_ID}.png'))
    ap.add_argument('--res', type=float, default=0.1, help='grid size [m]')
    ap.add_argument('--animate', action='store_true')
    ap.add_argument('--live', action='store_true',
                    help='follow data.csv live while the robot is driving (needs Gazebo/ROS running)')
    ap.add_argument('--no-show', action='store_true', help='only save files, do not open a window')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg' if args.no_show else 'TkAgg')

    if args.live:
        live(OccupancyGridMap(resolution=args.res), args.data, args, load_world(args.world))
        return

    X, Y, TH, L, A = load_data(args.data)
    print(f'{len(X)} poses, {L.shape[1]} rays each')
    ogm = OccupancyGridMap(resolution=args.res)

    if args.animate:
        animate(ogm, X, Y, TH, L, A)
    else:
        for k in range(len(X)):
            ogm.update_scan(X[k], Y[k], TH[k], L[k], A[k])
            if (k + 1) % 100 == 0:
                print(f'  mapped {k + 1}/{len(X)} poses')
    prob = ogm.probability()

    np.savetxt(args.out, prob, delimiter=',', fmt='%.4f')
    print(f'saved {args.out}  shape={prob.shape}  min={prob.min():.3f} max={prob.max():.3f}')

    objs = load_world(args.world)
    save_comparison(args.png, ogm, prob, X, Y, objs)
    print(f'saved {args.png}')

    gt = ground_truth_grid(ogm, objs)
    occ = prob > 0.7
    known = np.abs(prob - 0.5) > 0.05
    recall = (gt & dilate(occ)).sum() / max(gt.sum(), 1)
    precision = (occ & dilate(gt)).sum() / max(occ.sum(), 1)
    print(f'check vs. scene: obstacle cells found = {recall * 100:.1f}%, '
          f'occupied cells that are real obstacles = {precision * 100:.1f}%, '
          f'map explored = {known.sum() / known.size * 100:.1f}% of the mapped area')

    if not args.no_show:
        import matplotlib.pyplot as plt
        print('showing the map window (close it to exit)')
        plt.show()


if __name__ == '__main__':
    main()
