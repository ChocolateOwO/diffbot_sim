"""Planned path (RRT*) against what the robot really drives, and the map updating while it drives.

  python plot_result_660610829.py            after the run: plan + trajectory.csv -> result.png
  python plot_result_660610829.py --live     while the robot drives in Gazebo (rrt_nav.launch.py opens it):
        left  : the occupancy grid map is BUILT / UPDATED live from the LiDAR scans (drive_scan.csv,
                written by lidar_recorder while the robot moves), with the RRT* path, the robot and
                its laser beams on top
        right : the OGM that was used for planning, with the same RRT* path and the driven path
      when the robot has arrived the updated map is saved as ocgm_updated_660610829.csv + result.png
"""
import argparse
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- occupancy grid (same method as myworld_ocgm_660610829.py: Bresenham + log-odds) ---------------
def plotLineLow(x0, y0, x1, y1):
    pts, dx, dy, yi = [], x1 - x0, y1 - y0, 1
    if dy < 0:
        yi, dy = -1, -dy
    D, y = 2 * dy - dx, y0
    for x in range(x0, x1 + 1):
        pts.append((x, y))
        if D > 0:
            y += yi
            D += 2 * (dy - dx)
        else:
            D += 2 * dy
    return pts


def plotLineHigh(x0, y0, x1, y1):
    pts, dx, dy, xi = [], x1 - x0, y1 - y0, 1
    if dx < 0:
        xi, dx = -1, -dx
    D, x = 2 * dx - dy, x0
    for y in range(y0, y1 + 1):
        pts.append((x, y))
        if D > 0:
            x += xi
            D += 2 * (dx - dy)
        else:
            D += 2 * dx
    return pts


def bresenham(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            p = plotLineLow(x1, y1, x0, y0)
            p.reverse()
            return p
        return plotLineLow(x0, y0, x1, y1)
    if y0 > y1:
        p = plotLineHigh(x1, y1, x0, y0)
        p.reverse()
        return p
    return plotLineHigh(x0, y0, x1, y1)


class LiveGrid:
    L_OCC, L_FREE, L_CLIP = 0.85, -0.40, 6.0

    def __init__(self, extent, res, range_max=12.0):
        self.x_min, self.x_max, self.y_min, self.y_max = extent
        self.res, self.range_max = res, range_max
        self.cols = int(round((self.x_max - self.x_min) / res))
        self.rows = int(round((self.y_max - self.y_min) / res))
        self.logodds = np.zeros((self.cols, self.rows))      # [ix, iy]

    def cell(self, x, y):
        return int(math.floor((x - self.x_min) / self.res)), int(math.floor((y - self.y_min) / self.res))

    def update_ray(self, rx, ry, rth, length, angle):
        hit = not np.isnan(length)
        z = length if hit else self.range_max
        b = rth + angle
        i0, j0 = self.cell(rx, ry)
        i1, j1 = self.cell(rx + z * math.cos(b), ry + z * math.sin(b))
        c = np.array(bresenham(i0, j0, i1, j1))
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.cols) & (c[:, 1] >= 0) & (c[:, 1] < self.rows)
        end_in = bool(ok[-1])
        c = c[ok]
        if len(c) == 0:
            return
        n_free = len(c) - 1 if (hit and end_in) else len(c)
        if n_free > 0:
            f = c[:n_free]
            self.logodds[f[:, 0], f[:, 1]] = np.maximum(self.logodds[f[:, 0], f[:, 1]] + self.L_FREE, -self.L_CLIP)
        if hit and end_in:
            i, j = c[-1]
            self.logodds[i, j] = min(self.logodds[i, j] + self.L_OCC, self.L_CLIP)

    def update_scan(self, x, y, th, lengths, angles):
        for le, an in zip(lengths, angles):
            self.update_ray(x, y, th, le, an)

    def probability(self):
        return np.flipud((1.0 / (1.0 + np.exp(-self.logodds))).T)     # row 0 = y max


# ---- drawing -------------------------------------------------------------------------------------
def arrow(ax, pose, col):
    ax.plot(pose[0], pose[1], 'o', color=col, ms=9)
    ax.arrow(pose[0], pose[1], 1.4 * math.cos(pose[2]), 1.4 * math.sin(pose[2]), color=col,
             width=0.12, head_width=0.5, length_includes_head=True, zorder=6)


def result_title(res_file):
    r = json.load(open(res_file))
    word = 'SUCCESS - goal reached' if r['success'] else 'FAILED'
    return (f"{word}\nposition error {r['position_error_m']:.2f} m | heading error "
            f"{r['heading_error_deg']:.1f} deg\nmax {r['max_distance_from_plan_m']:.2f} m off the plan | "
            f"{r['time_s']:.0f} s")


def posthoc(args, plan, prob, res_file):
    import matplotlib.pyplot as plt
    ext, p = plan['extent'], np.array(plan['path'])
    rows = list(csv.DictReader(open(args.traj)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={'width_ratios': [1.15, 1]})
    a1.imshow(prob, cmap='gray_r', vmin=0, vmax=1, origin='upper', extent=(ext[0], ext[1], ext[2], ext[3]),
              interpolation='nearest')
    a1.plot(p[:, 0], p[:, 1], 'b-', lw=2.5, label='RRT* path (planned)')
    a1.plot(p[:, 0], p[:, 1], 'b.', ms=6)
    a1.plot([float(r['x']) for r in rows], [float(r['y']) for r in rows], 'r-', lw=1.6, label='driven in Gazebo')
    arrow(a1, plan['start'], 'g')
    arrow(a1, plan['goal'], 'm')
    a1.set_aspect('equal')
    a1.legend(loc='upper right', fontsize=8)
    if os.path.exists(res_file):
        a1.set_title(result_title(res_file), fontsize=10)
    t = [float(r['t']) for r in rows]
    a2.plot(t, [float(r['v_cmd']) for r in rows], 'k-', lw=0.8, label='forward speed [m/s]')
    a2.plot(t, [float(r['wheel_left']) for r in rows], 'c-', lw=0.8, label='left wheel [rad/s]')
    a2.plot(t, [float(r['wheel_right']) for r in rows], 'm-', lw=0.8, label='right wheel [rad/s]')
    a2.set_xlabel('time [s]')
    a2.grid(True, alpha=0.3)
    a2.legend(fontsize=8)
    a2.set_title('commands sent to the wheels')
    fig.tight_layout()
    fig.savefig(args.png, dpi=130)
    print('saved', args.png)
    if not args.no_show:
        plt.show()


def live(args, plan, prob, res_file):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    ext, p = plan['extent'], np.array(plan['path'])
    grid = LiveGrid(ext, plan['resolution'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.4))
    im = a1.imshow(grid.probability(), cmap='gray_r', vmin=0, vmax=1, origin='upper',
                   extent=(ext[0], ext[1], ext[2], ext[3]), interpolation='nearest')
    a2.imshow(prob, cmap='gray_r', vmin=0, vmax=1, origin='upper', extent=(ext[0], ext[1], ext[2], ext[3]),
              interpolation='nearest')
    for ax in (a1, a2):
        ax.plot(p[:, 0], p[:, 1], 'b-', lw=2.2)
        ax.plot(p[:, 0], p[:, 1], 'b.', ms=5)
        arrow(ax, plan['start'], 'g')
        arrow(ax, plan['goal'], 'm')
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
    a1.set_ylabel('y [m]')
    (path1,) = a1.plot([], [], 'r-', lw=1.4)
    (path2,) = a2.plot([], [], 'r-', lw=1.4)
    rays = LineCollection([], colors='r', linewidths=0.5, alpha=0.55)
    a1.add_collection(rays)
    robot = Circle((plan['start'][0], plan['start'][1]), 0.6, fill=False, ec='orange', lw=2, zorder=7)
    a1.add_patch(robot)
    (head,) = a1.plot([], [], '-', color='orange', lw=2, zorder=7)
    a2.set_title('OGM used for planning\n(blue = RRT* path, red = driven)', fontsize=10)
    a1.set_title('map updating from the LiDAR\nwaiting for the robot ...', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    try:                      # WSLg sometimes opens new Tk windows off-screen
        fig.canvas.manager.window.wm_geometry('1050x560+40+40')
    except Exception:
        pass
    plt.ion()
    plt.show()

    offset, n, saved = 0, 0, False
    xs, ys = [], []
    gx, gy, _ = plan['goal']
    while plt.fignum_exists(fig.number):
        new = []
        if os.path.exists(args.scan):
            size = os.path.getsize(args.scan)
            if size < offset:                      # a new run restarted the file
                grid.logodds[:] = 0
                offset, n, saved = 0, 0, False
                xs.clear()
                ys.clear()
            if size > offset:
                with open(args.scan, 'rb') as f:
                    f.seek(offset)
                    chunk = f.read()
                end = chunk.rfind(b'\n') + 1       # complete lines only
                offset += end
                new = [[float(v) for v in ln.split(',')] for ln in chunk[:end].decode().splitlines() if ln.strip()]
        for row in new:
            x, y, th = row[0], row[1], row[2]
            lengths, angles = np.array(row[3::2]), np.array(row[4::2])
            grid.update_scan(x, y, th, lengths, angles)
            xs.append(x)
            ys.append(y)
            n += 1
        if new:
            im.set_data(grid.probability())
            path1.set_data(xs, ys)
            path2.set_data(xs, ys)
            robot.center = (x, y)
            head.set_data([x, x + math.cos(th)], [y, y + math.sin(th)])
            segs = []
            for le, an in zip(lengths[::24], angles[::24]):
                z = grid.range_max if np.isnan(le) else le
                segs.append([(x, y), (x + z * math.cos(th + an), y + z * math.sin(th + an))])
            rays.set_segments(segs)
            a1.set_title(f'map updating from the LiDAR   scan {n}\n'
                         f'robot ({x:.1f}, {y:.1f})  |  {math.hypot(gx - x, gy - y):.1f} m to the goal', fontsize=10)
        if (not saved and n and os.path.exists(res_file) and not new
                and os.path.getmtime(res_file) >= os.path.getmtime(args.scan) - 5):
            np.savetxt(args.updated_map, grid.probability(), delimiter=',', fmt='%.4f')
            a1.set_title(result_title(res_file), fontsize=10)
            fig.savefig(args.png, dpi=130)
            saved = True
            print('finished, saved', args.png, 'and', args.updated_map)
        plt.pause(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default=os.path.join(HERE, 'plan.json'))
    ap.add_argument('--traj', default=os.path.join(HERE, 'trajectory.csv'))
    ap.add_argument('--scan', default=os.path.join(HERE, 'drive_scan.csv'),
                    help='LiDAR rows written by lidar_recorder while driving (--live)')
    ap.add_argument('--map', default=os.path.join(HERE, 'ocgm_660610829.csv'))
    ap.add_argument('--png', default=os.path.join(HERE, 'result.png'))
    ap.add_argument('--updated-map', default=os.path.join(HERE, 'ocgm_updated_660610829.csv'))
    ap.add_argument('--live', action='store_true', help='update the map live while the robot drives')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg' if args.no_show else 'TkAgg')

    plan = json.load(open(args.plan))
    prob = np.loadtxt(args.map, delimiter=',')
    res_file = args.traj.rsplit('.', 1)[0] + '_result.json'
    (live if args.live else posthoc)(args, plan, prob, res_file)


if __name__ == '__main__':
    main()
