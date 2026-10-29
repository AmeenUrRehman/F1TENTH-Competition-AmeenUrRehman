#!/usr/bin/env python3
r"""Turn recorded runs into README media (GIF + PNG screenshots).

Inputs are real data only:
  * an .npz from tools/record_run.py (official simulator, ROS topics), or
  * a trajectory CSV from tools/harness.py --trace (headless test).
The track background is the competition's own map image.

Run with the system Python (matplotlib), e.g.:
  python3 tools/render_media.py official run.npz --map levine --out docs/media --name stage1
  python3 tools/render_media.py trace trace.csv --map levine_blocked --out docs/media \\
      --name stage2 --title "Stage 2 - levine_blocked (headless harness)"
"""

import argparse
import math
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import numpy as np  # noqa: E402

MAPS = (pathlib.Path.home() / 'CompetionF1' / 'CIE-KFUPM-F1tenth-Autonomous-Racing-Competition'
        / 'starter_ws' / 'src' / 'f1tenth_gym_ros' / 'maps')

# Chart tokens (light surface).
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
WALL = '#8a8984'
LIDAR = '#eb6834'
BLUE = '#2a78d6'
SPEED_CMAP = LinearSegmentedColormap.from_list(
    'speed', ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b'])

LIDAR_X = 0.275          # m ahead of the rear axle (odom frame)
CAR = np.array([[-0.12, -0.155], [0.46, -0.155], [0.46, 0.155], [-0.12, 0.155]])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
    'axes.edgecolor': INK_2, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
    'ytick.color': INK_2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE})


def load_map(name):
    import yaml
    from PIL import Image
    meta = yaml.safe_load((MAPS / f'{name}.yaml').read_text())
    img = np.asarray(Image.open(MAPS / meta['image']).convert('L'), dtype=np.float32) / 255.0
    res = float(meta['resolution'])
    ox, oy = meta['origin'][:2]
    h, w = img.shape
    # Walls only (occupied = dark pixels), drawn as an RGBA overlay.
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    occ = img < 0.5
    rgba[occ] = matplotlib.colors.to_rgba(WALL)
    return rgba, (ox, ox + w * res, oy, oy + h * res)


def load_run(kind, path):
    if kind == 'official':
        d = np.load(path, allow_pickle=False)
        cols = list(d['columns'])
        rows = d['rows']
        col = {c: rows[:, i] for i, c in enumerate(cols)}
        a0, da = d['scan_meta']
        return dict(t=col['t'], x=col['x'], y=col['y'], yaw=col['yaw'], v=col['v'],
                    laps=col['laps'].astype(int), coll=col['collisions'].astype(int),
                    cur=col['cur_lap'], scans=d['scans'], a0=float(a0), da=float(da),
                    history=list(d['lap_history']))
    arr = np.loadtxt(path, delimiter=',', comments='#')
    t, x, y, v = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    yaw = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    return dict(t=t, x=x, y=y, yaw=yaw, v=v, laps=None, coll=None, cur=None,
                scans=None, history=[])


def car_poly(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = CAR @ np.array([[c, s], [-s, c]])
    return pts + [x, y]


def speed_lines(ax, run, vmax, lw=2.0):
    pts = np.column_stack([run['x'], run['y']]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=SPEED_CMAP, norm=plt.Normalize(0, vmax),
                        linewidths=lw, capstyle='round')
    lc.set_array(run['v'][:-1])
    ax.add_collection(lc)
    return lc


def track_axes(ax, walls, extent, run, pad=2.5):
    ax.imshow(walls, extent=extent, origin='upper', interpolation='nearest', zorder=1)
    ax.set_xlim(run['x'].min() - pad, run['x'].max() + pad)
    ax.set_ylim(run['y'].min() - pad, run['y'].max() + pad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def screenshot_track(run, walls, extent, out, title, subtitle):
    vmax = max(1.0, float(np.ceil(run['v'].max())))
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
    track_axes(ax, walls, extent, run)
    lc = speed_lines(ax, run, vmax)
    ax.plot(run['x'][0], run['y'][0], marker='o', ms=9, color=INK, mec=SURFACE, mew=2, zorder=5)
    ax.annotate('start', (run['x'][0], run['y'][0]), xytext=(8, -14),
                textcoords='offset points', color=INK_2)
    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('speed (m/s)')
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.02, ha='left', fontsize=13, fontweight='bold')
    ax.set_title(subtitle, loc='left', color=INK_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def screenshot_laps(history, collisions, out, title):
    if not history:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=130)
    labels = [f'lap {i}' for i in range(1, len(history) + 1)]
    colors = [BLUE] * len(history)
    bars = ax.bar(labels, history, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, h in zip(bars, history):
        ax.annotate(f'{h:.2f} s', (b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                    textcoords='offset points', ha='center', color=INK)
    timed = history[:3]
    ax.set_ylabel('lap time (s)')
    ax.set_ylim(0, max(history) * 1.22)
    ax.grid(axis='y', color='#e4e3df', linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    note = f'3 timed laps: {sum(timed):.2f} s   |   collisions: {collisions}' if len(timed) == 3 \
        else f'collisions: {collisions}'
    ax.set_title(note, loc='left', color=INK_2, fontsize=10)
    fig.suptitle(title, x=0.02, ha='left', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def screenshot_speed(run, out, title):
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=130)
    ax.plot(run['t'], run['v'], color=BLUE, linewidth=2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (m/s)')
    ax.grid(color='#e4e3df', linewidth=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if run['laps'] is not None:
        for k in range(1, run['laps'].max() + 1):
            idx = np.flatnonzero(run['laps'] >= k)
            if len(idx):
                ax.axvline(run['t'][idx[0]], color=INK_2, linewidth=0.8, linestyle='--')
                ax.annotate(f'lap {k} done', (run['t'][idx[0]], ax.get_ylim()[1]),
                            xytext=(3, -12), textcoords='offset points', color=INK_2, fontsize=8)
    fig.suptitle(title, x=0.02, ha='left', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def lidar_world(run, i):
    r = run['scans'][i]
    ang = run['a0'] + run['da'] * np.arange(len(r))
    ok = (r > 0.05) & (r < 9.5)
    x, y, yaw = run['x'][i], run['y'][i], run['yaw'][i]
    lx, ly = x + LIDAR_X * math.cos(yaw), y + LIDAR_X * math.sin(yaw)
    return lx + r[ok] * np.cos(yaw + ang[ok]), ly + r[ok] * np.sin(yaw + ang[ok])


def screenshot_lidar(run, walls, extent, out, title):
    # The frame where the car turns hardest: shows what the LiDAR sees in a corner.
    turn = np.abs(np.gradient(np.unwrap(run['yaw'])))
    i = int(np.argmax(turn[len(turn) // 4:])) + len(turn) // 4
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    ax.imshow(walls, extent=extent, origin='upper', interpolation='nearest', zorder=1)
    px, py = lidar_world(run, i)
    ax.scatter(px, py, s=6, color=LIDAR, zorder=3, label='LiDAR hits')
    j0 = max(0, i - 60)
    ax.plot(run['x'][j0:i + 1], run['y'][j0:i + 1], color=BLUE, linewidth=2, zorder=2,
            label='path driven')
    ax.add_patch(Polygon(car_poly(run['x'][i], run['y'][i], run['yaw'][i]), closed=True,
                         facecolor=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4))
    ax.set_xlim(run['x'][i] - 5, run['x'][i] + 5)
    ax.set_ylim(run['y'][i] - 5, run['y'][i] + 5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower left', frameon=False)
    ax.set_title(f'{title}, speed {run["v"][i]:.1f} m/s', loc='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_gif(run, walls, extent, out, title, speedup=2.0, fps=12, width_px=900):
    vmax = max(1.0, float(np.ceil(run['v'].max())))
    fig = plt.figure(figsize=(width_px / 100, width_px / 100 * 0.5), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 0.66, 0.88])
    track_axes(ax, walls, extent, run, pad=1.5)
    zoom = fig.add_axes([0.67, 0.02, 0.32, 0.64])
    zoom.imshow(walls, extent=extent, origin='upper', interpolation='nearest', zorder=1)
    zoom.set_xticks([])
    zoom.set_yticks([])
    zoom.set_aspect('equal')
    for sp in zoom.spines.values():
        sp.set_color(INK_2)
    fig.text(0.01, 0.94, title, fontsize=13, fontweight='bold', va='center')
    hud = fig.text(0.68, 0.84, '', fontsize=10, va='top', family='DejaVu Sans Mono')

    trail = LineCollection([], cmap=SPEED_CMAP, norm=plt.Normalize(0, vmax), linewidths=2.2)
    ax.add_collection(trail)
    car = Polygon(CAR, closed=True, facecolor=INK, edgecolor=SURFACE, linewidth=1, zorder=5)
    ax.add_patch(car)
    zcar = Polygon(CAR, closed=True, facecolor=INK, edgecolor=SURFACE, linewidth=1, zorder=5)
    zoom.add_patch(zcar)
    hits = zoom.scatter([], [], s=5, color=LIDAR, zorder=3)
    pts = np.column_stack([run['x'], run['y']]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    t = run['t']
    frame_times = np.arange(t[0], t[-1], speedup / fps)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(out), dpi=100):
        for ft in frame_times:
            i = int(np.searchsorted(t, ft))
            i = min(i, len(t) - 1)
            trail.set_segments(segs[:max(i, 1)])
            trail.set_array(run['v'][:max(i, 1)])
            poly = car_poly(run['x'][i], run['y'][i], run['yaw'][i])
            car.set_xy(poly)
            zcar.set_xy(poly)
            zoom.set_xlim(run['x'][i] - 4, run['x'][i] + 4)
            zoom.set_ylim(run['y'][i] - 4, run['y'][i] + 4)
            if run['scans'] is not None:
                hx, hy = lidar_world(run, i)
                hits.set_offsets(np.column_stack([hx, hy]))
            lines = [f'speed      {run["v"][i]:5.2f} m/s']
            if run['laps'] is not None:
                laps = int(run['laps'][i])
                lines.append(f'lap time   {run["cur"][i]:5.2f} s')
                lines.append(f'laps done  {laps} / 3')
                lines.append(f'collisions {int(run["coll"][i])}')
            else:
                lines.append(f'time       {t[i] - t[0]:5.1f} s')
            lines.append(f'playback   x{speedup:g}')
            hud.set_text('\n'.join(lines))
            writer.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['official', 'trace'])
    ap.add_argument('path')
    ap.add_argument('--map', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', required=True, help='file name prefix')
    ap.add_argument('--title', default='')
    ap.add_argument('--gif', action='store_true', help='also render an animated GIF')
    ap.add_argument('--speedup', type=float, default=2.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = load_run(args.kind, args.path)
    walls, extent = load_map(args.map)
    title = args.title or f'RobotForge - {args.name}'
    src = 'official simulator + race monitor' if args.kind == 'official' else 'headless harness'

    screenshot_track(run, walls, extent, out / f'{args.name}_track.png', title,
                     f'path driven, coloured by speed  ({src}, map {args.map})')
    screenshot_speed(run, out / f'{args.name}_speed.png', f'{title} - speed over time')
    if args.kind == 'official':
        coll = int(run['coll'][-1])
        screenshot_laps(run['history'], coll, out / f'{args.name}_laps.png',
                        f'{title} - official race monitor')
        screenshot_lidar(run, walls, extent, out / f'{args.name}_lidar.png',
                         f'{title}\nwhat the LiDAR sees in a corner')
    if args.gif:
        make_gif(run, walls, extent, out / f'{args.name}.gif', title, speedup=args.speedup)
    for f in sorted(out.glob(f'{args.name}*')):
        print(f, f'{f.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
