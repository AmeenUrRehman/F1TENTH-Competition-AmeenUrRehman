#!/usr/bin/env python3
"""Build a racing line (x, y, speed) from an official centerline CSV.

Only for the stages whose map and route are officially released (1 and 2).

1. Lateral offsets from the centerline are optimised to minimise curvature,
   bounded by the track widths in the CSV minus the car half-width and a
   safety margin (and checked against the map image's walls).
2. A speed profile is computed from the lateral-acceleration limit, then
   forward/backward passes apply acceleration and braking limits.

Usage:
  make_raceline.py CENTERLINE.csv MAP.yaml OUT.csv [--margin 0.25] [--a-lat 7]
"""

import argparse
import math
import pathlib

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize


def load_centerline(path):
    c = np.loadtxt(path, delimiter=",", comments="#")
    return c[:, :2], c[:, 2], c[:, 3]   # xy, width right, width left


def wall_distance(map_yaml):
    """Distance (m) to the nearest wall at every pixel of the map."""
    spec = yaml.safe_load(open(map_yaml))
    img = Image.open(pathlib.Path(map_yaml).parent / spec["image"]).convert("L")
    grid = np.array(img, dtype=np.float64) / 255.0
    occupied = (1.0 - grid) > spec["occupied_thresh"]
    if spec.get("negate", 0):
        occupied = ~occupied
    dist = distance_transform_edt(~occupied) * spec["resolution"]
    return dist, spec["resolution"], spec["origin"], grid.shape[0]


def lookup(dist, res, origin, height, xy):
    col = ((xy[:, 0] - origin[0]) / res).astype(int)
    row = (height - 1 - (xy[:, 1] - origin[1]) / res).astype(int)
    return dist[np.clip(row, 0, height - 1), np.clip(col, 0, dist.shape[1] - 1)]


def normals(xy):
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], axis=1)   # points to the left


def curvature(xy):
    a, b, c = np.roll(xy, 1, axis=0), xy, np.roll(xy, -1, axis=0)
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = (b - a)[:, 0] * (c - b)[:, 1] - (b - a)[:, 1] * (c - b)[:, 0]
    return 2.0 * cross / np.maximum(ab * bc * ca, 1e-9)


def optimise_offsets(xy, n, lo, hi, smooth_weight=0.02):
    def path(d):
        return xy + n * d[:, None]

    def cost(d):
        p = path(d)
        second = np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)
        dd = np.roll(d, -1) - d
        return float((second ** 2).sum() * 1e3 + smooth_weight * (dd ** 2).sum())

    res = minimize(cost, np.zeros(len(xy)), method="L-BFGS-B",
                   bounds=list(zip(lo, hi)), options={"maxiter": 3000})
    return path(res.x)


def resample(xy, step):
    closed = np.vstack([xy, xy[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    total = s[-1]
    q = np.arange(0.0, total, step)
    return np.stack([np.interp(q, s, closed[:, 0]), np.interp(q, s, closed[:, 1])], axis=1)


def speed_profile(xy, a_lat, a_acc, a_brake, v_max):
    k = np.abs(curvature(xy))
    # Light smoothing so single noisy points do not create speed spikes.
    k = np.convolve(np.concatenate([k[-3:], k, k[:3]]), np.ones(7) / 7, "valid")
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(k, 1e-6)))
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    for _ in range(2):   # two laps so the loop closes consistently
        for i in range(len(v)):          # acceleration limit
            j = (i + 1) % len(v)
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * a_acc * ds[i]))
        for i in range(len(v) - 1, -1, -1):   # braking limit
            j = (i + 1) % len(v)
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * a_brake * ds[i]))
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("centerline")
    ap.add_argument("map_yaml")
    ap.add_argument("out")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="clearance from walls beyond half the car width (m)")
    ap.add_argument("--a-lat", type=float, default=7.0, help="lateral accel limit")
    ap.add_argument("--a-acc", type=float, default=6.0)
    ap.add_argument("--a-brake", type=float, default=6.0)
    ap.add_argument("--v-max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.1, help="output spacing (m)")
    args = ap.parse_args()

    xy, w_right, w_left = load_centerline(args.centerline)
    half = 0.31 / 2 + args.margin
    n = normals(xy)
    line = optimise_offsets(xy, n, -(w_right - half), (w_left - half))
    line = resample(line, args.step)

    dist, res, origin, height = wall_distance(args.map_yaml)
    clearance = lookup(dist, res, origin, height, line)
    v = speed_profile(line, args.a_lat, args.a_acc, args.a_brake, args.v_max)

    length = float(np.linalg.norm(np.diff(np.vstack([line, line[:1]]), axis=0), axis=1).sum())
    lap = float((np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1) / v).sum())
    with open(args.out, "w") as f:
        f.write(f"# x_m, y_m, v_mps  from {pathlib.Path(args.centerline).name}, "
                f"margin {args.margin} m, a_lat {args.a_lat}, length {length:.2f} m, "
                f"ideal lap {lap:.2f} s\n")
        np.savetxt(f, np.column_stack([line, v]), fmt="%.4f", delimiter=", ")
    print(f"{args.out}: {len(line)} pts, length {length:.2f} m, ideal lap {lap:.2f} s, "
          f"min wall clearance {clearance.min():.2f} m, v {v.min():.2f}-{v.max():.2f} m/s")
    if clearance.min() < 0.31 / 2 + 0.05:
        print("WARNING: line passes closer than 5 cm beyond the car half-width to a wall")


if __name__ == "__main__":
    main()
