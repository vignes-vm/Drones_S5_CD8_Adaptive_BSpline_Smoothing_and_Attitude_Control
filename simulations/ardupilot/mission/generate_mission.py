"""
Generate an ArduPilot mission from a GA-style B-Spline-smoothed A* path.

Pipeline (Part 4 scenario, shared obstacle_env):
  1. 3D A* (astar_3d) -> raw waypoints
  2. Cubic B-Spline SMOOTHING via scipy.interpolate (chord-length parameterised,
     using scipy's well-tested make_interp_spline / BSpline — NOT a from-scratch
     Cox-de Boor, to avoid the knot-vector bugs hit in MATLAB)
  3. Arc-length resample every ~20 m (density follows curvature)
  4. Local X,Y,Z metres -> QGC WPL 110 mission (relative altitude)
  5. Save mission/bspline_mission.waypoints + mission/path_metadata.json

Outputs a summary table comparing A* baseline vs B-Spline smoothed metrics.
"""
import os
import json
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

from obstacle_env import START, GOAL, get_obstacles
from astar_3d import AStar3D

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Home reference GPS coordinate --------------------------------------------
# Placeholder: Chennai, India.  >>> CHANGE THIS to your actual test location. <<<
HOME_LAT = 13.0827
HOME_LON = 80.2707
_EARTH_M_PER_DEG = 111320.0   # metres per degree latitude (flat-earth approx)

WAYPOINT_SPACING_M = 20.0     # arc-length spacing between mission waypoints


def chord_length_bspline(waypoints, degree=3):
    """Cubic B-Spline through waypoints, chord-length parameterised.
    Returns a scipy BSpline object mapping u in [0,1] -> (x,y,z)."""
    wp = np.asarray(waypoints, float)
    # dedupe consecutive identical points (zero chord breaks parameterisation)
    keep = [0] + [i for i in range(1, len(wp))
                  if np.linalg.norm(wp[i] - wp[i - 1]) > 1e-6]
    wp = wp[keep]
    d = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    u /= u[-1]                                  # normalise to [0,1]
    # scipy make_interp_spline returns a BSpline; k=3 cubic, clamped by default
    spline = make_interp_spline(u, wp, k=min(degree, len(wp) - 1))
    return spline


def curve_length(spline, n=4000):
    u = np.linspace(0, 1, n)
    pts = spline(u)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))), u, pts


def max_curvature(spline, n=2000):
    u = np.linspace(0, 1, n)
    d1 = spline.derivative(1)(u)
    d2 = spline.derivative(2)(u)
    cross = np.cross(d1, d2)
    num = np.linalg.norm(cross, axis=1)
    den = np.linalg.norm(d1, axis=1) ** 3
    den[den < 1e-9] = 1e-9
    return float(np.max(num / den))


def arc_length_resample(spline, spacing=WAYPOINT_SPACING_M, n_dense=8000):
    """Sample the curve at points ~`spacing` metres apart along arc length."""
    u = np.linspace(0, 1, n_dense)
    pts = spline(u)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    s_targets = np.arange(0.0, total, spacing)
    s_targets = np.append(s_targets, total)     # always include the goal
    u_targets = np.interp(s_targets, s, u)
    return spline(u_targets)


def local_to_latlon(x, y):
    """Local metres (x=East, y=North) -> GPS lat/lon about HOME."""
    lat = HOME_LAT + (y / _EARTH_M_PER_DEG)
    lon = HOME_LON + (x / (_EARTH_M_PER_DEG * np.cos(np.radians(HOME_LAT))))
    return lat, lon


def write_wpl110(path_local, out_file):
    """Write QGC WPL 110 mission. Waypoints use frame 3 (GLOBAL_RELATIVE_ALT),
    command 16 (MAV_CMD_NAV_WAYPOINT). Item 0 is HOME."""
    lines = ["QGC WPL 110"]
    # Item 0: HOME (frame 0 GLOBAL, cmd 16, current=1)
    lines.append(f"0\t1\t0\t16\t0\t0\t0\t0\t{HOME_LAT:.7f}\t{HOME_LON:.7f}\t0.000000\t1")
    seq = 1
    for (x, y, z) in path_local:
        lat, lon = local_to_latlon(x, y)
        alt = max(0.0, float(z))                 # relative altitude (m)
        # seq curr frame cmd p1 p2 p3 p4 lat lon alt autocontinue
        lines.append(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat:.7f}\t{lon:.7f}\t{alt:.6f}\t1")
        seq += 1
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    return seq - 1                               # number of waypoints (excl. home)


def main():
    # 1. A* baseline
    planner = AStar3D()
    raw = planner.plan(START, GOAL)
    raw_len = AStar3D.path_length(raw)

    # 2. B-Spline smoothing (scipy)
    spline = chord_length_bspline(raw, degree=3)
    smooth_len, _, _ = curve_length(spline)
    kappa_max = max_curvature(spline)

    # 3. arc-length resample -> mission waypoints
    wp_local = arc_length_resample(spline, WAYPOINT_SPACING_M)

    # 4. write mission file
    wp_file = os.path.join(HERE, "bspline_mission.waypoints")
    n_wp = write_wpl110(wp_local, wp_file)

    # 5. metadata (visualization reads this; store enough to re-evaluate the curve)
    meta = {
        "home": {"lat": HOME_LAT, "lon": HOME_LON},
        "start": START.tolist(), "goal": GOAL.tolist(),
        "astar_path_length_m": raw_len,
        "bspline_path_length_m": smooth_len,
        "max_curvature": kappa_max,
        "num_waypoints": n_wp,
        "waypoint_spacing_m": WAYPOINT_SPACING_M,
        "obstacles": get_obstacles(),
        # curve reconstruction for tracking-error analysis
        "spline": {
            "degree": int(spline.k),
            "knots": spline.t.tolist(),
            "coeffs": spline.c.tolist(),
        },
        # sampled planned path (fallback for viz) — local metres
        "sampled_path_local": spline(np.linspace(0, 1, 600)).tolist(),
        "mission_waypoints_local": wp_local.tolist(),
    }
    meta_file = os.path.join(HERE, "path_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    # 6. summary table (Part 4 style)
    print("\n" + "=" * 56)
    print("  PATH PLANNING SUMMARY  (Part 4 dense-obstacle scenario)")
    print("=" * 56)
    print(f"  {'Metric':<26}{'A* baseline':>13}{'B-Spline':>15}")
    print("  " + "-" * 52)
    print(f"  {'Path length (m)':<26}{raw_len:>13.1f}{smooth_len:>15.1f}")
    print(f"  {'Max curvature (1/m)':<26}{'-':>13}{kappa_max:>15.4f}")
    print(f"  {'# waypoints':<26}{len(raw):>13d}{n_wp:>15d}")
    reduction = 100.0 * (raw_len - smooth_len) / raw_len
    print("  " + "-" * 52)
    print(f"  Length reduction vs A*: {reduction:+.1f}%")
    print("=" * 56)
    print(f"\n  mission file : {wp_file}")
    print(f"  metadata     : {meta_file}")
    print(f"  home (PLACEHOLDER, change me): lat={HOME_LAT}, lon={HOME_LON}\n")


if __name__ == "__main__":
    main()
