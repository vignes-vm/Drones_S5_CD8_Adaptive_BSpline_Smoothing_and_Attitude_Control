"""
Path utilities for the Gazebo mission: collision-checked smoothing and arc-length
reference timing (position + speed profile that stops at the goal).
"""
import numpy as np


def count_obstacle_hits(environment_3d, points, clearance=0.0):
    """Number of sample points inside any obstacle inflated by `clearance` metres."""
    hits = 0
    for p in np.asarray(points, dtype=float):
        for obs in environment_3d.obstacles:
            delta = p - obs.position
            if obs.shape == 'cylinder':
                radius, height = obs.dimensions
                inside = (np.hypot(delta[0], delta[1]) <= radius + clearance and
                          abs(delta[2]) <= height / 2.0 + clearance)
            else:
                inside = bool(np.all(np.abs(delta) <= obs.dimensions / 2.0 + clearance))
            if inside:
                hits += 1
                break
    return hits


def plan_safe_path(environment_3d, astar, smoother_factory, min_altitude=15.0,
                   clearance=2.0, min_spline_altitude=3.0, prune_distances=(100, 50, 0),
                   n_samples=3000, verbose=True):
    """
    A* -> altitude floor -> adaptive B-spline, retrying with denser waypoints until the
    smoothed spline is collision-free (with `clearance`) and stays above ground.

    The shared smoother is a least-squares fit that is not collision-aware, so a sparse
    waypoint set can produce a spline that cuts through obstacles.
    Returns (waypoints, spline, report) where report has 'hits' and 'min_z'.
    """
    raw = np.array(astar.plan(environment_3d.start, environment_3d.goal))
    best = None
    for md in prune_distances:
        wp = np.array(astar.smooth_waypoints(raw, min_distance=md)) if md else raw.copy()
        wp[:, 2] = np.maximum(wp[:, 2], min_altitude)
        spline = smoother_factory().smooth(wp, verbose=False)
        pts = spline.sample_path(n_samples)
        hits = count_obstacle_hits(environment_3d, pts, clearance)
        min_z = float(pts[:, 2].min())
        if verbose:
            print(f"  Waypoint spacing {md or 'raw'} m -> {len(wp)} waypoints | "
                  f"obstacle hits={hits} | min altitude={min_z:.1f} m", flush=True)
        report = {'hits': hits, 'min_z': min_z, 'n_waypoints': len(wp)}
        if best is None or (hits, -min_z) < (best[2]['hits'], -best[2]['min_z']):
            best = (wp, spline, report)
        if hits == 0 and min_z >= min_spline_altitude:
            return wp, spline, report
    if verbose:
        print("  [Warning] No collision-free spline found; using the best candidate.", flush=True)
    return best


class ArcLengthProfile:
    """
    Maps travelled distance s to spline parameter u and a speed reference that cruises,
    then decelerates so the vehicle stops at the end of the path.
    """

    def __init__(self, spline, cruise_speed, decel=2.0, n_samples=4000):
        self.u_grid = np.linspace(0.0, 0.999, n_samples)
        pts = np.array([spline.evaluate(u) for u in self.u_grid])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.s_grid = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s_grid[-1])
        self.cruise_speed = cruise_speed
        self.decel = decel

    def u_at(self, s):
        return float(np.interp(min(s, self.length), self.s_grid, self.u_grid))

    def speed_at(self, s):
        remaining = max(0.0, self.length - s)
        return min(self.cruise_speed, float(np.sqrt(2.0 * self.decel * remaining)))
