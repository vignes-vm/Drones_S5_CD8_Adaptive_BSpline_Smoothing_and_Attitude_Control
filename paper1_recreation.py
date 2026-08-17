"""
Recreation of Table 2 from Paper 1:
"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"

Compares 4 path-smoothing methods on the same A*-generated baseline
polyline, averaged over 5 trials with different obstacle-placement seeds.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.bspline import BSplinePath
from path_planning.knot_ga import KnotGA

SEEDS = [42, 123, 456, 789, 999]


def discrete_curvature(waypoints):
    """
    Menger (circumradius) curvature at each interior vertex of a polyline —
    the standard way to assign a finite curvature to a piecewise-linear
    path, since the continuous C'xC''/|C'|^3 formula is undefined at
    polyline vertices (curvature there is a Dirac delta).
    """
    pts = np.asarray(waypoints, dtype=float)
    if len(pts) < 3:
        return np.array([0.0])

    curvatures = []
    for i in range(1, len(pts) - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        c = pts[i + 1] - pts[i - 1]
        len_a, len_b, len_c = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        cross_norm = np.linalg.norm(np.cross(a, b))
        denom = len_a * len_b * len_c
        curvatures.append(0.0 if denom < 1e-10 else 2 * cross_norm / denom)
    return np.array(curvatures)


def chord_length_knot_vector(waypoints, degree=3):
    """
    Classical chord-length-parameterized knot vector via the standard
    knot-averaging technique (Piegl & Tiller, "The NURBS Book", Eq. 9.8) —
    distinct from a uniform knot vector because interior knots are placed
    according to accumulated segment length, not evenly in u.
    """
    pts = np.asarray(waypoints, dtype=float)
    n = len(pts) - 1
    p = degree

    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total = cumulative[-1]
    u = cumulative / total if total > 1e-10 else np.linspace(0, 1, n + 1)

    knots = np.zeros(n + p + 2)
    knots[-(p + 1):] = 1.0
    for j in range(1, n - p + 1):
        knots[j + p] = np.sum(u[j:j + p]) / p
    return knots


def compute_metrics(waypoints, is_spline, start, goal):
    dist = np.linalg.norm(np.array(goal) - np.array(start))
    if is_spline:
        length = waypoints.path_length()
        kappa  = waypoints.max_curvature()
    else:
        pts = np.array(waypoints)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        kappa  = float(np.max(discrete_curvature(pts)))
    return length, kappa, length / dist


def run_trial(seed):
    print(f"\n--- Trial (seed={seed}) ---")
    env = Environment3D()
    env.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)

    astar = AStarPlanner3D(env, grid_resolution=50)
    baseline_wp = np.array(astar.plan(env.start, env.goal))
    print(f"  A* baseline: {len(baseline_wp)} waypoints")

    trial = {}

    # Method 1 — Baseline Polyline
    L, k, SR = compute_metrics(baseline_wp, False, env.start, env.goal)
    trial['baseline'] = {'length': L, 'kappa': k, 'time': 0.0, 'SR': SR,
                          'waypoints': baseline_wp}

    # Method 2 — Uniform B-Spline (true uniform parameterization,
    # paired with uniform knots — the textbook-correct combination;
    # chord-length parameterization + uniform knots is ill-conditioned
    # for irregularly-spaced points like a jagged A* path)
    t0 = time.time()
    spline_uniform = BSplinePath(baseline_wp, param_mode='uniform', n_ctrl=14)
    t_uniform = time.time() - t0
    L, k, SR = compute_metrics(spline_uniform, True, env.start, env.goal)
    trial['uniform'] = {'length': L, 'kappa': k, 'time': t_uniform, 'SR': SR,
                         'spline': spline_uniform}

    # Method 3 — Classical B-Spline (chord-length parameterization)
    t0 = time.time()
    chord_knots  = chord_length_knot_vector(baseline_wp, degree=3)
    spline_chord = BSplinePath(baseline_wp, chord_knots)
    t_chord = time.time() - t0
    L, k, SR = compute_metrics(spline_chord, True, env.start, env.goal)
    trial['chord'] = {'length': L, 'kappa': k, 'time': t_chord, 'SR': SR,
                       'spline': spline_chord}

    # Method 4 — Proposed Enhanced Adaptive B-Spline (GA-optimized knots)
    t0 = time.time()
    ga = KnotGA(baseline_wp, n_generations=30, population_size=20)
    best_knots = ga.run()
    spline_adaptive = BSplinePath(baseline_wp, best_knots)
    t_adaptive = time.time() - t0
    L, k, SR = compute_metrics(spline_adaptive, True, env.start, env.goal)
    trial['adaptive'] = {'length': L, 'kappa': k, 'time': t_adaptive, 'SR': SR,
                          'spline': spline_adaptive}

    trial['env'] = env
    return trial


def average_method(trials, method_key, field):
    return np.mean([t[method_key][field] for t in trials])


if __name__ == '__main__':
    trials = [run_trial(seed) for seed in SEEDS]

    methods = ['baseline', 'uniform', 'chord', 'adaptive']
    display_names = {
        'baseline': 'Baseline Polyline',
        'uniform':  'Uniform B-Spline',
        'chord':    'Classical B-Spline',
        'adaptive': 'Adaptive B-Spline',
    }

    avg = {}
    for m in methods:
        avg[m] = {
            'length': average_method(trials, m, 'length'),
            'kappa':  average_method(trials, m, 'kappa'),
            'time':   average_method(trials, m, 'time'),
            'SR':     average_method(trials, m, 'SR'),
        }

    print("\n" + "=" * 55)
    print("Table 2: Performance of Four Path-Smoothing Methods")
    print("=" * 55)
<<<<<<< HEAD
    print(f"{'Method':<20} {'Length(m)':<10} {'Max kappa(m^-1)':<15} "
=======
    print(f"{'Method':<20} {'Length(m)':<10} {'Max_kappa':<11} "
>>>>>>> 8f5c5c48981b958f90da18e40fcb1cb2d8584ef3
          f"{'Time(s)':<9} {'SR'}")

    print("-" * 55)
    for m in methods:
        a = avg[m]
        print(f"{display_names[m]:<20} {a['length']:<10.1f} "
              f"{a['kappa']:<15.2f} {a['time']:<9.2f} {a['SR']:.3f}")
    print("=" * 55)

    print("Paper Reference Values:")
    ref = {
        'baseline': (2984.6, 0.65, 0.00, 1.107),
        'uniform':  (2840.8, 0.56, 0.25, 1.053),
        'chord':    (2803.2, 0.49, 0.31, 1.039),
        'adaptive': (2731.4, 0.38, 0.42, 1.013),
    }
    for m in methods:
        L, k, tm, sr = ref[m]
        print(f"{display_names[m]:<20} {L:<10.1f} {k:<11.2f} "
              f"{tm:<9.2f} {sr:.3f}")
    print("=" * 55)

    # ════════════════════════════════════════
    # Figure 4 recreation — first trial's environment + all 4 paths
    # ════════════════════════════════════════
    ref_trial = trials[0]
    env = ref_trial['env']

    fig = plt.figure(figsize=(13, 10))
    ax  = fig.add_subplot(111, projection='3d')

    baseline_pts = ref_trial['baseline']['waypoints']
    uniform_pts  = ref_trial['uniform']['spline'].sample_path(300)
    chord_pts    = ref_trial['chord']['spline'].sample_path(300)
    adaptive_pts = ref_trial['adaptive']['spline'].sample_path(300)

    env.visualize(ax)
    ax.plot(baseline_pts[:, 0], baseline_pts[:, 1], baseline_pts[:, 2],
            color='gray', lw=1.2, linestyle=':', label='Baseline Polyline (A*)')
    ax.plot(uniform_pts[:, 0], uniform_pts[:, 1], uniform_pts[:, 2],
            color='blue', lw=1.5, linestyle='--', label='Uniform B-Spline')
    ax.plot(chord_pts[:, 0], chord_pts[:, 1], chord_pts[:, 2],
            color='orange', lw=1.5, linestyle='-.', label='Classical B-Spline')
    ax.plot(adaptive_pts[:, 0], adaptive_pts[:, 1], adaptive_pts[:, 2],
            color='red', lw=2.5, label='Adaptive B-Spline (Proposed)')

    ax.set_title('Path Comparison — 4 Smoothing Methods (Paper 1 Figure 4)')
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig('results/paper1_path_comparison.png', dpi=150,
                bbox_inches='tight')
    print("\nSaved: results/paper1_path_comparison.png")
