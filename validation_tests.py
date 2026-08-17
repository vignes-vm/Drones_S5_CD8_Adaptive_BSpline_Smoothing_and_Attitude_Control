"""
Validation across multiple test scenarios — robustness check for the
merged Paper 1 (A* + 4 path-smoothing methods) pipeline.
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


def discrete_curvature(waypoints):
    pts = np.asarray(waypoints, dtype=float)
    if len(pts) < 3:
        return np.array([0.0])
    curvatures = []
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i] - pts[i - 1], pts[i + 1] - pts[i], pts[i + 1] - pts[i - 1]
        la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        denom = la * lb * lc
        curvatures.append(0.0 if denom < 1e-10 else
                           2 * np.linalg.norm(np.cross(a, b)) / denom)
    return np.array(curvatures)


def chord_length_knot_vector(waypoints, degree=3):
    pts = np.asarray(waypoints, dtype=float)
    n, p = len(pts) - 1, degree
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(dists)])
    total = cumulative[-1]
    u = cumulative / total if total > 1e-10 else np.linspace(0, 1, n + 1)
    knots = np.zeros(n + p + 2)
    knots[-(p + 1):] = 1.0
    for j in range(1, n - p + 1):
        knots[j + p] = np.sum(u[j:j + p]) / p
    return knots


def run_scenario(name, start, goal, n_cyl, n_cub, seed=42, notes=""):
    print(f"\n{'='*60}")
    print(f"{name}")
    if notes:
        print(f"  ({notes})")
    print(f"{'='*60}")

    env = Environment3D()
    env.start = np.array(start, dtype=float)
    env.goal  = np.array(goal, dtype=float)
    env.add_random_obstacles(n_cylinders=n_cyl, n_cuboids=n_cub, seed=seed)

    astar = AStarPlanner3D(env, grid_resolution=50)
    baseline_wp = np.array(astar.plan(env.start, env.goal))
    dist = np.linalg.norm(env.goal - env.start)

    metrics = {}

    L = astar.compute_path_length(baseline_wp)
    k = float(np.max(discrete_curvature(baseline_wp)))
    metrics['baseline'] = {'length': L, 'kappa': k, 'SR': L / dist}

    spline_uniform = BSplinePath(baseline_wp, param_mode='uniform', n_ctrl=14)
    metrics['uniform'] = {
        'length': spline_uniform.path_length(),
        'kappa':  spline_uniform.max_curvature(),
        'SR':     spline_uniform.path_length() / dist,
    }

    chord_knots  = chord_length_knot_vector(baseline_wp, degree=3)
    spline_chord = BSplinePath(baseline_wp, chord_knots)
    metrics['chord'] = {
        'length': spline_chord.path_length(),
        'kappa':  spline_chord.max_curvature(),
        'SR':     spline_chord.path_length() / dist,
    }

    # kappa_max tuned to this environment's actual curvature scale
    # (achieved curvatures are ~0.003-0.04, so the paper's kappa_max=0.5
    # default never activates the GA's curvature penalty). n_inner_knots=10
    # gives n_ctrl=14, matching Uniform/Classical's control-point budget
    # for a fair comparison.
    ga = KnotGA(baseline_wp, n_generations=30, population_size=20,
                n_inner_knots=10, kappa_max=0.05)
    best_knots = ga.run()
    spline_adaptive = BSplinePath(baseline_wp, best_knots)
    metrics['adaptive'] = {
        'length': spline_adaptive.path_length(),
        'kappa':  spline_adaptive.max_curvature(),
        'SR':     spline_adaptive.path_length() / dist,
    }

    print(f"\n{'Method':<20} {'Length(m)':<10} {'Max_kappa':<11} {'SR'}")
    print("-" * 50)
    names = {'baseline': 'Baseline', 'uniform': 'Uniform',
             'chord': 'Classical', 'adaptive': 'Adaptive'}
    for key in ['baseline', 'uniform', 'chord', 'adaptive']:
        m = metrics[key]
        print(f"{names[key]:<20} {m['length']:<10.1f} {m['kappa']:<11.4f} {m['SR']:.3f}")

    len_improve = ((metrics['uniform']['length'] - metrics['adaptive']['length'])
                   / metrics['uniform']['length'] * 100)
    kappa_improve = (0.0 if metrics['uniform']['kappa'] < 1e-10 else
                     (metrics['uniform']['kappa'] - metrics['adaptive']['kappa'])
                     / metrics['uniform']['kappa'] * 100)
    print(f"\nGA improvement over Uniform: {len_improve:.1f}% shorter, "
          f"{kappa_improve:.1f}% less curvature")

    return {
        'name': name, 'env': env, 'baseline_wp': baseline_wp,
        'spline_uniform': spline_uniform, 'spline_chord': spline_chord,
        'spline_adaptive': spline_adaptive, 'metrics': metrics,
        'len_improve': len_improve, 'kappa_improve': kappa_improve,
    }


SCENARIOS = [
    dict(name="SCENARIO 1 — Sparse Obstacles (Easy)",
         start=[0, 0, 20], goal=[2000, 1800, 200],
         n_cyl=3, n_cub=2, seed=42,
         notes="Expected: short path, low curvature"),
    dict(name="SCENARIO 2 — Dense Obstacles (Hard)",
         start=[0, 0, 20], goal=[2000, 1800, 200],
         n_cyl=8, n_cub=7, seed=42,
         notes="Expected: longer path, higher curvature, clearer GA advantage"),
    dict(name="SCENARIO 3 — Different Start/Goal",
         start=[0, 1000, 50], goal=[2000, 800, 150],
         n_cyl=5, n_cub=5, seed=42,
         notes="Cross-environment start/goal pair"),
    dict(name="SCENARIO 4 — High Altitude Path",
         start=[0, 0, 400], goal=[2000, 1800, 450],
         n_cyl=5, n_cub=5, seed=42,
         notes="Obstacles randomly sized 100-400m (cylinders) / 100-300m "
               "(cuboids) tall, resting on the ground — expected to fly "
               "mostly above the shorter ones"),
    dict(name="SCENARIO 5 — Paper Exact Scenario",
         start=[0, 0, 20], goal=[2000, 1800, 200],
         n_cyl=5, n_cub=5, seed=42,
         notes="Should give results closest to Table 2"),
]

if __name__ == '__main__':
    all_results = []
    for sc in SCENARIOS:
        result = run_scenario(sc['name'], sc['start'], sc['goal'],
                               sc['n_cyl'], sc['n_cub'], sc['seed'], sc['notes'])
        all_results.append(result)

    # ════════════════════════════════════════
    # Combined figure — 5 subplots
    # ════════════════════════════════════════
    fig = plt.figure(figsize=(20, 12))
    for i, res in enumerate(all_results):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        env = res['env']
        env.visualize(ax)
        ax.plot(res['baseline_wp'][:, 0], res['baseline_wp'][:, 1],
                res['baseline_wp'][:, 2], color='gray', lw=1.0,
                linestyle=':', label='Baseline')
        pts_u = res['spline_uniform'].sample_path(200)
        pts_c = res['spline_chord'].sample_path(200)
        pts_a = res['spline_adaptive'].sample_path(200)
        ax.plot(pts_u[:, 0], pts_u[:, 1], pts_u[:, 2], color='blue',
                lw=1.2, linestyle='--', label='Uniform')
        ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], color='orange',
                lw=1.2, linestyle='-.', label='Classical')
        ax.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], color='red',
                lw=2.0, label='Adaptive')
        ax.set_title(res['name'], fontsize=10)
        ax.legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    plt.savefig('results/validation_all_scenarios.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: results/validation_all_scenarios.png")

    # ════════════════════════════════════════
    # Overall summary
    # ════════════════════════════════════════
    len_improvements   = [r['len_improve'] for r in all_results]
    kappa_improvements = [r['kappa_improve'] for r in all_results]

    chord_len_improve = [
        (r['metrics']['chord']['length'] - r['metrics']['adaptive']['length'])
        / r['metrics']['chord']['length'] * 100
        for r in all_results
    ]

    best_idx  = int(np.argmax(len_improvements))
    worst_idx = int(np.argmin(len_improvements))

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY — Adaptive B-Spline vs. Uniform / Classical")
    print(f"{'='*60}")
    print(f"Average improvement over Uniform:   "
          f"{np.mean(len_improvements):.1f}% shorter, "
          f"{np.mean(kappa_improvements):.1f}% less curvature")
    print(f"Average improvement over Classical: "
          f"{np.mean(chord_len_improve):.1f}% shorter")
    print(f"\nBest case:  {all_results[best_idx]['name']} "
          f"({len_improvements[best_idx]:.1f}% shorter than Uniform)")
    print(f"Worst case: {all_results[worst_idx]['name']} "
          f"({len_improvements[worst_idx]:.1f}% shorter than Uniform)")
    print(f"{'='*60}")
