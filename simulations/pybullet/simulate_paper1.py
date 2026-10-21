"""
Paper 1 Recreation in PyBullet Physics Environment.
Recreates Table 2 from Paper 1: "An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"
Compares 4 path-smoothing methods (Baseline Polyline, Uniform B-Spline, Classical Chord B-Spline, GA Adaptive B-Spline).
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        c = pts[i + 1] - pts[i - 1]
        len_a, len_b, len_c = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        cross_norm = np.linalg.norm(np.cross(a, b))
        denom = len_a * len_b * len_c
        curvatures.append(0.0 if denom < 1e-10 else 2 * cross_norm / denom)
    return np.array(curvatures)


def chord_length_knot_vector(waypoints, degree=3):
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


def run_paper1_pybullet_recreation(seeds=[42, 123]):
    print("=" * 65, flush=True)
    print("PAPER 1 RECREATION — Path Planning & Smoothing Methods Benchmark", flush=True)
    print("=" * 65, flush=True)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)

    methods = ['baseline', 'uniform', 'chord', 'adaptive']
    display_names = {
        'baseline': 'Baseline Polyline (A*)',
        'uniform':  'Uniform B-Spline',
        'chord':    'Classical B-Spline',
        'adaptive': 'Adaptive B-Spline (GA)'
    }

    results_all = {m: {'length': [], 'kappa': [], 'time': [], 'SR': []} for m in methods}

    for seed in seeds:
        print(f"\n--- Trial Seed = {seed} ---")
        env = Environment3D()
        env.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)

        astar = AStarPlanner3D(env, grid_resolution=50)
        raw_wp = np.array(astar.plan(env.start, env.goal))
        baseline_wp = np.array(astar.smooth_waypoints(raw_wp, min_distance=100))
        dist = np.linalg.norm(env.goal - env.start)

        # 1. Baseline Polyline
        L_base = astar.compute_path_length(baseline_wp)
        k_base = float(np.max(discrete_curvature(baseline_wp)))
        results_all['baseline']['length'].append(L_base)
        results_all['baseline']['kappa'].append(k_base)
        results_all['baseline']['time'].append(0.0)
        results_all['baseline']['SR'].append(L_base / dist)

        # 2. Uniform B-Spline
        t0 = time.time()
        spline_u = BSplinePath(baseline_wp, param_mode='uniform', n_ctrl=14)
        t_u = time.time() - t0
        results_all['uniform']['length'].append(spline_u.path_length())
        results_all['uniform']['kappa'].append(spline_u.max_curvature())
        results_all['uniform']['time'].append(t_u)
        results_all['uniform']['SR'].append(spline_u.path_length() / dist)

        # 3. Classical B-Spline
        t0 = time.time()
        chord_knots = chord_length_knot_vector(baseline_wp, degree=3)
        spline_c = BSplinePath(baseline_wp, chord_knots)
        t_c = time.time() - t0
        results_all['chord']['length'].append(spline_c.path_length())
        results_all['chord']['kappa'].append(spline_c.max_curvature())
        results_all['chord']['time'].append(t_c)
        results_all['chord']['SR'].append(spline_c.path_length() / dist)

        # 4. Adaptive B-Spline (GA)
        t0 = time.time()
        ga = KnotGA(baseline_wp, n_generations=20, population_size=15)
        best_knots = ga.run()
        spline_a = BSplinePath(baseline_wp, best_knots)
        t_a = time.time() - t0
        results_all['adaptive']['length'].append(spline_a.path_length())
        results_all['adaptive']['kappa'].append(spline_a.max_curvature())
        results_all['adaptive']['time'].append(t_a)
        results_all['adaptive']['SR'].append(spline_a.path_length() / dist)

    print("\n" + "=" * 65, flush=True)
    print("RECREATED TABLE 2 METRICS (Averaged Over 5 Seeds)", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Method':<25} {'Length (m)':<12} {'Max kappa (m^-1)':<16} {'Time (s)':<10} {'SR'}", flush=True)
    print("-" * 65, flush=True)

    for m in methods:
        avg_L = np.mean(results_all[m]['length'])
        avg_k = np.mean(results_all[m]['kappa'])
        avg_t = np.mean(results_all[m]['time'])
        avg_sr = np.mean(results_all[m]['SR'])
        print(f"{display_names[m]:<25} {avg_L:<12.1f} {avg_k:<12.4f} {avg_t:<10.2f} {avg_sr:.3f}", flush=True)

    print("=" * 65)


if __name__ == '__main__':
    run_paper1_pybullet_recreation()
