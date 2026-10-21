"""
Paper 1 Recreation in Gazebo Airspace Environment.
Recreates Table 2 of Paper 1: "An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"
Compares 4 path planning & smoothing methods:
  1. Baseline Polyline (3D A*)
  2. Uniform B-Spline
  3. Classical Chord-Length B-Spline
  4. Proposed Enhanced Adaptive B-Spline (GA Knot Optimization)
Evaluates Path Length, Max Curvature, Computation Time, and Straightness Ratio.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.bspline import BSplinePath
from path_planning.knot_ga import KnotGA
from gazebo.src.gazebo_env import generate_gazebo_world_sdf


def discrete_curvature(waypoints):
    """Computes discrete curvature along a polygonal path."""
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
        curvatures.append(0.0 if denom < 1e-10 else 2.0 * cross_norm / denom)
    return np.array(curvatures)


def chord_length_knot_vector(waypoints, degree=3):
    """Computes classical chord-length parameterization knot vector with clamped ends."""
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


def run_paper1_gazebo_benchmark(seeds=[42, 123]):
    print("=" * 70, flush=True)
    print("PAPER 1 BENCHMARK IN GAZEBO AIRSPACE", flush=True)
    print("Comparing 4 UAV Path Planning & Smoothing Methods", flush=True)
    print("=" * 70, flush=True)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)

    methods = ['baseline', 'uniform', 'chord', 'adaptive']
    display_names = {
        'baseline': 'Baseline Polyline (A*)',
        'uniform':  'Uniform B-Spline',
        'chord':    'Classical B-Spline',
        'adaptive': 'Adaptive B-Spline (GA)'
    }

    results = {m: {'length': [], 'kappa': [], 'time': [], 'SR': []} for m in methods}
    sample_vis_data = None

    for seed in seeds:
        print(f"\n--- Running Seed = {seed} ---", flush=True)
        env = Environment3D()
        env.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)

        # Generate Gazebo SDF world for this seed
        sdf_path = os.path.join(os.path.dirname(__file__), 'worlds', f'airspace_seed_{seed}.sdf')
        generate_gazebo_world_sdf(env, sdf_path, world_name=f"airspace_seed_{seed}")

        # 3D A* Global Planning
        astar = AStarPlanner3D(env, grid_resolution=50)
        raw_wp = np.array(astar.plan(env.start, env.goal))
        baseline_wp = np.array(astar.smooth_waypoints(raw_wp, min_distance=100))
        dist_direct = np.linalg.norm(env.goal - env.start)

        # 1. Baseline Polyline
        L_base = astar.compute_path_length(baseline_wp)
        k_base = float(np.max(discrete_curvature(baseline_wp)))
        results['baseline']['length'].append(L_base)
        results['baseline']['kappa'].append(k_base)
        results['baseline']['time'].append(0.0)
        results['baseline']['SR'].append(L_base / dist_direct)

        # 2. Uniform B-Spline
        t0 = time.time()
        spline_u = BSplinePath(baseline_wp, param_mode='uniform', n_ctrl=14)
        t_u = time.time() - t0
        results['uniform']['length'].append(spline_u.path_length())
        results['uniform']['kappa'].append(spline_u.max_curvature())
        results['uniform']['time'].append(t_u)
        results['uniform']['SR'].append(spline_u.path_length() / dist_direct)

        # 3. Classical B-Spline (Chord-length knots)
        t0 = time.time()
        knots_chord = chord_length_knot_vector(baseline_wp, degree=3)
        spline_c = BSplinePath(baseline_wp, knots_chord)
        t_c = time.time() - t0
        results['chord']['length'].append(spline_c.path_length())
        results['chord']['kappa'].append(spline_c.max_curvature())
        results['chord']['time'].append(t_c)
        results['chord']['SR'].append(spline_c.path_length() / dist_direct)

        # 4. Adaptive B-Spline (GA-Optimized Knots)
        t0 = time.time()
        ga = KnotGA(baseline_wp, n_generations=20, population_size=15)
        best_knots = ga.run()
        spline_a = BSplinePath(baseline_wp, best_knots)
        t_a = time.time() - t0
        results['adaptive']['length'].append(spline_a.path_length())
        results['adaptive']['kappa'].append(spline_a.max_curvature())
        results['adaptive']['time'].append(t_a)
        results['adaptive']['SR'].append(spline_a.path_length() / dist_direct)

        if sample_vis_data is None:
            sample_vis_data = (env, baseline_wp, spline_u, spline_c, spline_a)

    # Print Aggregate Results Table
    print("\n" + "=" * 75, flush=True)
    print(f"{'Method':<25} {'Length (m)':<12} {'Max κ (m⁻¹)':<15} {'Time (s)':<12} {'Straightness'}")
    print("-" * 75, flush=True)
    for m in methods:
        avg_L = np.mean(results[m]['length'])
        avg_k = np.mean(results[m]['kappa'])
        avg_t = np.mean(results[m]['time'])
        avg_sr = np.mean(results[m]['SR'])
        print(f"{display_names[m]:<25} {avg_L:<12.1f} {avg_k:<15.4f} {avg_t:<12.2f} {avg_sr:.3f}")
    print("=" * 75, flush=True)

    # Visualization
    env_sample, b_wp, s_u, s_c, s_a = sample_vis_data
    pts_u = s_u.sample_path(400)
    pts_c = s_c.sample_path(400)
    pts_a = s_a.sample_path(400)

    fig = plt.figure(figsize=(16, 6.5))

    # Subplot 1: 3D Paths in Gazebo Environment
    ax1 = fig.add_subplot(131, projection='3d')
    env_sample.visualize(ax1)
    ax1.plot(b_wp[:, 0], b_wp[:, 1], b_wp[:, 2], 'k:', lw=1.5, label='Baseline (A*)')
    ax1.plot(pts_u[:, 0], pts_u[:, 1], pts_u[:, 2], 'b--', lw=1.5, label='Uniform B-Spline')
    ax1.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], 'g-.', lw=1.5, label='Classical B-Spline')
    ax1.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], 'r-', lw=2.5, label='Adaptive B-Spline (GA)')
    ax1.set_title('Full Airspace', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left')

    # Subplot 2: zoomed top-down (XY) view of the segment where the 4 paths diverge the most,
    # since at full ~2000 m scale the curves sit on top of each other (they differ by <1%).
    def resample_xy(pts, n_ref):
        idx = np.linspace(0, len(pts) - 1, n_ref)
        return np.column_stack([np.interp(idx, np.arange(len(pts)), pts[:, 0]),
                                np.interp(idx, np.arange(len(pts)), pts[:, 1])])
    n_ref = len(pts_a)
    ref_xy = resample_xy(pts_a, n_ref)
    spread = np.zeros(n_ref)
    for candidate in (pts_u, pts_c):
        cand_xy = resample_xy(candidate, n_ref)
        spread += np.linalg.norm(cand_xy - ref_xy, axis=1)
    peak_i = int(np.argmax(spread))
    window = max(15, n_ref // 12)
    lo, hi = max(0, peak_i - window), min(n_ref, peak_i + window)

    ax2 = fig.add_subplot(132)
    ax2.plot(b_wp[:, 0], b_wp[:, 1], 'ko:', lw=1.2, ms=3, label='Baseline (A*)')
    ax2.plot(pts_u[lo:hi, 0], pts_u[lo:hi, 1], 'b--', lw=2.0, label='Uniform B-Spline')
    ax2.plot(pts_c[lo:hi, 0], pts_c[lo:hi, 1], 'g-.', lw=2.0, label='Classical B-Spline')
    ax2.plot(pts_a[lo:hi, 0], pts_a[lo:hi, 1], 'r-', lw=2.5, label='Adaptive B-Spline (GA)')
    ax2.set_xlim(pts_a[lo:hi, 0].min() - 10, pts_a[lo:hi, 0].max() + 10)
    ax2.set_ylim(pts_a[lo:hi, 1].min() - 10, pts_a[lo:hi, 1].max() + 10)
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('Zoom: largest path divergence', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.set_aspect('equal', adjustable='datalim')

    # Subplot 3: Curvature Profiles Along Path, auto-scaled to the data. The 0.5 m^-1 design
    # limit is 30-50x larger than what any method actually uses here, so drawing it to scale
    # would flatten the differences between methods; it is reported as an annotation instead.
    ax3 = fig.add_subplot(133)
    u_vals = np.linspace(0.01, 0.99, 300)
    k_u = np.array([s_u.compute_curvature(u) for u in u_vals])
    k_c = np.array([s_c.compute_curvature(u) for u in u_vals])
    k_a = np.array([s_a.compute_curvature(u) for u in u_vals])

    ax3.plot(u_vals, k_u, 'b--', lw=1.5, label=f'Uniform (max={k_u.max():.4f})')
    ax3.plot(u_vals, k_c, 'g-.', lw=1.5, label=f'Classical (max={k_c.max():.4f})')
    ax3.plot(u_vals, k_a, 'r-', lw=2.0, label=f'Adaptive GA (max={k_a.max():.4f})')
    data_max = max(k_u.max(), k_c.max(), k_a.max())
    ax3.set_ylim(0, data_max * 1.25)
    ax3.text(0.03, 0.95, f'κ_max design limit = 0.5 m⁻¹\n(all methods use <{100*data_max/0.5:.0f}% of it)',
             transform=ax3.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax3.set_xlabel('Path Parameter u', fontsize=10)
    ax3.set_ylabel('Curvature κ (m⁻¹)', fontsize=10)
    ax3.set_title('Curvature Profile (zoomed)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle('Gazebo 3D Airspace Path Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'gazebo_paper1_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved Artifact] {plot_file}", flush=True)

    # Metrics bar chart: the clearest "Table 2 as a chart" view for a presentation, since the
    # 4 methods' paths are visually almost identical but their metrics differ measurably.
    fig2, bar_axes = plt.subplots(1, 4, figsize=(18, 5))
    metric_specs = [
        ('length', 'Path Length (m)', '{:.0f}'),
        ('kappa', 'Max Curvature (m⁻¹)', '{:.4f}'),
        ('time', 'Computation Time (s)', '{:.2f}'),
        ('SR', 'Straightness Ratio', '{:.3f}'),
    ]
    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#d62728']
    for ax, (key, label, fmt) in zip(bar_axes, metric_specs):
        vals = [np.mean(results[m][key]) for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([display_names[m] for m in methods], fontsize=8, rotation=20, ha='right')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        span = max(vals) - min(vals)
        pad = span * 0.2 if span > 1e-9 else max(vals) * 0.1
        if key == 'time':
            # Comp time genuinely spans ~0s to several seconds (GA search dominates) -
            # a 0 baseline is the honest comparison here, not a scale trick.
            ax.set_ylim(0, max(vals) + pad)
        else:
            # Length/curvature/straightness differ by a fraction of a percent between
            # methods; zoom to the data range instead of a 0 baseline, or the bars
            # would look identical like the overlapping path plots did.
            ax.set_ylim(max(0, min(vals) - pad), max(vals) + pad)
        for bar, v in zip(bars, vals):
            label_y = bar.get_height() + pad * 0.05 if bar.get_height() >= 0 else bar.get_height() - pad * 0.15
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    fmt.format(v), ha='center', va='bottom', fontsize=7.5)
    fig2.suptitle('Paper 1 Benchmark — Method Comparison (averaged over seeds)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, wspace=0.35)
    bar_file = os.path.join(output_dir, 'gazebo_paper1_metrics.png')
    plt.savefig(bar_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved Artifact] {bar_file}", flush=True)


if __name__ == '__main__':
    run_paper1_gazebo_benchmark()
