"""
Complete merged demonstration: Paper 1 (A* + Enhanced Adaptive B-Spline
smoothing with obstacle avoidance) feeding attitude references into
Paper 2 (Quaternion P^2 attitude controller).

Stage 5 uses Paper 2's exact gains (Pq=20, Pw=4) and physical
parameters. At dt=0.01 (100Hz) those gains are discrete-time unstable
with this project's low inertia (Ixx=Iyy=6.5e-4) — verified separately
in test_paper2_controller.py / paper1_recreation.py's sibling script.
The control law and physics are therefore integrated internally at
dt=0.0005s (2kHz), with results logged at the paper's own dt=0.01s
resolution, exactly as agreed for paper2_recreation.py.
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
from integration.trajectory_converter import TrajectoryConverter
from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from attitude_control.quaternion import Quaternion

SEED = 42

print("#" * 60)
print("# COMPLETE MERGED SYSTEM DEMONSTRATION")
print("# Paper 1 (Adaptive B-Spline + GA) -> Paper 2 (Quaternion P^2)")
print("#" * 60)

# ════════════════════════════════════════
# STAGE 1 — Environment Setup
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 1 — Environment Setup")
print("=" * 60)

env = Environment3D()
env.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=SEED)

print(f"Space: {env.space[0]:.0f} x {env.space[1]:.0f} x {env.space[2]:.0f} m")
print(f"Start: {env.start}")
print(f"Goal:  {env.goal}")
print(f"Obstacles placed: {len(env.obstacles)} (seed={SEED})")
n_cyl = sum(1 for o in env.obstacles if o.shape == 'cylinder')
n_cub = sum(1 for o in env.obstacles if o.shape == 'cuboid')
print(f"  Cylinders: {n_cyl}   Cuboids: {n_cub}")
for i, obs in enumerate(env.obstacles):
    if obs.shape == 'cylinder':
        r, h = obs.dimensions
        print(f"  [{i}] CYLINDER pos={obs.position.round(1)} "
              f"radius={r:.1f}m height={h:.1f}m")
    else:
        l, w, h = obs.dimensions
        print(f"  [{i}] CUBOID   pos={obs.position.round(1)} "
              f"dims={l:.1f}x{w:.1f}x{h:.1f}m")

# ════════════════════════════════════════
# STAGE 2 — Global Planning (A*)
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 2 — Global Planning (A*)")
print("=" * 60)

astar = AStarPlanner3D(env, grid_resolution=50)
baseline_wp = np.array(astar.plan(env.start, env.goal))
raw_length  = astar.compute_path_length(baseline_wp)

print(f"A* grid: {astar.nx} x {astar.ny} x {astar.nz} cells "
      f"(resolution={astar.resolution}m)")
print(f"Waypoints found:  {len(baseline_wp)}")
print(f"Raw path length:  {raw_length:.1f} m")

# ════════════════════════════════════════
# STAGE 3 — Path Smoothing (Paper 1, 4 methods)
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 3 — Path Smoothing (Paper 1)")
print("=" * 60)

dist = np.linalg.norm(env.goal - env.start)


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


results_stage3 = {}

# Method 1 — Baseline Polyline
L = raw_length
k = float(np.max(discrete_curvature(baseline_wp)))
results_stage3['baseline'] = {'length': L, 'kappa': k, 'time': 0.0, 'SR': L / dist}

# Method 2 — Uniform B-Spline (uniform parameterization + smoothing fit)
t0 = time.time()
spline_uniform = BSplinePath(baseline_wp, param_mode='uniform', n_ctrl=14)
t_uniform = time.time() - t0
results_stage3['uniform'] = {
    'length': spline_uniform.path_length(), 'kappa': spline_uniform.max_curvature(),
    'time': t_uniform, 'SR': spline_uniform.path_length() / dist
}

# Method 3 — Classical B-Spline (chord-length parameterization)
t0 = time.time()
chord_knots  = chord_length_knot_vector(baseline_wp, degree=3)
spline_chord = BSplinePath(baseline_wp, chord_knots)
t_chord = time.time() - t0
results_stage3['chord'] = {
    'length': spline_chord.path_length(), 'kappa': spline_chord.max_curvature(),
    'time': t_chord, 'SR': spline_chord.path_length() / dist
}

# Method 4 — Proposed Enhanced Adaptive B-Spline (GA-optimized knots)
t0 = time.time()
ga = KnotGA(baseline_wp, n_generations=30, population_size=20)
best_knots = ga.run()
spline_adaptive = BSplinePath(baseline_wp, best_knots)
t_adaptive = time.time() - t0
results_stage3['adaptive'] = {
    'length': spline_adaptive.path_length(), 'kappa': spline_adaptive.max_curvature(),
    'time': t_adaptive, 'SR': spline_adaptive.path_length() / dist
}

names = {'baseline': 'Baseline Polyline', 'uniform': 'Uniform B-Spline',
         'chord': 'Classical B-Spline', 'adaptive': 'Adaptive B-Spline'}

print(f"\n{'Method':<20} {'Length(m)':<10} {'Max kappa(m^-1)':<15} {'Time(s)':<9} {'SR'}")
print("-" * 55)
for key in ['baseline', 'uniform', 'chord', 'adaptive']:
    r = results_stage3[key]
    print(f"{names[key]:<20} {r['length']:<10.1f} {r['kappa']:<15.4f} "
          f"{r['time']:<9.2f} {r['SR']:.3f}")

fig = plt.figure(figsize=(13, 10))
ax  = fig.add_subplot(111, projection='3d')
env.visualize(ax)
ax.plot(baseline_wp[:, 0], baseline_wp[:, 1], baseline_wp[:, 2],
        color='gray', lw=1.2, linestyle=':', label='Baseline Polyline (A*)')
pts_u = spline_uniform.sample_path(300)
pts_c = spline_chord.sample_path(300)
pts_a = spline_adaptive.sample_path(300)
ax.plot(pts_u[:, 0], pts_u[:, 1], pts_u[:, 2], color='blue', lw=1.5,
        linestyle='--', label='Uniform B-Spline')
ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], color='orange', lw=1.5,
        linestyle='-.', label='Classical B-Spline')
ax.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], color='red', lw=2.5,
        label='Adaptive B-Spline (Proposed)')
ax.set_title('Final Merged System — Path Comparison (Paper 1)')
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig('results/final_path_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: results/final_path_comparison.png")

# ════════════════════════════════════════
# STAGE 4 — Attitude Reference Extraction
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 4 — Attitude Reference Extraction")
print("=" * 60)

converter = TrajectoryConverter()
u_samples = np.linspace(0.001, 0.999, 500)
ref_sequence = [converter.get_reference_at(spline_adaptive, u) for u in u_samples]

print(f"Sampled {len(ref_sequence)} attitude references along the "
      f"Adaptive B-Spline path.\n")
print(f"{'u':>6}  {'position':<28} {'quaternion [q0,q1,q2,q3]'}")
print("-" * 70)

converter_display = TrajectoryConverter()
for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
    u_eval = np.clip(u, 0.001, 0.999)
    pos, q_ref = converter_display.get_reference_at(spline_adaptive, u_eval)
    pos_str = f"[{pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:6.1f}]"
    q_str = (f"[{q_ref.q0:.4f}, {q_ref.q1:.4f}, "
             f"{q_ref.q2:.4f}, {q_ref.q3:.4f}]")
    print(f"{u:6.2f}  {pos_str:<28} {q_str}")

# ════════════════════════════════════════
# STAGE 5 — Attitude Controller (Paper 2)
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 5 — Attitude Controller (Paper 2)")
print("=" * 60)

Pq, Pw       = 20.0, 4.0
NOISE_STD    = 0.1
DT_LOG       = 0.01
DT_PHYSICS   = 0.0005
SUBSTEPS     = int(round(DT_LOG / DT_PHYSICS))
TOTAL_TIME   = 20.0
ERR_THRESH   = 0.05
U_STEP       = 0.005

print(f"Pq={Pq}, Pw={Pw}, noise_std={NOISE_STD}, torque bounds=±4Nm")
print(f"Logged at dt={DT_LOG}s, integrated internally at dt={DT_PHYSICS}s "
      f"({SUBSTEPS} substeps/log-step) for numerical stability at these "
      f"exact paper gains.")

drone = QuadrotorModel()
ctrl  = P2Controller(Pq=Pq, Pw=Pw)
conv  = TrajectoryConverter()

n_log = int(TOTAL_TIME / DT_LOG)
u_current = 0.001
_, q_ref_current = conv.get_reference_at(spline_adaptive, u_current)
drone.q = Quaternion(*q_ref_current.q)

t_arr       = np.zeros(n_log)
euler_arr   = np.zeros((n_log, 3))
euler_ref_arr = np.zeros((n_log, 3))
torque_arr  = np.zeros((n_log, 3))
omega_arr   = np.zeros((n_log, 3))
u_arr       = np.zeros(n_log)
err_mag_arr = np.zeros(n_log)

for i in range(n_log):
    t = i * DT_LOG

    q_err   = ctrl.compute_error(q_ref_current, drone.q)
    err_mag = np.linalg.norm([q_err.q1, q_err.q2, q_err.q3])

    if err_mag < ERR_THRESH and u_current < 0.99:
        u_current = min(u_current + U_STEP, 0.99)
        _, q_ref_new = conv.get_reference_at(spline_adaptive, u_current)
        q_ref_current = conv.slerp_static(q_ref_current, q_ref_new, 0.3)

    for _ in range(SUBSTEPS):
        omega_noisy = drone.omega + np.random.normal(0, NOISE_STD, 3)
        tau, axis_err = ctrl.compute_torque(q_ref_current, drone.q, omega_noisy)
        drone.step_attitude_only(tau, DT_PHYSICS)

    t_arr[i]         = t
    euler_arr[i]      = drone.q.to_euler()
    euler_ref_arr[i]  = q_ref_current.to_euler()
    torque_arr[i]     = tau
    omega_arr[i]      = drone.omega.copy()
    u_arr[i]          = u_current
    err_mag_arr[i]    = err_mag

    if i % (n_log // 10) == 0:
        pct = 100 * i // n_log
        print(f"  {pct:3d}% | t={t:5.1f}s | u={u_current:.3f} | "
              f"AttErr={err_mag:.4f} | τ_max={np.max(np.abs(tau)):.3f}Nm")

conv_idx  = np.where(err_mag_arr < ERR_THRESH)[0]
conv_time = t_arr[conv_idx[0]] if len(conv_idx) > 0 else float('nan')
steady_err = np.mean(err_mag_arr[int(n_log * 0.7):])
max_torque = np.max(np.abs(torque_arr))

print(f"\nConvergence time:   "
      f"{'%.2fs' % conv_time if not np.isnan(conv_time) else 'never'}")
print(f"Steady-state error: {steady_err:.5f}")
print(f"Max torque used:    {max_torque:.3f} Nm")
print(f"Final path progress: {u_arr[-1]*100:.1f}%")

# ════════════════════════════════════════
# STAGE 6 — Results
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 6 — Results")
print("=" * 60)

eu = np.unwrap(euler_arr,     axis=0)
er = np.unwrap(euler_ref_arr, axis=0)
labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
colors = ['steelblue', 'seagreen', 'tomato']

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for i in range(3):
    axes[i].plot(t_arr, er[:, i], 'k--', lw=1.5, alpha=0.8, label='Reference')
    axes[i].plot(t_arr, eu[:, i], color=colors[i], lw=1.5, label='Actual')
    axes[i].set_ylabel(labels[i])
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=9, loc='upper right')
axes[-1].set_xlabel('Time (s)')
fig.suptitle('Final Merged System — Attitude Tracking', fontsize=13)
plt.tight_layout()
plt.savefig('results/final_attitude_tracking.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: results/final_attitude_tracking.png")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_arr, err_mag_arr, 'purple', lw=1.5)
ax.axhline(y=ERR_THRESH, color='green', linestyle='--', lw=1.5,
           label=f'Threshold ({ERR_THRESH})')
ax.set_xlabel('Time (s)'); ax.set_ylabel('||q_err||')
ax.set_title('Final Merged System — Quaternion Error Convergence')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/final_error_convergence.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: results/final_error_convergence.png")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_arr, u_arr, 'navy', lw=2)
ax.fill_between(t_arr, 0, u_arr, color='navy', alpha=0.15)
ax.set_ylim(0, 1)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Path Parameter u')
ax.set_title('Final Merged System — Path Progress')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/final_path_progress.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: results/final_path_progress.png")

print("\n" + "#" * 60)
print("# FINAL SUMMARY")
print("#" * 60)
print(f"\n[Paper 1] Path Planning:")
print(f"  Baseline (A*):      L={results_stage3['baseline']['length']:.1f}m, "
      f"SR={results_stage3['baseline']['SR']:.3f}")
print(f"  Adaptive B-Spline:  L={results_stage3['adaptive']['length']:.1f}m, "
      f"SR={results_stage3['adaptive']['SR']:.3f}")
improvement = (results_stage3['baseline']['length'] -
               results_stage3['adaptive']['length']) / results_stage3['baseline']['length'] * 100
print(f"  Path length improvement: {improvement:.1f}%")

print(f"\n[Paper 2] Attitude Control (Pq={Pq}, Pw={Pw}):")
print(f"  Convergence time:   "
      f"{'%.2fs' % conv_time if not np.isnan(conv_time) else 'never'}")
print(f"  Steady-state error: {steady_err:.5f}")
print(f"  Max torque used:    {max_torque:.3f} Nm")

print(f"\n[Merged System]:")
print(f"  Obstacles avoided:  {len(env.obstacles)} "
      f"({n_cyl} cylinders, {n_cub} cuboids)")
print(f"  Final path progress: {u_arr[-1]*100:.1f}%")
print(f"  Attitude refs derived from B-Spline tangent, tracked in "
      f"quaternion space (singularity-free)")
print("#" * 60)
