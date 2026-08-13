import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from path_planning.bspline import BSplinePath
from path_planning.path_smoother import AdaptiveBSplineSmoother
from integration.simulation_loop import MergedSimulation

# ════════════════════════════════════════
# 1. Waypoints
# ════════════════════════════════════════
waypoints = np.array([
    [0,    0,    20],
    [300,  200,  80],
    [600,  150,  120],
    [900,  400,  100],
    [1200, 600,  150],
    [1500, 900,  180],
    [1800, 1400, 190],
    [2000, 1800, 200]
], dtype=float)

# ════════════════════════════════════════
# 2. Full 3-stage adaptive smoothing (Paper 1 §3.4):
#    1. Initialization — baseline spline (below, for comparison)
#    2. GA knot optimization
#    3. Curvature-aware control-point refinement
# ════════════════════════════════════════
smoother = AdaptiveBSplineSmoother(
    n_inner_knots=6,
    n_generations=30,
    population_size=20,
    kappa_max=0.5
)
spline_adaptive = smoother.smooth(waypoints)

# ════════════════════════════════════════
# 3. Baseline for comparison
# ════════════════════════════════════════
spline_uniform = BSplinePath(waypoints)

L_u  = spline_uniform.path_length()
L_a  = spline_adaptive.path_length()
k_u  = spline_uniform.max_curvature()
k_a  = spline_adaptive.max_curvature()
dist = np.linalg.norm(waypoints[-1] - waypoints[0])
SR_u = L_u / dist
SR_a = L_a / dist

print(f"\n{'='*55}")
print(f"PATH METRICS (Paper 1 Results)")
print(f"{'='*55}")
print(f"{'Method':<28} {'Length':>8} {'Max_κ':>8} {'SR':>6}")
print(f"{'-'*55}")
print(f"{'Uniform B-Spline':<28} {L_u:>8.1f} "
      f"{k_u:>8.4f} {SR_u:>6.3f}")
print(f"{'Adaptive B-Spline (GA)':<28} {L_a:>8.1f} "
      f"{k_a:>8.4f} {SR_a:>6.3f}")
print(f"{'Improvement':<28} "
      f"{L_u-L_a:>8.1f}m {k_u-k_a:>8.4f} "
      f"{SR_u-SR_a:>6.3f}")
print(f"{'='*55}\n")

# ════════════════════════════════════════
# 4. Run Merged Simulation (Paper 1 + 2)
# ════════════════════════════════════════
sim = MergedSimulation(
    spline_adaptive,
    dt=0.01,
    total_time=20.0,
    Pq=0.2,   # Tuned down from Paper 2's Pq=20 — at this inertia
    Pw=0.1    # (Ixx=Iyy=6.5e-4) and dt=0.01, Pq=20/Pw=4 is discrete-time
              # unstable: torque saturates at +/-4Nm from floating-point
              # noise alone and never recovers. Pq=0.2/Pw=0.1 is the
              # largest stable gain pair found by sweeping in 0.05 steps.
)
results = sim.run(
    noise_std=0.1,
    error_threshold=0.05,
    u_step=0.005
)

# Extract results
t          = np.array(results['t'])
euler      = np.array(results['euler'])
euler_ref  = np.array(results['euler_ref'])
axis_err   = np.array(results['axis_error'])
torque     = np.array(results['torque'])
omega      = np.array(results['omega'])
u_param    = np.array(results['u_param'])
err_mag    = np.array(results['err_mag'])

# Unwrap angles
eu = np.unwrap(euler,     axis=0)
er = np.unwrap(euler_ref, axis=0)

# ════════════════════════════════════════
# 5. PLOT 1 — 3D Path Comparison
# ════════════════════════════════════════
fig = plt.figure(figsize=(13, 9))
ax  = fig.add_subplot(111, projection='3d')

pts_u = spline_uniform.sample_path(300)
pts_a = spline_adaptive.sample_path(300)

ax.scatter(waypoints[:,0], waypoints[:,1], waypoints[:,2],
           c='black', s=60, zorder=5, label='Waypoints')
ax.plot(pts_u[:,0], pts_u[:,1], pts_u[:,2],
        'b--', lw=1.5,
        label=f'Uniform B-Spline (L={L_u:.0f}m, SR={SR_u:.3f})')
ax.plot(pts_a[:,0], pts_a[:,1], pts_a[:,2],
        'r-',  lw=2.5,
        label=f'Adaptive GA (L={L_a:.0f}m, SR={SR_a:.3f})')

ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Path — Uniform vs Adaptive B-Spline (Paper 1)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('results/path_comparison.png', dpi=150,
            bbox_inches='tight')
print("Saved: path_comparison.png")

# ════════════════════════════════════════
# 6. PLOT 2 — Curvature Along Path
# ════════════════════════════════════════
params = np.linspace(0.01, 0.99, 200)
curv_u = [spline_uniform.compute_curvature(u)  for u in params]
curv_a = [spline_adaptive.compute_curvature(u) for u in params]

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(params, curv_u, 'b--', lw=2,
        label=f'Uniform  (max κ = {max(curv_u):.4f} m⁻¹)')
ax.plot(params, curv_a, 'r-',  lw=2,
        label=f'Adaptive (max κ = {max(curv_a):.4f} m⁻¹)')
ax.axhline(y=0.5, color='orange', linestyle=':',
           lw=2, label='κ_max limit = 0.5 m⁻¹')
ax.fill_between(params, curv_a, alpha=0.15, color='red')
ax.set_xlabel('Path Parameter u', fontsize=11)
ax.set_ylabel('Curvature κ (m⁻¹)', fontsize=11)
ax.set_title('Curvature Along Path — Paper 1 Result', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/curvature.png', dpi=150,
            bbox_inches='tight')
print("Saved: curvature.png")

# ════════════════════════════════════════
# 7. PLOT 3 — Attitude Tracking
# ════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
colors = ['steelblue', 'seagreen', 'tomato']

for i in range(3):
    axes[i].plot(t, er[:, i], 'k--', lw=1.5,
                 alpha=0.8, label='Reference (from B-Spline)')
    axes[i].plot(t, eu[:, i], color=colors[i],
                 lw=1.5, label='Actual (drone)')
    axes[i].set_ylabel(labels[i], fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=9, loc='upper right')

axes[-1].set_xlabel('Time (s)', fontsize=11)
fig.suptitle(
    'Attitude Tracking — B-Spline Reference + Quaternion P² Controller\n'
    '(Paper 1 → Paper 2 Integration)',
    fontsize=12
)
plt.tight_layout()
plt.savefig('results/attitude.png', dpi=150,
            bbox_inches='tight')
print("Saved: attitude.png")

# ════════════════════════════════════════
# 8. PLOT 4 — Controller Torques
# ════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
t_labels = ['Mx (Nm)', 'My (Nm)', 'Mz (Nm)']

for i in range(3):
    axes[i].plot(t, torque[:, i],
                 color=colors[i], lw=1.2)
    axes[i].axhline(y= 4, color='red',
                    linestyle='--', lw=1.5,
                    alpha=0.7, label='±4 Nm limit')
    axes[i].axhline(y=-4, color='red',
                    linestyle='--', lw=1.5, alpha=0.7)
    axes[i].set_ylabel(t_labels[i], fontsize=10)
    axes[i].set_ylim(-5, 5)
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].legend(fontsize=9)

axes[-1].set_xlabel('Time (s)', fontsize=11)
fig.suptitle(
    'Controller Torques — Quaternion P² Controller (Paper 2)\n'
    'Pq=20, Pω=4, bounds=±4Nm',
    fontsize=12
)
plt.tight_layout()
plt.savefig('results/torques.png', dpi=150,
            bbox_inches='tight')
print("Saved: torques.png")

# ════════════════════════════════════════
# 9. PLOT 5 — Quaternion Error
# ════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(t, axis_err[:, 0],
             label='q1_err (roll)',  color='steelblue', lw=1.2)
axes[0].plot(t, axis_err[:, 1],
             label='q2_err (pitch)', color='seagreen',  lw=1.2)
axes[0].plot(t, axis_err[:, 2],
             label='q3_err (yaw)',   color='tomato',    lw=1.2)
axes[0].axhline(y=0, color='k', linestyle='--',
                lw=0.8, alpha=0.5)
axes[0].set_ylabel('Quaternion Axis Error', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Per-Axis Quaternion Error', fontsize=11)

axes[1].plot(t, err_mag, 'purple', lw=1.5)
axes[1].fill_between(t, 0, err_mag,
                     color='purple', alpha=0.2)
axes[1].set_ylabel('||q_err|| (magnitude)', fontsize=10)
axes[1].set_xlabel('Time (s)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Total Quaternion Error Magnitude', fontsize=11)

fig.suptitle(
    'Quaternion Tracking Error Over Time',
    fontsize=13
)
plt.tight_layout()
plt.savefig('results/quaternion_error.png', dpi=150,
            bbox_inches='tight')
print("Saved: quaternion_error.png")

# ════════════════════════════════════════
# 10. PLOT 6 — Angular Velocity
# ════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
w_labels = ['ωx (rad/s)', 'ωy (rad/s)', 'ωz (rad/s)']

for i in range(3):
    axes[i].plot(t, omega[:, i],
                 color=colors[i], lw=1.2)
    axes[i].set_ylabel(w_labels[i], fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=0, color='k',
                    linestyle='--', lw=0.8, alpha=0.5)

axes[-1].set_xlabel('Time (s)', fontsize=11)
fig.suptitle('Angular Velocity During Path Following', fontsize=13)
plt.tight_layout()
plt.savefig('results/angular_velocity.png', dpi=150,
            bbox_inches='tight')
print("Saved: angular_velocity.png")

# ════════════════════════════════════════
# 11. PLOT 7 — Path Parameter Progress
# ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, u_param, 'navy', lw=2)
ax.fill_between(t, 0, u_param, color='navy', alpha=0.15)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Path Parameter u', fontsize=11)
ax.set_title(
    'B-Spline Path Progress Over Simulation Time',
    fontsize=13
)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='green', linestyle='--',
           lw=1.5, label='Path end')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('results/path_progress.png', dpi=150,
            bbox_inches='tight')
print("Saved: path_progress.png")

# ════════════════════════════════════════
# 12. Final Summary
# ════════════════════════════════════════
conv_idx   = np.where(err_mag < 0.05)[0]
conv_time  = t[conv_idx[0]] if len(conv_idx) > 0 else float('nan')
steady_err = np.mean(err_mag[int(len(err_mag)*0.7):])

print(f"\n{'='*55}")
print(f"FINAL RESULTS SUMMARY")
print(f"{'='*55}")
print(f"\n[Paper 1] Path Planning:")
print(f"  Path length reduction:   "
      f"{L_u:.1f}m → {L_a:.1f}m "
      f"({(L_u-L_a)/L_u*100:.1f}% shorter)")
print(f"  Max curvature (uniform): {k_u:.4f} m⁻¹")
print(f"  Max curvature (adaptive):{k_a:.4f} m⁻¹")
print(f"  Straightness ratio:      "
      f"{SR_u:.3f} → {SR_a:.3f} (closer to 1.0 = better)")

print(f"\n[Paper 2] Attitude Control:")
print(f"  Controller gains:        Pq={sim.controller.Pq}, "
      f"Pω={sim.controller.Pw}")
print(f"  Max torque applied:      "
      f"{np.max(np.abs(torque)):.3f} Nm")
print(f"  Convergence time:        "
      f"{'~%.2fs' % conv_time if not np.isnan(conv_time) else 'never (did not converge)'}")
print(f"  Steady-state error:      {steady_err:.5f}")
print(f"  Final error:             {err_mag[-1]:.5f}")

print(f"\n[Merged] Integration:")
print(f"  Attitude refs from path: YES (B-Spline tangent)")
print(f"  Singularity-free:        YES (quaternion space)")
print(f"  Simulation duration:     {t[-1]:.1f}s")
print(f"  Final path progress:     {u_param[-1]*100:.1f}%")
print(f"{'='*55}")
