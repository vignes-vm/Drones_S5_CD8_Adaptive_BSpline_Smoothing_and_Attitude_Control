"""
Publication-ready plots for the project review — reruns the full
Paper 1 (GA-optimized B-Spline) + Paper 2 (Quaternion P^2 controller)
pipeline and renders polished figures to results/presentation/.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    plt.style.use('bmh')

TITLE_SIZE  = 14
LABEL_SIZE  = 12
TICK_SIZE   = 10
LEGEND_SIZE = 10
DPI         = 200

plt.rcParams.update({
    'axes.titlesize':  TITLE_SIZE,
    'axes.labelsize':  LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
})

from path_planning.bspline import BSplinePath
from path_planning.knot_ga import KnotGA
from integration.simulation_loop import MergedSimulation

OUT_DIR = 'results/presentation'
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════
# Run the pipeline (Paper 1 + Paper 2)
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

ga = KnotGA(
    waypoints,
    n_inner_knots=6,
    n_generations=30,
    population_size=20,
    kappa_max=0.5
)
best_knots = ga.run()

spline_uniform  = BSplinePath(waypoints)
spline_adaptive = BSplinePath(waypoints, best_knots)

L_u  = spline_uniform.path_length()
L_a  = spline_adaptive.path_length()
k_u  = spline_uniform.max_curvature()
k_a  = spline_adaptive.max_curvature()
dist = np.linalg.norm(waypoints[-1] - waypoints[0])
SR_u = L_u / dist
SR_a = L_a / dist

baseline_diffs  = np.diff(waypoints, axis=0)
L_baseline      = np.sum(np.linalg.norm(baseline_diffs, axis=1))
SR_baseline     = L_baseline / dist

sim = MergedSimulation(
    spline_adaptive,
    dt=0.01,
    total_time=20.0,
    Pq=0.2,
    Pw=0.1
)
results = sim.run(noise_std=0.1, error_threshold=0.05, u_step=0.005)

t         = np.array(results['t'])
euler     = np.array(results['euler'])
euler_ref = np.array(results['euler_ref'])
axis_err  = np.array(results['axis_error'])
torque    = np.array(results['torque'])
err_mag   = np.array(results['err_mag'])

eu = np.unwrap(euler,     axis=0)
er = np.unwrap(euler_ref, axis=0)

conv_idx  = np.where(err_mag < 0.05)[0]
conv_time = t[conv_idx[0]] if len(conv_idx) > 0 else float('nan')

print(f"Data ready — L_u={L_u:.1f} L_a={L_a:.1f} "
      f"conv_time={conv_time:.2f}s")

# ════════════════════════════════════════
# FIG 1 — 3D Path Comparison
# ════════════════════════════════════════
fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')

pts_u = spline_uniform.sample_path(300)
pts_a = spline_adaptive.sample_path(300)

ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
           c='black', s=50, zorder=5, label='Waypoints')
ax.scatter(*waypoints[0],  c='green', s=120, marker='^',
           zorder=6, label='Start')
ax.scatter(*waypoints[-1], c='red',   s=120, marker='s',
           zorder=6, label='End')
ax.plot(pts_u[:, 0], pts_u[:, 1], pts_u[:, 2],
        'b--', lw=1.5, label='Uniform B-Spline')
ax.plot(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2],
        'r-', lw=2.5, label='Adaptive B-Spline (GA)')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('B-Spline Path Comparison (Paper 1)')
ax.legend(loc='upper left', fontsize=LEGEND_SIZE)

ax.text2D(0.02, 0.90,
          f"Uniform: L={L_u:.0f}m, SR={SR_u:.3f}",
          transform=ax.transAxes, fontsize=10,
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text2D(0.02, 0.83,
          f"Adaptive: L={L_a:.0f}m, SR={SR_a:.3f}",
          transform=ax.transAxes, fontsize=10,
          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_path_3d.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/fig1_path_3d.png")

# ════════════════════════════════════════
# FIG 2 — Curvature Profile
# ════════════════════════════════════════
params = np.linspace(0.01, 0.99, 200)
curv_u = np.array([spline_uniform.compute_curvature(u)  for u in params])
curv_a = np.array([spline_adaptive.compute_curvature(u) for u in params])

max_u_idx = np.argmax(curv_u)
max_a_idx = np.argmax(curv_a)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(params, curv_u, 'b--', lw=2, label='Uniform B-Spline')
ax.plot(params, curv_a, 'r-',  lw=2, label='Adaptive B-Spline')
ax.fill_between(params, curv_a, color='red', alpha=0.15)

ax.plot(params[max_u_idx], curv_u[max_u_idx], 'bo', ms=9, zorder=5)
ax.plot(params[max_a_idx], curv_a[max_a_idx], 'ro', ms=9, zorder=5)

ax.axhline(y=0.5, color='orange', linestyle=':', lw=2,
           label='κ_max limit = 0.5 m⁻¹')

ax.annotate(
    f"Max κ (uniform) = {curv_u[max_u_idx]:.4f} m⁻¹\n"
    f"Max κ (adaptive) = {curv_a[max_a_idx]:.4f} m⁻¹",
    xy=(0.98, 0.95), xycoords='axes fraction',
    ha='right', va='top', fontsize=10,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
)

ax.set_xlabel('Path Parameter u')
ax.set_ylabel('Curvature κ (m⁻¹)')
ax.set_title('Curvature Profile Along Path')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_curvature.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/fig2_curvature.png")

# ════════════════════════════════════════
# FIG 3 — Attitude Tracking
# ════════════════════════════════════════
labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
colors = ['steelblue', 'seagreen', 'tomato']

fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
for i in range(3):
    axes[i].fill_between(
        t, er[:, i] - 0.05, er[:, i] + 0.05,
        color='green', alpha=0.15, label='Convergence zone (±0.05 rad)'
    )
    axes[i].plot(t, er[:, i], 'k--', lw=1.5, label='Reference')
    axes[i].plot(t, eu[:, i], color=colors[i], lw=1.5, label='Actual')
    axes[i].set_ylabel(labels[i])
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Time (s)')
fig.suptitle('Quaternion P² Controller — Attitude Tracking (Paper 2)',
             fontsize=TITLE_SIZE)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_attitude_tracking.png', dpi=DPI,
            bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/fig3_attitude_tracking.png")

# ════════════════════════════════════════
# FIG 4 — Error Convergence
# ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(t, 0, 0.05, color='green', alpha=0.15,
                 label='Good tracking zone (< 0.05)')
ax.plot(t, err_mag, color='purple', lw=1.5, label='||q_err||')

if not np.isnan(conv_time):
    ax.axvline(x=conv_time, color='black', linestyle='--', lw=1.5)
    ax.annotate(
        f"Converged at t={conv_time:.2f}s",
        xy=(conv_time, ax.get_ylim()[1] * 0.85),
        xytext=(conv_time + 0.5, ax.get_ylim()[1] * 0.85),
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        arrowprops=dict(arrowstyle='->')
    )

ax.set_xlabel('Time (s)')
ax.set_ylabel('||q_err|| (magnitude)')
ax.set_title('Quaternion Error Convergence')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_error_convergence.png', dpi=DPI,
            bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/fig4_error_convergence.png")

# ════════════════════════════════════════
# FIG 5 — Controller Torques
# ════════════════════════════════════════
t_labels = ['Mx (Nm)', 'My (Nm)', 'Mz (Nm)']

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for i in range(3):
    axes[i].fill_between(t, 0, torque[:, i],
                          color=colors[i], alpha=0.25)
    axes[i].plot(t, torque[:, i], color=colors[i], lw=1.2)
    axes[i].axhline(y=4,  color='red', linestyle='--', lw=1.5,
                     alpha=0.8, label='±4 Nm limit')
    axes[i].axhline(y=-4, color='red', linestyle='--', lw=1.5, alpha=0.8)
    axes[i].set_ylabel(t_labels[i])
    axes[i].set_ylim(-5, 5)
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].legend(loc='upper right')

axes[-1].set_xlabel('Time (s)')
fig.suptitle('Controller Torques — P² Quaternion Controller',
             fontsize=TITLE_SIZE)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_torques.png', dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/fig5_torques.png")

# ════════════════════════════════════════
# FIG 6 — Comparison Table
# ════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 3))
ax.axis('off')

col_labels = ['Method', 'Path Length', 'Max Curvature', 'SR']
rows = [
    ['Baseline Polyline', f'{L_baseline:.1f} m', 'N/A (polyline)', f'{SR_baseline:.3f}'],
    ['Uniform B-Spline',  f'{L_u:.1f} m',         f'{k_u:.4f} m⁻¹', f'{SR_u:.3f}'],
    ['Adaptive B-Spline', f'{L_a:.1f} m',         f'{k_a:.4f} m⁻¹', f'{SR_a:.3f}'],
]

table = ax.table(cellText=rows, colLabels=col_labels,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#404040')
    table[0, j].set_text_props(color='white', fontweight='bold')

adaptive_row = 3  # header is row 0
for j in range(len(col_labels)):
    table[adaptive_row, j].set_facecolor('#c6efce')

ax.set_title('Path Planning Comparison (Paper 1)', fontsize=TITLE_SIZE, pad=20)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_comparison_table.png', dpi=DPI,
            bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/fig6_comparison_table.png")

print("\nAll presentation figures generated.")
