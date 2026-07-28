import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from path_planning.bspline import BSplinePath
from path_planning.knot_ga import KnotGA
from integration.simulation_loop import MergedSimulation

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
spline_uniform   = BSplinePath(waypoints)
spline_adaptive  = BSplinePath(waypoints, best_knots)

print(f"\nUniform   — Length: {spline_uniform.path_length():.1f}m, "
      f"Max κ: {spline_uniform.max_curvature():.4f}")
print(f"Adaptive  — Length: {spline_adaptive.path_length():.1f}m, "
      f"Max κ: {spline_adaptive.max_curvature():.4f}")

sim = MergedSimulation(spline_adaptive, dt=0.01, total_time=20.0)
results = sim.run(noise_std=0.01)

# Plot 1 — 3D Path Comparison
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

pts_uniform  = spline_uniform.sample_path(300)
pts_adaptive = spline_adaptive.sample_path(300)

ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2],
        'ko', markersize=6, label='Waypoints')
ax.plot(pts_uniform[:,0],  pts_uniform[:,1],  pts_uniform[:,2],
        'b--', linewidth=1.5, label='Uniform B-Spline')
ax.plot(pts_adaptive[:,0], pts_adaptive[:,1], pts_adaptive[:,2],
        'r-',  linewidth=2,   label='Adaptive B-Spline (GA)')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Path Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('results/path_comparison.png', dpi=150)
plt.show()

# Plot 2 — Euler Angles Over Time
t = results['t']
euler = np.array(results['euler'])
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
for i in range(3):
    axes[i].plot(t, euler[:, i])
    axes[i].set_ylabel(labels[i])
    axes[i].grid(True)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('Drone Attitude During Path Following')
plt.tight_layout()
plt.savefig('results/attitude.png', dpi=150)
plt.show()

# Plot 3 — Control Torques
torque = np.array(results['torque'])
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
torque_labels = ['Mx (Nm)', 'My (Nm)', 'Mz (Nm)']
for i in range(3):
    axes[i].plot(t, torque[:, i])
    axes[i].axhline(y=4,  color='r', linestyle='--', alpha=0.5)
    axes[i].axhline(y=-4, color='r', linestyle='--', alpha=0.5)
    axes[i].set_ylabel(torque_labels[i])
    axes[i].grid(True)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('Controller Torques (red = ±4Nm limit)')
plt.tight_layout()
plt.savefig('results/torques.png', dpi=150)
plt.show()

print("\nAll plots saved to results/")