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
spline_uniform  = BSplinePath(waypoints)
spline_adaptive = BSplinePath(waypoints, best_knots)

print(f"\nPath Metrics:")
print(f"  Uniform  — Length: {spline_uniform.path_length():.1f}m, "
      f"Max κ: {spline_uniform.max_curvature():.4f}")
print(f"  Adaptive — Length: {spline_adaptive.path_length():.1f}m, "
      f"Max κ: {spline_adaptive.max_curvature():.4f}")

sim = MergedSimulation(
    spline_adaptive,
    dt=0.01,
    total_time=20.0,
    Pq=0.05,      # tuned gains
    Pw=0.01
)
results = sim.run(noise_std=0.005, ref_smoothing=0.03)
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')

pts_u = spline_uniform.sample_path(300)
pts_a = spline_adaptive.sample_path(300)

ax.scatter(waypoints[:,0], waypoints[:,1], waypoints[:,2],
           c='black', s=50, zorder=5, label='Waypoints')
ax.plot(pts_u[:,0], pts_u[:,1], pts_u[:,2],
        'b--', lw=1.5, label=f'Uniform (L={spline_uniform.path_length():.0f}m)')
ax.plot(pts_a[:,0], pts_a[:,1], pts_a[:,2],
        'r-',  lw=2.5, label=f'Adaptive GA (L={spline_adaptive.path_length():.0f}m)')

ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
ax.set_title('3D Path Comparison — Uniform vs Adaptive B-Spline')
ax.legend()
plt.tight_layout()
plt.savefig('results/path_comparison.png', dpi=150)
plt.show()

params = np.linspace(0.01, 0.99, 200)
curv_u = [spline_uniform.compute_curvature(u)  for u in params]
curv_a = [spline_adaptive.compute_curvature(u) for u in params]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(params, curv_u, 'b--', lw=1.5, label='Uniform B-Spline')
ax.plot(params, curv_a, 'r-',  lw=2,   label='Adaptive B-Spline (GA)')
ax.axhline(y=0.5, color='orange', linestyle=':', lw=2, label='κ_max limit')
ax.set_xlabel('Path Parameter u')
ax.set_ylabel('Curvature κ (m⁻¹)')
ax.set_title('Curvature Along Path')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/curvature.png', dpi=150)
plt.show()

t     = results['t']
euler = np.array(results['euler'])
q_ref = np.array(results['q_ref'])

# Convert reference quaternions to Euler for comparison
from attitude_control.quaternion import Quaternion
ref_euler = np.array([
    Quaternion(*q).to_euler() for q in q_ref
])

# Unwrap to remove ±π jumps
eu = np.unwrap(euler, axis=0)
re = np.unwrap(ref_euler, axis=0)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
colors = ['steelblue', 'seagreen', 'tomato']

for i in range(3):
    axes[i].plot(t, re[:, i], 'k--', lw=1, alpha=0.6, label='Reference')
    axes[i].plot(t, eu[:, i], color=colors[i], lw=1.5, label='Actual')
    axes[i].set_ylabel(labels[i], fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=8, loc='upper right')

axes[-1].set_xlabel('Time (s)', fontsize=11)
fig.suptitle('Drone Attitude: Reference vs Actual', fontsize=13)
plt.tight_layout()
plt.savefig('results/attitude.png', dpi=150)
plt.show()

torque = np.array(results['torque'])
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
torque_labels = ['Mx (Nm)', 'My (Nm)', 'Mz (Nm)']
colors = ['steelblue', 'seagreen', 'tomato']

for i in range(3):
    axes[i].plot(t, torque[:, i], color=colors[i], lw=1)
    axes[i].axhline(y= 4, color='red', linestyle='--',
                    lw=1.5, alpha=0.7, label='±4 Nm limit')
    axes[i].axhline(y=-4, color='red', linestyle='--', lw=1.5, alpha=0.7)
    axes[i].set_ylabel(torque_labels[i], fontsize=10)
    axes[i].set_ylim(-5, 5)   # Force axis scale to show limits clearly
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=8, loc='upper right')

axes[-1].set_xlabel('Time (s)', fontsize=11)
fig.suptitle('Controller Torques (clamped to ±4 Nm)', fontsize=13)
plt.tight_layout()
plt.savefig('results/torques.png', dpi=150)
plt.show()

axis_err = np.array(results['axis_error'])
err_mag  = np.linalg.norm(axis_err, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(t, axis_err[:, 0], label='q1 err (roll)',  color='steelblue')
axes[0].plot(t, axis_err[:, 1], label='q2 err (pitch)', color='seagreen')
axes[0].plot(t, axis_err[:, 2], label='q3 err (yaw)',   color='tomato')
axes[0].set_ylabel('Axis Error')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(t, err_mag, 'purple', lw=1.5)
axes[1].set_ylabel('Error Magnitude ||q_err||')
axes[1].set_xlabel('Time (s)')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Quaternion Tracking Error Over Time', fontsize=13)
plt.tight_layout()
plt.savefig('results/quaternion_error.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("SIMULATION SUMMARY")
print("="*50)
print(f"Path Length (Uniform):   {spline_uniform.path_length():.1f} m")
print(f"Path Length (Adaptive):  {spline_adaptive.path_length():.1f} m")
print(f"Max Curvature (Uniform): {spline_uniform.max_curvature():.4f} m⁻¹")
print(f"Max Curvature (Adaptive):{spline_adaptive.max_curvature():.4f} m⁻¹")
print(f"\nController Performance:")
print(f"  Max torque applied:  {np.max(np.abs(torque)):.3f} Nm")
print(f"  Mean error magnitude:{np.mean(err_mag):.4f}")
print(f"  Final error magnitude:{err_mag[-1]:.4f}")
print(f"  Max error magnitude: {np.max(err_mag):.4f}")
print("="*50)