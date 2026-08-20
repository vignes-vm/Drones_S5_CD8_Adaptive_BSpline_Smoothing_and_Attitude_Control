"""
Simulation Suite for Paper 2:
"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning" (Mykola Nikolaiev & Mykhailo Novotarskyi, IJISA 2025)

Executes:
1. Benchmark 3D obstacle environment setup (2000m x 2000m x 500m & scaled MuJoCo scene).
2. Four Path Planning & Smoothing Methods:
   - Method 1: Baseline Polyline (No Smoothing)
   - Method 2: Uniform B-Spline
   - Method 3: Classical B-Spline (Fixed Knots - Chord-Length)
   - Method 4: Proposed Enhanced Adaptive B-Spline (GA Knot Optimization + Curvature Refinement)
3. Performance Metrics Table calculation matching Table 2 in Paper 2.
4. MuJoCo Physics Flight Simulation with smooth quintic time-scaling for 100% precision flight.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.bspline_planner import (
    fit_baseline_polyline,
    fit_uniform_bspline,
    fit_classical_bspline,
    fit_enhanced_adaptive_bspline
)
from src.quadrotor_mujoco import QuadrotorMuJoCoSim
from src.trajectory_tracker import CascadedQuadrotorTracker

def quintic_time_scaling(t: float, T: float) -> tuple:
    """
    Computes quintic s(t), s_dot(t), s_ddot(t) for t in [0, T].
    Guarantees v(0)=0, a(0)=0 at start and v(T)=0, a(T)=0 at goal.
    """
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
    ds = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / T
    dds = (60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)) / (T**2)
    return s, ds, dds

def run_paper2_simulations(gui: bool = False, output_dir: str = "output_plots"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n=================================================================")
    print("RUNNING PAPER 2 SIMULATION IN MUJOCO: ADAPTIVE B-SPLINE PATH PLANNING")
    print(f"3D GUI Visualizer Window: {'ENABLED' if gui else 'DISABLED (Use --gui to open)'}")
    print("=================================================================")

    start_pt = np.array([0.0, 0.0, 20.0])
    goal_pt = np.array([2000.0, 1800.0, 200.0])

    raw_waypoints = np.array([
        [0.0, 0.0, 20.0],
        [400.0, 250.0, 120.0],
        [750.0, 600.0, 280.0],
        [1100.0, 950.0, 350.0],
        [1450.0, 1300.0, 180.0],
        [1750.0, 1600.0, 240.0],
        [2000.0, 1800.0, 200.0]
    ], dtype=np.float64)

    obstacles = [
        {'type': 'box', 'bounds': [500, 700, 400, 600, 0, 300], 'color': 'g'},
        {'type': 'cylinder', 'center': [1100, 800], 'radius': 120, 'height': 400, 'color': 'r'},
        {'type': 'box', 'bounds': [1400, 1600, 1100, 1350, 0, 280], 'color': 'g'},
        {'type': 'cylinder', 'center': [800, 1400], 'radius': 150, 'height': 350, 'color': 'r'}
    ]

    print("\nExecuting Path-Smoothing Methods...")
    print(" -> Method 1: Baseline Polyline...")
    pos_m1, curv_m1, metrics_m1 = fit_baseline_polyline(raw_waypoints)

    print(" -> Method 2: Uniform B-Spline...")
    pos_m2, curv_m2, metrics_m2, ev_m2 = fit_uniform_bspline(raw_waypoints)

    print(" -> Method 3: Classical B-Spline (Fixed Knots)...")
    pos_m3, curv_m3, metrics_m3, ev_m3 = fit_classical_bspline(raw_waypoints)

    print(" -> Method 4: Proposed Enhanced Adaptive B-Spline (GA Knot Opt + Curvature Refinement)...")
    pos_m4, curv_m4, metrics_m4, ev_m4 = fit_enhanced_adaptive_bspline(raw_waypoints)

    # Print Table 2
    print("\n" + "="*80)
    print("TABLE 2: Performance Comparison of Four Path-Smoothing Methods (Paper 2 Replication)")
    print("="*80)
    header = f"{'Method':<35} | {'Path Length (m)':<15} | {'Max Curv (m^-1)':<15} | {'Comp Time (s)':<13} | {'Straightness Ratio':<18}"
    print(header)
    print("-" * len(header))

    methods_data = [
        ("Baseline Polyline (No Smoothing)", metrics_m1),
        ("Uniform B-Spline", metrics_m2),
        ("Classical B-Spline (Fixed Knots)", metrics_m3),
        ("Proposed Enhanced Adaptive B-Spline", metrics_m4)
    ]

    for name, m in methods_data:
        print(f"{name:<35} | {m['path_length']:<15.1f} | {m['max_curvature']:<15.3f} | {m['comp_time']:<13.3f} | {m['straightness_ratio']:<18.3f}")
    print("="*80)

    summary_path = os.path.join(output_dir, 'paper2_metrics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TABLE 2: Performance Comparison of Four Path-Smoothing Methods (Paper 2 Replication)\n")
        f.write("="*80 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for name, m in methods_data:
            f.write(f"{name:<35} | {m['path_length']:<15.1f} | {m['max_curvature']:<15.3f} | {m['comp_time']:<13.3f} | {m['straightness_ratio']:<18.3f}\n")
        f.write("="*80 + "\n")
    print(f" Summary saved to: {summary_path}")

    # Plot 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for obs in obstacles:
        if obs['type'] == 'box':
            b = obs['bounds']
            xx = [b[0], b[1], b[1], b[0], b[0], b[0], b[1], b[1], b[0], b[0], b[1], b[1]]
            yy = [b[2], b[2], b[3], b[3], b[2], b[2], b[2], b[3], b[3], b[2], b[3], b[3]]
            zz = [b[4], b[4], b[4], b[4], b[4], b[5], b[5], b[5], b[5], b[5], b[4], b[5]]
            ax.plot(xx, yy, zz, color='green', alpha=0.6, linewidth=1.5)
        elif obs['type'] == 'cylinder':
            cx, cy = obs['center']
            r = obs['radius']
            h = obs['height']
            z_vals = np.linspace(0, h, 20)
            theta_vals = np.linspace(0, 2*np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta_vals, z_vals)
            x_grid = cx + r * np.cos(theta_grid)
            y_grid = cy + r * np.sin(theta_grid)
            ax.plot_wireframe(x_grid, y_grid, z_grid, color='red', alpha=0.3, rstride=5, cstride=5)

    ax.scatter([start_pt[0]], [start_pt[1]], [start_pt[2]], color='blue', s=80, marker='o', label='Start S=(0,0,20)')
    ax.scatter([goal_pt[0]], [goal_pt[1]], [goal_pt[2]], color='red', s=100, marker='^', label='Target T=(2000,1800,200)')
    ax.scatter(raw_waypoints[:, 0], raw_waypoints[:, 1], raw_waypoints[:, 2], color='black', s=40, zorder=5, label='Raw Waypoints')

    ax.plot(pos_m1[:, 0], pos_m1[:, 1], pos_m1[:, 2], 'k--', linewidth=1.2, label='Baseline Polyline')
    ax.plot(pos_m2[:, 0], pos_m2[:, 1], pos_m2[:, 2], 'm-', linewidth=1.8, label='Uniform B-Spline')
    ax.plot(pos_m3[:, 0], pos_m3[:, 1], pos_m3[:, 2], 'g-', linewidth=2.0, label='Classical B-Spline (Fixed Knots)')
    ax.plot(pos_m4[:, 0], pos_m4[:, 1], pos_m4[:, 2], 'r-', linewidth=2.5, label='Proposed Enhanced Adaptive B-Spline')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_xlim([0, 2000])
    ax.set_ylim([0, 2000])
    ax.set_zlim([0, 500])
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Fig. 4: 3D Airspace Environment & Trajectory Smoothing Comparison (Paper 2 Replication)', fontsize=11)
    plt.tight_layout()

    fig4_path = os.path.join(output_dir, 'paper2_3d_trajectories.png')
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f" Saved: {fig4_path}")

    # Plot Curvature
    plt.figure(figsize=(9, 5))
    u_domain = np.linspace(0, 1, len(curv_m4))
    plt.plot(u_domain, curv_m1, 'k--', label='Baseline Polyline', alpha=0.7)
    plt.plot(u_domain, curv_m2, 'm-', label='Uniform B-Spline', alpha=0.8)
    plt.plot(u_domain, curv_m3, 'g-', label='Classical B-Spline (Fixed Knots)', alpha=0.8)
    plt.plot(u_domain, curv_m4, 'r-', label='Proposed Enhanced Adaptive B-Spline', linewidth=2.0)
    plt.xlabel('Normalized Trajectory Parameter u ∈ [0, 1]')
    plt.ylabel('Curvature κ(u) [m^-1]')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Curvature Comparison Across Path-Smoothing Methods (Paper 2 Replication)', fontsize=12)
    plt.tight_layout()

    curv_plot_path = os.path.join(output_dir, 'paper2_curvature_comparison.png')
    plt.savefig(curv_plot_path, dpi=200)
    plt.close()
    print(f" Saved: {curv_plot_path}")

    # -------------------------------------------------------------------------
    # MUJOCO PHYSICS FLIGHT SIMULATION
    # -------------------------------------------------------------------------
    print("\nExecuting MuJoCo Physics Flight Simulation along B-Spline Trajectory...")

    sim = QuadrotorMuJoCoSim(xml_path="models/airspace_scene.xml", gui=gui)
    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81)

    scale_factor = 0.01
    scaled_positions = pos_m4 * scale_factor

    sim.reset(pos=scaled_positions[0], quat=np.array([1.0, 0.0, 0.0, 0.0]))
    tracker.reset()

    dt = sim.dt
    total_flight_time = 15.0
    total_steps = int(total_flight_time / dt)

    flight_pos_arr = []
    flight_ref_arr = []
    tracking_errors = []

    for step_idx in range(total_steps):
        t = sim.get_state()['time']

        # Quintic time scaling s(t) in [0, 1] with v(0)=0 and v(T)=0
        s, ds, dds = quintic_time_scaling(t, total_flight_time)

        p_ref = ev_m4.eval_pos(s) * scale_factor
        v_ref = ev_m4.eval_vel(s) * ds * scale_factor
        a_ref = (ev_m4.eval_acc(s) * (ds**2) + ev_m4.eval_vel(s) * dds) * scale_factor

        state = sim.get_state()
        tau, F_thrust = tracker.compute_control(p_ref, v_ref, a_ref, state)
        sim.step(tau=tau, thrust_force=F_thrust, real_time=gui)

        err_mujoco = np.linalg.norm(state['pos'] - p_ref)
        flight_pos_arr.append(state['pos'])
        flight_ref_arr.append(p_ref)
        tracking_errors.append(err_mujoco)

    if gui:
        sim.keep_alive_until_closed()

    flight_pos_arr = np.array(flight_pos_arr)
    flight_ref_arr = np.array(flight_ref_arr)
    tracking_errors = np.array(tracking_errors)
    mean_tracking_err_mujoco = float(np.mean(tracking_errors))
    final_arrival_err_mujoco = float(np.linalg.norm(flight_pos_arr[-1] - flight_ref_arr[-1]))

    print(f"\nMuJoCo Flight Simulation Completed!")
    print(f" -> Mean Trajectory Tracking Error in Arena: {mean_tracking_err_mujoco:.4f} m ({mean_tracking_err_mujoco/scale_factor:.2f} m full scale)")
    print(f" -> Final Goal Arrival Error in Arena:      {final_arrival_err_mujoco:.4f} m ({final_arrival_err_mujoco/scale_factor:.2f} m full scale)")

    # Plot Flight Tracking
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axis_names = ['X [m]', 'Y [m]', 'Z [m]']
    t_flight = np.linspace(0, total_flight_time, len(flight_pos_arr))

    for k in range(3):
        axes[k].plot(t_flight, flight_ref_arr[:, k], 'r--', label='B-Spline Target' if k == 0 else "")
        axes[k].plot(t_flight, flight_pos_arr[:, k], 'b-', label='MuJoCo Quadrotor' if k == 0 else "")
        axes[k].set_ylabel(axis_names[k])
        axes[k].grid(True)

    axes[2].set_xlabel('Flight Time [s]')
    axes[0].legend(loc='upper left')
    fig.suptitle('MuJoCo Physics Quadrotor Flight Tracking along Enhanced B-Spline Path', fontsize=12)
    plt.tight_layout()

    flight_plot_path = os.path.join(output_dir, 'paper2_mujoco_flight_tracking.png')
    plt.savefig(flight_plot_path, dpi=200)
    plt.close()
    print(f" Saved: {flight_plot_path}")

    print("\nPaper 2 Simulation completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Launch live 3D MuJoCo GUI window')
    args = parser.parse_args()
    run_paper2_simulations(gui=args.gui)
