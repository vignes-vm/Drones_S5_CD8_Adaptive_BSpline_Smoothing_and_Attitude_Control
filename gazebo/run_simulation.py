"""
Master Integrated UAV Navigation & Flight Simulation in Gazebo Physics.
Integrates:
  Stage 1: 3D Obstacle Environment (Cylinders & Cuboids)
  Stage 2: 3D A* Discrete Global Path Planning
  Stage 3: 3-Stage Enhanced Adaptive B-Spline Smoothing (GA Knots + Curvature Refinement)
  Stage 4: Gazebo 6-DOF Quadrotor Dynamics with Motor Mixing & Wind Disturbances
  Stage 5: Dual-Loop Guidance (Outer Position PD + Inner Paper 2 Quaternion P^2 Control)
  Stage 6: Error-Gated Closed-Loop Progress Tracking and Telemetry Generation
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.path_smoother import AdaptiveBSplineSmoother
from integration.trajectory_converter import TrajectoryConverter

from gazebo.src.gazebo_env import GazeboEnvironment, generate_gazebo_world_sdf
from gazebo.src.quadrotor_gazebo import GazeboQuadrotor
from gazebo.src.trajectory_tracker import GazeboTrajectoryTracker
from gazebo.src.quaternion_math import QuaternionMath
from gazebo.src.wind_model import WindDisturbanceModel


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Gazebo UAV Simulation")
    parser.add_argument('--gui', action='store_true', help="Launch Gazebo with 3D GUI window")
    parser.add_argument('--seed', type=int, default=42, help="Obstacle random seed")
    parser.add_argument('--duration', type=float, default=25.0, help="Simulation duration (seconds)")
    parser.add_argument('--wind', action='store_true', help="Enable wind & gust disturbances")
    return parser.parse_args()


def run_integrated_gazebo_simulation(gui=False, seed=42, duration=25.0, enable_wind=False):
    print("=" * 70, flush=True)
    print("INTEGRATED UAV NAVIGATION & CONTROL IN GAZEBO PHYSICS", flush=True)
    print("Paper 1 (A* + Adaptive B-Spline) -> Paper 2 (Quaternion P^2 Control)", flush=True)
    print("=" * 70, flush=True)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 1: Environment Setup ──
    print("\n[Stage 1] Constructing 3D Obstacle Airspace...", flush=True)
    env_3d = Environment3D()
    env_3d.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)
    print(f"  Start: {env_3d.start} | Goal: {env_3d.goal}", flush=True)
    print(f"  Obstacles placed: {len(env_3d.obstacles)} (seed={seed})", flush=True)

    # Export Gazebo SDF World
    world_sdf = os.path.join(os.path.dirname(__file__), 'worlds', 'airspace_world.sdf')
    generate_gazebo_world_sdf(env_3d, world_sdf, world_name="airspace_world")
    print(f"  Exported Gazebo SDF World -> {world_sdf}", flush=True)

    # ── Stage 2: Global Path Planning (A*) ──
    print("\n[Stage 2] Running 3D A* Search on 26-connected Voxel Grid...", flush=True)
    astar = AStarPlanner3D(env_3d, grid_resolution=50)
    raw_wp = np.array(astar.plan(env_3d.start, env_3d.goal))
    baseline_wp = np.array(astar.smooth_waypoints(raw_wp, min_distance=100))
    raw_length = astar.compute_path_length(baseline_wp)
    print(f"  Raw grid cells: {len(raw_wp)} -> Pruned waypoints: {len(baseline_wp)} | Length: {raw_length:.1f}m", flush=True)

    # ── Stage 3: Adaptive B-Spline Path Smoothing ──
    print("\n[Stage 3] Applying 3-Stage GA Adaptive B-Spline Smoothing...", flush=True)
    smoother = AdaptiveBSplineSmoother(
        n_inner_knots=6, n_generations=20, population_size=15, kappa_max=0.5
    )
    spline = smoother.smooth(baseline_wp, verbose=False)
    spline_length = spline.path_length()
    spline_kappa = spline.max_curvature()
    print(f"  Smoothed Length: {spline_length:.1f} m ({(raw_length - spline_length)/raw_length*100:.1f}% reduction)", flush=True)
    print(f"  Max Curvature κ_max: {spline_kappa:.4f} m⁻¹ (limit=0.5)", flush=True)

    # Sample spline path
    spline_pts = spline.sample_path(300)

    # ── Stage 4: Gazebo Environment & Quadrotor Setup ──
    print("\n[Stage 4] Initializing Gazebo Physics Engine...", flush=True)
    dt_sim = 0.002  # 500 Hz physics rate
    gazebo_env = GazeboEnvironment(world_sdf_path=world_sdf, environment_3d=env_3d, gui=gui, dt=dt_sim)
    gazebo_env.launch()

    # Align initial drone pose with path start
    pos_start = spline.evaluate(0.001)
    tangent_start = spline.evaluate_derivative(0.001, 1)
    norm_t0 = np.linalg.norm(tangent_start)
    tan_dir0 = tangent_start / norm_t0 if norm_t0 > 1e-4 else np.array([1.0, 0.0, 0.0])
    yaw_start = np.arctan2(tan_dir0[1], tan_dir0[0])

    drone = GazeboQuadrotor(
        gazebo_environment=gazebo_env,
        start_position=pos_start,
        start_orientation_euler=[0.0, 0.0, yaw_start],
        enable_motor_model=True
    )
    drone.velocity = flight_speed * tan_dir0 if 'flight_speed' in locals() else 25.0 * tan_dir0

    tracker = GazeboTrajectoryTracker(
        mass=drone.mass, g=drone.g, Pq=2.5, Pw=0.08, tau_limit=1.0,
        Kp_pos=[1.2, 1.2, 2.5], Kd_pos=[1.5, 1.5, 2.0]
    )

    converter = TrajectoryConverter()
    wind_model = WindDisturbanceModel() if enable_wind else None
    if enable_wind:
        print("  Wind & Gust Disturbances: ENABLED", flush=True)

    # ── Stage 5: Closed-Loop Guidance & Flight Tracking ──
    flight_speed = 25.0  # Cruise speed m/s
    sim_duration = (spline_length / flight_speed) + 5.0 if duration is None or duration <= 30.0 else duration
    print(f"\n[Stage 5] Executing Full Trajectory Flight Simulation for {sim_duration:.1f}s at {flight_speed} m/s (500 Hz)...", flush=True)
    total_steps = int(sim_duration / dt_sim)
    u_current = 0.001
    dt_u = (flight_speed / max(1.0, spline_length)) * dt_sim

    # Telemetry logging arrays
    t_log = np.zeros(total_steps)
    pos_actual_log = np.zeros((total_steps, 3))
    pos_target_log = np.zeros((total_steps, 3))
    euler_actual_log = np.zeros((total_steps, 3))
    euler_target_log = np.zeros((total_steps, 3))
    torque_log = np.zeros((total_steps, 3))
    thrust_log = np.zeros(total_steps)
    u_log = np.zeros(total_steps)
    err_pos_log = np.zeros(total_steps)
    err_att_log = np.zeros(total_steps)
    motor_thrusts_log = np.zeros((total_steps, 4))

    for step in range(total_steps):
        t = step * dt_sim

        # 1. Target state from spline
        pos_target = spline.evaluate(u_current)
        tangent = spline.evaluate_derivative(u_current, 1)
        norm_tan = np.linalg.norm(tangent)
        tangent_dir = tangent / norm_tan if norm_tan > 1e-4 else np.array([1.0, 0.0, 0.0])

        vel_target = flight_speed * tangent_dir
        yaw_target = np.arctan2(tangent_dir[1], tangent_dir[0])

        # 2. Wind disturbances
        if wind_model:
            w_force, w_torque = wind_model.compute_aerodynamic_wrench(drone.velocity, drone.omega, t)
        else:
            w_force, w_torque = np.zeros(3), np.zeros(3)

        # 3. Dual-loop Guidance & Control
        thrust_cmd, torques_cmd, q_des, axis_err = tracker.compute_full_control(
            pos_target=pos_target,
            pos_current=drone.position,
            q_current=drone.q,
            omega_current=drone.omega,
            vel_target=vel_target,
            vel_current=drone.velocity,
            tangent=tangent,
            yaw_target=yaw_target
        )

        # 4. Step Gazebo 6-DOF Dynamics
        drone.step(thrust_cmd, torques_cmd, wind_force_world=w_force, wind_torque_body=w_torque)

        # 5. Path progression advancing along spline
        pos_err = np.linalg.norm(pos_target - drone.position)
        att_err = np.linalg.norm(axis_err)

        if u_current < 0.999:
            u_current = min(u_current + dt_u, 0.999)

        # 6. Record Telemetry
        t_log[step] = t
        pos_actual_log[step] = drone.position.copy()
        pos_target_log[step] = pos_target.copy()
        euler_actual_log[step] = drone.get_euler()
        euler_target_log[step] = QuaternionMath.to_euler(q_des)
        torque_log[step] = drone.applied_torques.copy()
        thrust_log[step] = drone.applied_thrust
        u_log[step] = u_current
        err_pos_log[step] = pos_err
        err_att_log[step] = att_err
        motor_thrusts_log[step] = drone.motor_thrusts.copy()

        if step % (total_steps // 10) == 0:
            pct = 100 * step // total_steps
            print(f"  {pct:3d}% | t={t:5.1f}s | u={u_current:.3f} | "
                  f"PosErr={pos_err:5.1f}m | AttErr={att_err:.4f} | "
                  f"Pos=[{drone.position[0]:.0f}, {drone.position[1]:.0f}, {drone.position[2]:.0f}]m", flush=True)

    print(f"\n[Simulation Complete] Final Path Progress: {u_current*100:.1f}% in {duration}s", flush=True)
    gazebo_env.close()

    # ── Stage 6: Generate Visualization Artifacts ──
    print("\n[Stage 6] Rendering Telemetry & Comparison Plots...", flush=True)

    # Plot 1: 3D Flight Path & Obstacles
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    env_3d.visualize(ax)
    ax.plot(baseline_wp[:, 0], baseline_wp[:, 1], baseline_wp[:, 2], 'k:', lw=1.2, label='A* Waypoints')
    ax.plot(spline_pts[:, 0], spline_pts[:, 1], spline_pts[:, 2], 'b--', lw=1.5, label='B-Spline Reference')
    ax.plot(pos_actual_log[:, 0], pos_actual_log[:, 1], pos_actual_log[:, 2], 'r-', lw=2.5, label='Gazebo Quadrotor Trajectory')
    ax.scatter(env_3d.start[0], env_3d.start[1], env_3d.start[2], c='green', s=100, label='Start')
    ax.scatter(env_3d.goal[0], env_3d.goal[1], env_3d.goal[2], c='gold', s=100, label='Goal')
    ax.set_title('Gazebo 3D UAV Flight in Cluttered Airspace', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    p1 = os.path.join(output_dir, 'gazebo_3d_flight.png')
    plt.tight_layout()
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {p1}", flush=True)

    # Plot 2: Position Tracking vs Time
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    coords = ['X (m)', 'Y (m)', 'Z (m)']
    for i in range(3):
        axes[i].plot(t_log, pos_target_log[:, i], 'k--', lw=1.5, label='Reference')
        axes[i].plot(t_log, pos_actual_log[:, i], 'r-', lw=1.5, label='Gazebo Actual')
        axes[i].set_ylabel(coords[i], fontsize=10)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('Gazebo 3D Position Tracking Telemetry', fontsize=12, fontweight='bold')
    p2 = os.path.join(output_dir, 'gazebo_position_tracking.png')
    plt.tight_layout()
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {p2}", flush=True)

    # Plot 3: Attitude & Control Torques
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    att_names = ['Roll φ (deg)', 'Pitch θ (deg)', 'Yaw ψ (deg)']
    torque_names = ['Mx (Roll Torque, Nm)', 'My (Pitch Torque, Nm)', 'Mz (Yaw Torque, Nm)']

    for i in range(3):
        axes[0, i].plot(t_log, np.degrees(euler_target_log[:, i]), 'k--', label='Ref')
        axes[0, i].plot(t_log, np.degrees(euler_actual_log[:, i]), 'b-', label='Actual')
        axes[0, i].set_ylabel(att_names[i])
        axes[0, i].grid(True, alpha=0.3)
        if i == 0:
            axes[0, i].legend()

        axes[1, i].plot(t_log, torque_log[:, i], 'r-', lw=1.2)
        axes[1, i].axhline(4.0, color='gray', linestyle=':')
        axes[1, i].axhline(-4.0, color='gray', linestyle=':')
        axes[1, i].set_ylabel(torque_names[i])
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].grid(True, alpha=0.3)

    fig.suptitle('Gazebo Attitude Tracking & Quaternion P² Control Torques', fontsize=12, fontweight='bold')
    p3 = os.path.join(output_dir, 'gazebo_attitude_and_torques.png')
    plt.tight_layout()
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {p3}", flush=True)

    # Plot 4: Path Parameter Progress & Tracking Errors
    fig, (ax_u, ax_err) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax_u.plot(t_log, u_log, 'navy', lw=2.0, label='Path Progress u')
    ax_u.axhline(1.0, color='g', linestyle='--', label='Path End (u=1.0)')
    ax_u.set_ylabel('Parameter u ∈ [0, 1]', fontsize=10)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(loc='lower right')
    ax_u.set_title('Path Progression & Closed-Loop Tracking Errors in Gazebo', fontsize=12, fontweight='bold')

    ax_err.plot(t_log, err_pos_log, 'crimson', lw=1.5, label='Position Error (m)')
    ax_err.set_ylabel('Position Error (m)', fontsize=10)
    ax_err.set_xlabel('Time (s)', fontsize=11)
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc='upper right')

    p4 = os.path.join(output_dir, 'gazebo_path_progress.png')
    plt.tight_layout()
    plt.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {p4}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL GAZEBO SIMULATION ARTIFACTS GENERATED SUCCESSFULLY", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    args = parse_args()
    run_integrated_gazebo_simulation(
        gui=args.gui,
        seed=args.seed,
        duration=args.duration,
        enable_wind=args.wind
    )
