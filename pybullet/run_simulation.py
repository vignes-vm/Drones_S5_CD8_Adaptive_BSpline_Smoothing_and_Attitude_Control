"""
Complete Integrated Demonstration: Paper 1 + Paper 2 in PyBullet Physics Engine.
Combines 3D Obstacle Environment + 3D A* Search + 3-Stage GA Adaptive B-Spline Smoothing
with 6-DOF PyBullet Quadrotor Flight, Position PD + Paper 2 Quaternion P^2 Control, and Motor Mixing.
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.path_smoother import AdaptiveBSplineSmoother
from pybullet.src.pybullet_env import PyBulletEnvironment
from pybullet.src.quadrotor_pybullet import PyBulletQuadrotor
from pybullet.src.trajectory_tracker import PyBulletTrajectoryTracker
from pybullet.src.quaternion_math import QuaternionMath
from pybullet.src.wind_model import WindDisturbanceModel


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Paper 1 + Paper 2 PyBullet Simulation")
    parser.add_argument('--gui', action='store_true', help="Run PyBullet in GUI visual 3D mode")
    parser.add_argument('--seed', type=int, default=42, help="Random obstacle placement seed")
    parser.add_argument('--duration', type=float, default=25.0, help="Simulation duration (seconds)")
    parser.add_argument('--wind', action='store_true', help="Enable wind & gust disturbances")
    return parser.parse_args()


def run_integrated_pybullet_simulation(gui=False, seed=42, duration=25.0, enable_wind=False):
    print("=" * 70, flush=True)
    print("INTEGRATED UAV NAVIGATION & ATTITUDE CONTROL (PYBULLET PHYSICS)", flush=True)
    print("Paper 1 (3D A* + Adaptive B-Spline) -> Paper 2 (Quaternion P^2 Control)", flush=True)
    print("=" * 70, flush=True)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 1: Environment Setup ──
    print("\n[Stage 1] Creating 3D Obstacle Environment...", flush=True)
    env_3d = Environment3D()
    env_3d.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)
    print(f"  Start: {env_3d.start} | Goal: {env_3d.goal}", flush=True)
    print(f"  Obstacles placed: {len(env_3d.obstacles)} (seed={seed})", flush=True)

    # ── Stage 2: Global Path Planning (A*) ──
    print("\n[Stage 2] Running 3D A* Search...", flush=True)
    astar = AStarPlanner3D(env_3d, grid_resolution=50)
    raw_wp = np.array(astar.plan(env_3d.start, env_3d.goal))
    baseline_wp = np.array(astar.smooth_waypoints(raw_wp, min_distance=100))
    raw_length  = astar.compute_path_length(baseline_wp)
    print(f"  Grid waypoints: {len(raw_wp)} -> Pruned waypoints: {len(baseline_wp)} | Path length: {raw_length:.1f} m", flush=True)

    # ── Stage 3: Adaptive B-Spline Path Smoothing ──
    print("\n[Stage 3] Applying 3-Stage GA Adaptive B-Spline Smoothing...", flush=True)
    smoother = AdaptiveBSplineSmoother(
        n_inner_knots=6, n_generations=20, population_size=15, kappa_max=0.5
    )
    spline = smoother.smooth(baseline_wp, verbose=False)
    spline_length = spline.path_length()
    spline_kappa  = spline.max_curvature()
    print(f"  Baseline Length: {raw_length:.1f} m -> Smoothed Length: {spline_length:.1f} m "
          f"({(raw_length - spline_length)/raw_length*100:.1f}% shorter)", flush=True)
    print(f"  Max Curvature:   {spline_kappa:.4f} m^-1", flush=True)

    # Sample B-Spline path points for GUI debug line rendering
    spline_samples = spline.sample_path(300)

    # ── Stage 4: PyBullet Physics Simulation Engine Setup ──
    print("\n[Stage 4] Initializing PyBullet Physics Engine...", flush=True)
    dt_sim = 0.005  # 200 Hz physics update rate
    pb_env = PyBulletEnvironment(gui=gui, environment_3d=env_3d, dt=dt_sim)
    pb_env.draw_path_lines(waypoints=baseline_wp, spline_pts=spline_samples)

    start_euler = [0.0, 0.0, 0.0]
    drone = PyBulletQuadrotor(
        physics_environment=pb_env,
        start_position=env_3d.start,
        start_orientation_euler=start_euler,
        enable_motor_model=True
    )

    tracker = PyBulletTrajectoryTracker(
        mass=drone.mass, g=drone.g, Pq=2.5, Pw=0.08, tau_limit=1.0,
        Kp_pos=[1.2, 1.2, 2.5], Kd_pos=[1.5, 1.5, 2.0]
    )

    wind = WindDisturbanceModel() if enable_wind else None
    if enable_wind:
        print("  Wind & Gust Disturbances: ENABLED", flush=True)

    # ── Stage 5: Physical Simulation Loop ──
    flight_speed = 25.0  # Cruise flight speed 25.0 m/s
    duration = (spline_length / flight_speed) + 5.0  # Full trajectory flight duration
    print(f"\n[Stage 5] Executing Full Trajectory Flight Simulation for {duration:.1f}s at {flight_speed} m/s...", flush=True)
    n_steps = int(round(duration / dt_sim))

    u_current = 0.001
    dt_u = (flight_speed / max(1.0, spline_length)) * dt_sim

    # Logging structures
    log = {
        't': [], 'pos': [], 'pos_des': [], 'vel': [], 'euler': [], 'euler_ref': [],
        'q_meas': [], 'q_ref': [], 'axis_err': [], 'thrust': [], 'torques': [],
        'motor_w': [], 'u_param': [], 'pos_err': [], 'att_err': []
    }

    prev_pos = drone.start_pos.copy()

    for i in range(n_steps):
        t_curr = i * dt_sim

        # Read state from PyBullet
        pos_meas, vel_meas, q_meas, omega_meas = drone.get_state()

        # Extract target trajectory point along B-Spline
        pos_des = spline.evaluate(u_current)
        tangent = spline.evaluate_derivative(u_current, 1)
        tangent_norm = np.linalg.norm(tangent)
        v_des = (tangent / tangent_norm * flight_speed) if tangent_norm > 1e-4 else np.zeros(3)

        # Calculate Guidance & Control commands
        thrust_cmd, torques_cmd, q_ref, axis_err, pos_err_mag = tracker.step_control(
            pos_des, v_des, pos_meas, vel_meas, q_meas, omega_meas, tangent=tangent
        )
        att_err_mag = np.linalg.norm(axis_err)

        # Path parameter progress advancement based on flight speed
        if u_current < 0.999:
            u_current = min(0.999, u_current + dt_u)

        # Disturbance calculation
        f_dist, tau_dist = (None, None)
        if enable_wind:
            f_dist, tau_dist = wind.step(t_curr, vel_meas)

        # Apply forces to PyBullet quadrotor
        drone.apply_control_inputs(thrust_cmd, torques_cmd, tau_dist=tau_dist, force_dist=f_dist)

        # Step PyBullet physics engine
        pb_env.step()

        # Render active flight path segment in GUI
        if i % 10 == 0:
            pb_env.draw_trajectory_segment(prev_pos, pos_meas)
            prev_pos = pos_meas.copy()

        # Log telemetry data
        log['t'].append(t_curr)
        log['pos'].append(pos_meas.copy())
        log['pos_des'].append(pos_des.copy())
        log['vel'].append(vel_meas.copy())
        log['euler'].append(QuaternionMath.to_euler(q_meas))
        log['euler_ref'].append(QuaternionMath.to_euler(q_ref))
        log['q_meas'].append(q_meas.copy())
        log['q_ref'].append(q_ref.copy())
        log['axis_err'].append(axis_err.copy())
        log['thrust'].append(drone.applied_thrust)
        log['torques'].append(drone.applied_torques.copy())
        log['motor_w'].append(drone.motor_speeds.copy())
        log['u_param'].append(u_current)
        log['pos_err'].append(pos_err_mag)
        log['att_err'].append(att_err_mag)

        if i % max(1, (n_steps // 10)) == 0:
            pct = 100 * i // n_steps
            print(f"  {pct:3d}% | t={t_curr:5.2f}s | u={u_current:.3f} | PosErr={pos_err_mag:6.1f}m | "
                  f"AttErr={att_err_mag:.4f} | MaxMotorW={np.max(drone.motor_speeds):.1f}rad/s", flush=True)

    print("\nFlight simulation complete. Disconnecting PyBullet...", flush=True)
    pb_env.close()

    # Convert logs to NumPy arrays
    for k in log:
        log[k] = np.array(log[k])

    # ── Stage 6: Plotting & Metrics Report ──
    print("\n[Stage 6] Generating Output Figures & Performance Metrics...", flush=True)

    # Plot 1: 3D Trajectory Comparison
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    env_3d.visualize(ax)
    ax.plot(baseline_wp[:, 0], baseline_wp[:, 1], baseline_wp[:, 2], 'k:', lw=1.5, label='A* Baseline Grid')
    ax.plot(spline_samples[:, 0], spline_samples[:, 1], spline_samples[:, 2], 'r--', lw=2.0, label='Adaptive B-Spline')
    ax.plot(log['pos'][:, 0], log['pos'][:, 1], log['pos'][:, 2], 'g-', lw=2.5, label='PyBullet Quadrotor Path')
    ax.set_title('PyBullet 3D Flight Simulation — Integrated Navigation & Control', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pybullet_3d_flight.png'), dpi=150)
    plt.close(fig)

    # Plot 2: Position & Attitude Tracking
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    coords = ['X (m)', 'Y (m)', 'Z (m)']
    for i in range(3):
        axes[i].plot(log['t'], log['pos_des'][:, i], 'r--', lw=1.5, label='Desired')
        axes[i].plot(log['t'], log['pos'][:, i], 'g-', lw=1.5, label='PyBullet Actual')
        axes[i].set_ylabel(coords[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper left', fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('PyBullet 3D Position Tracking Performance', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pybullet_position_tracking.png'), dpi=150)
    plt.close(fig)

    # Plot 3: Motor Speeds & Torques
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for m in range(4):
        axes[0].plot(log['t'], log['motor_w'][:, m], label=f'Motor {m+1}')
    axes[0].set_ylabel('Motor Speed (rad/s)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_title('Individual Rotor Angular Speeds')

    axes[1].plot(log['t'], log['att_err'], 'purple', lw=1.5, label='||q_err||')
    axes[1].set_ylabel('Quaternion Error')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_title('Attitude Tracking Error Convergence')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pybullet_actuators_and_error.png'), dpi=150)
    plt.close(fig)

    print(f"Saved plots to: {output_dir}")
    print("\n" + "=" * 70)
    print("PYBULLET INTEGRATED SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    args = parse_args()
    run_integrated_pybullet_simulation(
        gui=args.gui, seed=args.seed, duration=args.duration, enable_wind=args.wind
    )
