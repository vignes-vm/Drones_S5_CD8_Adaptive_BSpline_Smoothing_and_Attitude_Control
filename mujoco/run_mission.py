"""
End-to-End UAV Mission Runner:
Combines Paper 2 Adaptive B-Spline Pre-Flight Path Planning with Paper 1 Quaternion Attitude Control
under Real-Time Atmospheric Wind & Gust Disturbances in MuJoCo.

Supports arbitrary user-selected Start/Goal coordinates and live MuJoCo 3D GUI rendering window (--gui).
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.obstacle_planner import generate_obstacle_free_waypoints
from src.bspline_planner import fit_enhanced_adaptive_bspline
from src.wind_model import WindDisturbanceModel
from src.quadrotor_mujoco import QuadrotorMuJoCoSim
from src.trajectory_tracker import CascadedQuadrotorTracker

def quintic_time_scaling(t: float, T: float) -> tuple:
    """Computes quintic s(t), s_dot(t), s_ddot(t) for smooth start and goal arrival."""
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
    ds = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / T
    dds = (60.0 * tau - 180.0 * (tau**2) + 120.0 * (tau**3)) / (T**2)
    return s, ds, dds

def execute_mission(start_pt: np.ndarray, goal_pt: np.ndarray, wind_speed: float = 3.5, gui: bool = False, output_dir: str = "output_plots"):
    os.makedirs(output_dir, exist_ok=True)

    print("========================================================================")
    print("        END-TO-END UAV MISSION: ADAPTIVE B-SPLINE & QUATERNION CONTROL ")
    print("========================================================================")
    print(f" Start Position: {start_pt} (m)")
    print(f" Goal Position:  {goal_pt} (m)")
    print(f" Crosswind Speed:{wind_speed} m/s + Dryden Turbulence Gusts")
    print(f" Interactive 3D GUI Window: {'ENABLED' if gui else 'DISABLED (Use --gui to open)'}")
    print("========================================================================\n")

    # -------------------------------------------------------------------------
    # STEP 1: PRE-FLIGHT PATH PLANNING & ENHANCED ADAPTIVE B-SPLINE SMOOTHING
    # -------------------------------------------------------------------------
    print("[Phase 1] Pre-Flight Planning: Generating obstacle-free waypoints...")
    raw_waypoints = generate_obstacle_free_waypoints(start_pt, goal_pt)
    print(f" -> Generated {len(raw_waypoints)} raw 3D collision-free waypoints.")

    print("\n[Phase 1] Pre-Flight Planning: Running Paper 2 Enhanced Adaptive B-Spline Optimization...")
    pos_bspline, curv_bspline, metrics, evaluator = fit_enhanced_adaptive_bspline(raw_waypoints, num_samples=600)

    print(f" -> Optimized Path Length:   {metrics['path_length']:.2f} m")
    print(f" -> Maximum Path Curvature:  {metrics['max_curvature']:.4f} m^-1")
    print(f" -> Straightness Ratio:      {metrics['straightness_ratio']:.3f}")
    print(f" -> Planning Time:           {metrics['comp_time']:.3f} s")

    # -------------------------------------------------------------------------
    # STEP 2: IN-FLIGHT MUJOCO SIMULATION WITH WIND DISTURBANCES & QUATERNION CONTROL
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Flight Execution: Launching MuJoCo Physics Engine...")

    sim = QuadrotorMuJoCoSim(xml_path="models/airspace_scene.xml", gui=gui)
    wind_engine = WindDisturbanceModel(wind_speed=wind_speed, wind_direction_deg=45.0, gust_intensity=0.5, dt=sim.dt)
    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81, dt=sim.dt)

    scale_factor = 0.01  # Map 2000m scale airspace to 20m visual arena in MuJoCo
    scaled_positions = pos_bspline * scale_factor

    # Reset quadrotor at scaled start point
    sim.reset(pos=scaled_positions[0], quat=np.array([1.0, 0.0, 0.0, 0.0]))
    tracker.reset()

    dt = sim.dt
    total_flight_time = 15.0
    total_steps = int(total_flight_time / dt)

    t_arr = []
    pos_actual_arr = []
    pos_target_arr = []
    wind_forces_arr = []
    tracking_errors = []

    print(" -> Flying quadrotor under active wind gusts and Paper 1 Quaternion Attitude balancing...")

    for step in range(total_steps):
        if not sim.is_viewer_alive():
            print("\n[Notice] User closed 3D viewer window mid-flight.")
            break

        state = sim.get_state()
        t = state['time']

        # Quintic time scaling s(t) in [0, 1] with v(0)=0 and v(T)=0
        s, ds, dds = quintic_time_scaling(t, total_flight_time)

        p_ref = evaluator.eval_pos(s) * scale_factor
        v_ref = evaluator.eval_vel(s) * ds * scale_factor
        a_ref = (evaluator.eval_acc(s) * (ds**2) + evaluator.eval_vel(s) * dds) * scale_factor

        # Compute wind force & torque disturbances acting on aircraft
        F_wind, tau_wind = wind_engine.compute_disturbance(quad_vel=state['vel'])

        # Compute cascaded control action (PID position + Paper 1 Quaternion Attitude controller)
        tau_control, F_thrust = tracker.compute_control(p_ref, v_ref, a_ref, state)

        # Apply forces and step simulation
        sim.data.qfrc_applied[0:3] = F_wind  # Apply wind drag force in world frame
        sim.step(tau=tau_control + tau_wind, thrust_force=F_thrust, real_time=gui)

        # Record metrics in MuJoCo arena scale (meters)
        err_mujoco = np.linalg.norm(state['pos'] - p_ref)

        t_arr.append(t)
        pos_actual_arr.append(state['pos'] / scale_factor)
        pos_target_arr.append(p_ref / scale_factor)
        wind_forces_arr.append(F_wind)
        tracking_errors.append(err_mujoco)

    t_arr = np.array(t_arr) if len(t_arr) > 0 else np.array([0.0])
    pos_actual_arr = np.array(pos_actual_arr) if len(pos_actual_arr) > 0 else np.array([[0,0,0]])
    pos_target_arr = np.array(pos_target_arr) if len(pos_target_arr) > 0 else np.array([[0,0,0]])
    wind_forces_arr = np.array(wind_forces_arr) if len(wind_forces_arr) > 0 else np.array([[0,0,0]])
    tracking_errors = np.array(tracking_errors) if len(tracking_errors) > 0 else np.array([0.0])

    mean_err_mujoco = float(np.mean(tracking_errors))
    max_err_mujoco = float(np.max(tracking_errors))
    final_arrival_err_mujoco = float(np.linalg.norm(pos_actual_arr[-1] * scale_factor - pos_target_arr[-1] * scale_factor))

    print(f"\n[Phase 2] Mission Completed Successfully!")
    print(f" -> Mean Trajectory Tracking Error in MuJoCo Arena: {mean_err_mujoco:.4f} m (Full scale: {mean_err_mujoco/scale_factor:.2f} m)")
    print(f" -> Max Trajectory Tracking Error in MuJoCo Arena:  {max_err_mujoco:.4f} m (Full scale: {max_err_mujoco/scale_factor:.2f} m)")
    print(f" -> Final Goal Arrival Position Error:             {final_arrival_err_mujoco:.4f} m ({final_arrival_err_mujoco/scale_factor:.2f} m full scale)")

    # -------------------------------------------------------------------------
    # STEP 3: GENERATE MISSION PLOTS
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([start_pt[0]], [start_pt[1]], [start_pt[2]], color='blue', s=90, marker='o', label=f'Start ({start_pt[0]},{start_pt[1]},{start_pt[2]})')
    ax.scatter([goal_pt[0]], [goal_pt[1]], [goal_pt[2]], color='red', s=110, marker='^', label=f'Goal ({goal_pt[0]},{goal_pt[1]},{goal_pt[2]})')
    ax.plot(pos_bspline[:, 0], pos_bspline[:, 1], pos_bspline[:, 2], 'r--', linewidth=2.0, label='Pre-Flight B-Spline Plan (Paper 2)')
    ax.plot(pos_actual_arr[:, 0], pos_actual_arr[:, 1], pos_actual_arr[:, 2], 'b-', linewidth=2.2, label=f'MuJoCo Flight Path under {wind_speed} m/s Crosswind (Paper 1 Control)')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend(loc='upper left')
    ax.set_title(f'End-to-End UAV Flight: Adaptive B-Spline Path & Quaternion Control Rejection ({wind_speed} m/s Wind)', fontsize=11)
    plt.tight_layout()

    mission_3d_path = os.path.join(output_dir, 'mission_flight_wind_rejection.png')
    plt.savefig(mission_3d_path, dpi=200)
    plt.close()
    print(f" Saved: {mission_3d_path}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_arr, tracking_errors / scale_factor, 'r-', linewidth=1.5)
    axes[0].set_ylabel('Tracking Error [m]')
    axes[0].grid(True)
    axes[0].set_title('Trajectory Tracking Error under Active Wind Gusts (Full Scale)', fontsize=11)

    axes[1].plot(t_arr, wind_forces_arr[:, 0], label='Wind Fx [N]')
    axes[1].plot(t_arr, wind_forces_arr[:, 1], label='Wind Fy [N]')
    axes[1].set_ylabel('Disturbance Force [N]')
    axes[1].set_xlabel('Flight Time [s]')
    axes[1].grid(True)
    axes[1].legend(loc='upper right')
    axes[1].set_title('Applied Atmospheric Wind & Gust Disturbance Forces', fontsize=11)

    plt.tight_layout()
    mission_error_path = os.path.join(output_dir, 'mission_wind_error_analysis.png')
    plt.savefig(mission_error_path, dpi=200)
    plt.close()
    print(f" Saved: {mission_error_path}")

    if gui:
        sim.keep_alive_until_closed()

    return {
        'mean_error': mean_err_mujoco,
        'max_error': max_err_mujoco,
        'final_arrival_error': final_arrival_err_mujoco,
        'path_length': metrics['path_length'],
        'max_curvature': metrics['max_curvature']
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-End UAV Mission Runner')
    parser.add_argument('--start', nargs=3, type=float, default=[0.0, 0.0, 20.0], help='Start position X Y Z')
    parser.add_argument('--goal', nargs=3, type=float, default=[2000.0, 1800.0, 200.0], help='Goal position X Y Z')
    parser.add_argument('--wind', type=float, default=3.5, help='Crosswind speed in m/s')
    parser.add_argument('--gui', action='store_true', help='Launch live interactive 3D MuJoCo GUI window')
    args = parser.parse_args()

    start_point = np.array(args.start)
    goal_point = np.array(args.goal)
    execute_mission(start_point, goal_point, wind_speed=args.wind, gui=args.gui)
