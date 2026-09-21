"""
Paper 2 Attitude Control Recreation in Gazebo Physics.
Recreates Paper 2 ("Full Quaternion Based Attitude Control for a Quadrotor", ECC 2013):
  1. Large-angle Step Response (Paper 2 Figure 3)
  2. Sinusoidal Attitude Tracking (Paper 2 Figure 5)
  3. 360° Flip Maneuver demonstrating singularity-free full attitude control (Paper 2 Figure 7)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gazebo.src.quadrotor_gazebo import GazeboQuadrotor
from gazebo.src.quaternion_math import QuaternionMath
from gazebo.src.trajectory_tracker import GazeboTrajectoryTracker


def run_step_response(output_dir):
    print("\n[Paper 2] Test 1: Step Response...", flush=True)
    drone = GazeboQuadrotor(enable_motor_model=False)
    tracker = GazeboTrajectoryTracker(Pq=20.0, Pw=4.0, tau_limit=4.0)

    dt = 0.001
    duration = 3.0
    steps = int(duration / dt)

    t_arr = np.linspace(0, duration, steps)
    euler_arr = np.zeros((steps, 3))
    ref_arr = np.zeros((steps, 3))
    tau_arr = np.zeros((steps, 3))

    # Reference: 45 degree step in roll at t = 0.2s
    for i, t in enumerate(t_arr):
        roll_ref = np.radians(45.0) if t >= 0.2 else 0.0
        pitch_ref = 0.0
        yaw_ref = 0.0
        q_ref = QuaternionMath.from_euler(roll_ref, pitch_ref, yaw_ref)

        tau, _, _ = tracker.compute_attitude_control(q_ref, drone.q, drone.omega)
        drone.step_attitude_only(tau, dt=dt)

        t_arr[i] = t
        euler_arr[i] = drone.get_euler()
        ref_arr[i] = [roll_ref, pitch_ref, yaw_ref]
        tau_arr[i] = tau

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t_arr, np.degrees(ref_arr[:, 0]), 'k--', label='Roll Reference')
    ax1.plot(t_arr, np.degrees(euler_arr[:, 0]), 'b-', lw=2, label='Roll Response (Gazebo)')
    ax1.plot(t_arr, np.degrees(euler_arr[:, 1]), 'g-', lw=1.5, label='Pitch')
    ax1.plot(t_arr, np.degrees(euler_arr[:, 2]), 'r-', lw=1.5, label='Yaw')
    ax1.set_ylabel('Euler Angles (deg)', fontsize=11)
    ax1.set_title('Gazebo Paper 2: 45° Step Response', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t_arr, tau_arr[:, 0], 'b-', lw=1.5, label='Mx (Roll Torque)')
    ax2.axhline(4.0, color='r', linestyle=':', label='Torque Limit (+/-4 Nm)')
    ax2.axhline(-4.0, color='r', linestyle=':')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Control Torque (Nm)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'gazebo_paper2_step_response.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {plot_path}", flush=True)


def run_sine_tracking(output_dir):
    print("\n[Paper 2] Test 2: Sinusoidal Attitude Tracking...", flush=True)
    drone = GazeboQuadrotor(enable_motor_model=False)
    tracker = GazeboTrajectoryTracker(Pq=20.0, Pw=4.0, tau_limit=4.0)

    dt = 0.001
    duration = 5.0
    steps = int(duration / dt)

    t_arr = np.linspace(0, duration, steps)
    euler_arr = np.zeros((steps, 3))
    ref_arr = np.zeros((steps, 3))
    err_arr = np.zeros((steps, 3))

    for i, t in enumerate(t_arr):
        roll_ref = np.radians(20.0) * np.sin(2.0 * np.pi * 0.5 * t)
        pitch_ref = np.radians(15.0) * np.cos(2.0 * np.pi * 0.4 * t)
        yaw_ref = 0.0
        q_ref = QuaternionMath.from_euler(roll_ref, pitch_ref, yaw_ref)

        tau, axis_err, _ = tracker.compute_attitude_control(q_ref, drone.q, drone.omega)
        drone.step_attitude_only(tau, dt=dt)

        t_arr[i] = t
        euler_arr[i] = drone.get_euler()
        ref_arr[i] = [roll_ref, pitch_ref, yaw_ref]
        err_arr[i] = axis_err

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t_arr, np.degrees(ref_arr[:, 0]), 'b--', label='Roll Ref')
    ax1.plot(t_arr, np.degrees(euler_arr[:, 0]), 'b-', lw=1.5, label='Roll Actual')
    ax1.plot(t_arr, np.degrees(ref_arr[:, 1]), 'g--', label='Pitch Ref')
    ax1.plot(t_arr, np.degrees(euler_arr[:, 1]), 'g-', lw=1.5, label='Pitch Actual')
    ax1.set_ylabel('Angle (deg)', fontsize=11)
    ax1.set_title('Gazebo Paper 2: Sinusoidal Attitude Tracking', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(t_arr, err_arr[:, 0], 'b-', lw=1.2, label='Roll Axis Error q1')
    ax2.plot(t_arr, err_arr[:, 1], 'g-', lw=1.2, label='Pitch Axis Error q2')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Quaternion Error', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'gazebo_paper2_sine_tracking.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {plot_path}", flush=True)


def run_flip_maneuver(output_dir):
    print("\n[Paper 2] Test 3: 360° Flip Maneuver (Singularity-Free Unwinding)...", flush=True)
    drone = GazeboQuadrotor(enable_motor_model=False)
    tracker = GazeboTrajectoryTracker(Pq=20.0, Pw=4.0, tau_limit=4.0)

    dt = 0.001
    duration = 4.0
    steps = int(duration / dt)

    t_arr = np.linspace(0, duration, steps)
    euler_arr = np.zeros((steps, 3))
    ref_arr = np.zeros((steps, 3))
    quat_arr = np.zeros((steps, 4))

    for i, t in enumerate(t_arr):
        if t < 0.5:
            roll_ref = 0.0
        elif t < 2.5:
            # Smooth 360-degree rotation in roll
            tau_rel = (t - 0.5) / 2.0
            roll_ref = 2.0 * np.pi * tau_rel
        else:
            roll_ref = 2.0 * np.pi

        q_ref = QuaternionMath.from_euler(roll_ref, 0.0, 0.0)
        tau, _, _ = tracker.compute_attitude_control(q_ref, drone.q, drone.omega)
        drone.step_attitude_only(tau, dt=dt)

        t_arr[i] = t
        euler_arr[i] = drone.get_euler()
        ref_arr[i] = [roll_ref, 0.0, 0.0]
        quat_arr[i] = drone.q.copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t_arr, np.degrees(np.unwrap(ref_arr[:, 0])), 'k--', label='Desired Roll (0° -> 360°)')
    ax1.plot(t_arr, np.degrees(np.unwrap(euler_arr[:, 0])), 'm-', lw=2, label='Actual Roll (Unwrapped)')
    ax1.set_ylabel('Roll Angle (deg)', fontsize=11)
    ax1.set_title('Gazebo Paper 2: 360° Singularity-Free Flip Maneuver', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t_arr, quat_arr[:, 0], label='q0 (w)', lw=1.5)
    ax2.plot(t_arr, quat_arr[:, 1], label='q1 (x)', lw=1.5)
    ax2.plot(t_arr, quat_arr[:, 2], label='q2 (y)', lw=1.5)
    ax2.plot(t_arr, quat_arr[:, 3], label='q3 (z)', lw=1.5)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Quaternion Components', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'gazebo_paper2_flip.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {plot_path}", flush=True)


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
    os.makedirs(out_dir, exist_ok=True)
    run_step_response(out_dir)
    run_sine_tracking(out_dir)
    run_flip_maneuver(out_dir)
