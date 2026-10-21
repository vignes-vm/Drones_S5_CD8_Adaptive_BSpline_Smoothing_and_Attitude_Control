"""
Paper 2 Recreation in PyBullet Physics Environment.
Fresk & Nikolakopoulos, "Full Quaternion Based Attitude Control for a Quadrotor" (ECC 2013).
Recreates Figures 3 (Step Response), 5 (Sinusoidal Tracking), and 7 (360° Flip Maneuver) in PyBullet physics simulation.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pybullet.src.pybullet_env import PyBulletEnvironment
from pybullet.src.quadrotor_pybullet import PyBulletQuadrotor
from pybullet.src.quaternion_math import QuaternionMath
from pybullet.src.trajectory_tracker import PyBulletTrajectoryTracker


def run_paper2_maneuvers(gui=False):
    print("=" * 65)
    print("PAPER 2 RECREATION IN PYBULLET — QUATERNION ATTITUDE CONTROL")
    print("Maneuvers: Step Response (Fig 3), Sinusoidal (Fig 5), 360° Flip (Fig 7)")
    print("=" * 65)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_plots')
    os.makedirs(output_dir, exist_ok=True)

    dt_sim = 0.001  # 1 kHz physics rate
    pb_env = PyBulletEnvironment(gui=gui, dt=dt_sim)

    drone = PyBulletQuadrotor(
        physics_environment=pb_env,
        start_position=[0, 0, 5],
        start_orientation_euler=[0, 0, 0],
        enable_motor_model=True
    )

    tracker = PyBulletTrajectoryTracker(mass=drone.mass, g=drone.g, Pq=2.5, Pw=0.08, tau_limit=0.5)

    # ── Maneuver 1: Step Response ──
    print("\n[Maneuver 1] Running Step Response Simulation...")
    drone.reset([0, 0, 5], [0, 0, 0])
    total_time = 15.0
    n_steps = int(total_time / dt_sim)

    t_arr, euler_actual, euler_ref, torque_arr, err_mag_arr = [], [], [], [], []

    for i in range(n_steps):
        t = i * dt_sim

        # Reference attitude schedule
        if t < 2.0:
            q_ref = QuaternionMath.from_euler(0.0, 0.0, 0.0)
        elif t < 6.0:
            q_ref = QuaternionMath.from_euler(0.5, 0.0, 0.0)   # Roll step
        elif t < 10.0:
            q_ref = QuaternionMath.from_euler(0.0, 0.5, 0.0)   # Pitch step
        else:
            q_ref = QuaternionMath.from_euler(0.0, 0.0, 0.8)   # Yaw step

        pos, vel, q_meas, omega_meas = drone.get_state()
        tau_cmd, axis_err = tracker.compute_attitude_control(q_ref, q_meas, omega_meas)

        # Hover thrust force to maintain altitude in PyBullet
        thrust_cmd = drone.mass * drone.g
        drone.apply_control_inputs(thrust_cmd, tau_cmd)
        pb_env.step()

        t_arr.append(t)
        euler_actual.append(QuaternionMath.to_euler(q_meas))
        euler_ref.append(QuaternionMath.to_euler(q_ref))
        torque_arr.append(drone.applied_torques.copy())
        err_mag_arr.append(np.linalg.norm(axis_err))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
    for i in range(3):
        axes[i].plot(t_arr, np.array(euler_ref)[:, i], 'k--', lw=1.5, label='Reference')
        axes[i].plot(t_arr, np.array(euler_actual)[:, i], 'b-', lw=1.5, label='PyBullet Actual')
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('PyBullet Paper 2 Recreation — Step Response (Figure 3)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pybullet_paper2_step_response.png'), dpi=150)
    plt.close(fig)

    # ── Maneuver 2: Sinusoidal Tracking ──
    print("\n[Maneuver 2] Running Sinusoidal Tracking Simulation...")
    drone.reset([0, 0, 5], [0, 0, 0])
    t_arr, euler_actual, euler_ref = [], [], []

    for i in range(n_steps):
        t = i * dt_sim
        phi   = 0.3 * np.sin(1.0 * t)
        theta = 0.3 * np.sin(1.0 * t + np.pi/4)
        psi   = 0.4 * np.sin(1.0 * t + np.pi/2)
        q_ref = QuaternionMath.from_euler(phi, theta, psi)

        pos, vel, q_meas, omega_meas = drone.get_state()
        tau_cmd, _ = tracker.compute_attitude_control(q_ref, q_meas, omega_meas)

        thrust_cmd = drone.mass * drone.g
        drone.apply_control_inputs(thrust_cmd, tau_cmd)
        pb_env.step()

        t_arr.append(t)
        euler_actual.append(QuaternionMath.to_euler(q_meas))
        euler_ref.append(QuaternionMath.to_euler(q_ref))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for i in range(3):
        axes[i].plot(t_arr, np.array(euler_ref)[:, i], 'k--', lw=1.5, label='Reference')
        axes[i].plot(t_arr, np.array(euler_actual)[:, i], 'b-', lw=1.5, label='PyBullet Actual')
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('PyBullet Paper 2 Recreation — Sinusoidal Tracking (Figure 5)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pybullet_paper2_sine_tracking.png'), dpi=150)
    plt.close(fig)

    # ── Maneuver 3: 360° Flip Maneuver ──
    print("\n[Maneuver 3] Running 360° Flip Maneuver Simulation...")
    drone.reset([0, 0, 5], [0, 0, 0])
    flip_time = 5.0
    n_flip_steps = int(flip_time / dt_sim)

    t_arr, roll_actual, roll_ref, q0_arr, q1_arr = [], [], [], [], []

    for i in range(n_flip_steps):
        t = i * dt_sim
        phi_ref = 2.0 * np.pi * (t / flip_time)
        q_ref = QuaternionMath.from_euler(phi_ref, 0.0, 0.0)

        pos, vel, q_meas, omega_meas = drone.get_state()
        tau_cmd, _ = tracker.compute_attitude_control(q_ref, q_meas, omega_meas)

        thrust_cmd = drone.mass * drone.g
        drone.apply_control_inputs(thrust_cmd, tau_cmd)
        pb_env.step()

        t_arr.append(t)
        roll_actual.append(QuaternionMath.to_euler(q_meas)[0])
        roll_ref.append(phi_ref)
        q0_arr.append(q_meas[0])
        q1_arr.append(q_meas[1])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t_arr, np.unwrap(roll_ref), 'k--', lw=1.5, label='Reference (ramp)')
    axes[0].plot(t_arr, np.unwrap(roll_actual), 'tomato', lw=1.5, label='PyBullet Roll φ')
    axes[0].set_ylabel('Roll φ (rad)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].set_title('360° Flip — Roll Angle')

    axes[1].plot(t_arr, q0_arr, color='steelblue', lw=1.5, label='q0 (scalar)')
    axes[1].plot(t_arr, q1_arr, color='seagreen', lw=1.5, label='q1 (x-axis)')
    axes[1].axhline(y=0, color='k', linestyle='--', lw=0.8, alpha=0.5)
    axes[1].set_ylabel('Quaternion Component')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_title('360° Flip — Quaternion Components (Singularity-Free)')
    fig.suptitle('PyBullet Paper 2 Recreation — 360° Flip Maneuver (Figure 7)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pybullet_paper2_flip.png'), dpi=150)
    plt.close(fig)

    pb_env.close()
    print(f"\nAll Paper 2 plots saved to: {output_dir}")


if __name__ == '__main__':
    run_paper2_maneuvers()
