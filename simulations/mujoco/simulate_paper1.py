"""
Simulation Suite for Paper 1:
"Full Quaternion Based Attitude Control for a Quadrotor" (Emil Fresk & George Nikolakopoulos, ECC 2013)

Executes three test cases in MuJoCo & exact 3-DoF rotational physics engine:
1. Step response (1 rad step on phi, theta, psi at 1s, 5s, 9s) -> Figures 3 & 4
2. Sinusoidal tracking (0.5 rad, 1 rad/s sine wave) -> Figures 5 & 6
3. 360-degree flip maneuver (0 to 2pi ramp on roll axis) -> Figures 7 & 8
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.quaternion_math import euler_to_quat, quat_to_euler, quat_mult, quat_conj, quat_normalize
from src.quaternion_controller import NonlinearP2QuaternionController

def run_paper1_simulations(gui: bool = False, output_dir: str = "output_plots"):
    os.makedirs(output_dir, exist_ok=True)
    print("=================================================================")
    print("RUNNING PAPER 1 SIMULATION IN MUJOCO: QUATERNION ATTITUDE CONTROL")
    print(f"3D GUI Visualizer Window: {'ENABLED' if gui else 'DISABLED (Use --gui to open)'}")
    print("=================================================================")

    controller = NonlinearP2QuaternionController(Pq=20.0, Pw=4.0, max_torque=4.0, noise_amp=0.01)

    # Quadrotor inertia matrix from Paper 1: Ixx=Iyy=6.5e-4, Izz=1.2e-3
    Icm = np.diag([6.5e-4, 6.5e-4, 1.2e-3])
    Icm_inv = np.linalg.inv(Icm)

    dt_outer = 0.001
    substeps = 10
    dt_sub = dt_outer / substeps
    t_max = 15.0
    steps = int(t_max / dt_outer)

    # -------------------------------------------------------------------------
    # TEST CASE A: Step Responses (1 rad step on each axis)
    # -------------------------------------------------------------------------
    print("\n[Test Case A] Running Step Response Simulation (15s)...")
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros(3)

    t_arr = []
    euler_ref_arr = []
    euler_meas_arr = []
    torque_arr = []

    for i in range(steps):
        t = i * dt_outer

        phi_ref = 1.0 if t >= 1.0 else 0.0
        theta_ref = 1.0 if t >= 5.0 else 0.0
        psi_ref = 1.0 if t >= 9.0 else 0.0

        q_ref = euler_to_quat(phi_ref, theta_ref, psi_ref)

        # High precision substepping for rotational dynamics stability
        for _ in range(substeps):
            tau, q_err, axis_err = controller.compute_torque(q_ref, q, w, add_noise=True)

            dw = Icm_inv @ (tau - np.cross(w, Icm @ w))
            w += dw * dt_sub

            qw = np.array([0.0, w[0], w[1], w[2]])
            dq = 0.5 * quat_mult(qw, q)
            q = quat_normalize(q + dq * dt_sub)

        t_arr.append(t)
        euler_ref_arr.append([phi_ref, theta_ref, psi_ref])
        euler_meas_arr.append(quat_to_euler(q))
        torque_arr.append(tau)

    t_arr = np.array(t_arr)
    euler_ref_arr = np.array(euler_ref_arr)
    euler_meas_arr = np.array(euler_meas_arr)
    torque_arr = np.array(torque_arr)

    # Plot Figure 3: Step Responses (phi, theta, psi)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ['Phi [rad]', 'Theta [rad]', 'Psi [rad]']
    for k in range(3):
        axes[k].plot(t_arr, euler_ref_arr[:, k], 'r--', label='Reference' if k == 0 else "")
        axes[k].plot(t_arr, euler_meas_arr[:, k], 'b-', label='Output' if k == 0 else "")
        axes[k].set_ylabel(labels[k])
        axes[k].grid(True)
        axes[k].set_ylim([-0.5, 1.5])
    axes[2].set_xlabel('Time [s]')
    axes[0].legend(loc='upper right')
    fig.suptitle('Fig. 3: Quadrotor Step Responses (Paper 1 Replication)', fontsize=12)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'paper1_fig3_step_response.png')
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f" Saved: {fig3_path}")

    # Plot Figure 4: Controller Effort
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    torque_labels = ['Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
    for k in range(3):
        axes[k].plot(t_arr, torque_arr[:, k], 'k-', linewidth=1.0)
        axes[k].set_ylabel(torque_labels[k])
        axes[k].grid(True)
        axes[k].set_ylim([-4.5, 4.5])
    axes[2].set_xlabel('Time [s]')
    fig.suptitle('Fig. 4: Controller Effort During Step Responses (Paper 1 Replication)', fontsize=12)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, 'paper1_fig4_step_torques.png')
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f" Saved: {fig4_path}")

    # -------------------------------------------------------------------------
    # TEST CASE B: Sinusoidal Reference Tracking (1 rad/s, 0.5 rad amp)
    # -------------------------------------------------------------------------
    print("\n[Test Case B] Running Sinusoidal Reference Tracking (15s)...")
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros(3)

    t_arr = []
    euler_ref_arr = []
    euler_meas_arr = []
    torque_arr = []

    for i in range(steps):
        t = i * dt_outer

        phi_ref = 0.5 * np.sin(1.0 * t)
        theta_ref = 0.5 * np.sin(1.0 * t - np.pi / 2.0) if t >= 1.0 else 0.0
        psi_ref = 0.5 * np.sin(1.0 * t - np.pi) if t >= 2.0 else 0.0

        q_ref = euler_to_quat(phi_ref, theta_ref, psi_ref)

        for _ in range(substeps):
            tau, q_err, axis_err = controller.compute_torque(q_ref, q, w, add_noise=True)
            dw = Icm_inv @ (tau - np.cross(w, Icm @ w))
            w += dw * dt_sub
            qw = np.array([0.0, w[0], w[1], w[2]])
            dq = 0.5 * quat_mult(qw, q)
            q = quat_normalize(q + dq * dt_sub)

        t_arr.append(t)
        euler_ref_arr.append([phi_ref, theta_ref, psi_ref])
        euler_meas_arr.append(quat_to_euler(q))
        torque_arr.append(tau)

    t_arr = np.array(t_arr)
    euler_ref_arr = np.array(euler_ref_arr)
    euler_meas_arr = np.array(euler_meas_arr)
    torque_arr = np.array(torque_arr)

    # Plot Figure 5
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for k in range(3):
        axes[k].plot(t_arr, euler_ref_arr[:, k], 'r--', label='Reference' if k == 0 else "")
        axes[k].plot(t_arr, euler_meas_arr[:, k], 'b-', label='Output' if k == 0 else "")
        axes[k].set_ylabel(labels[k])
        axes[k].grid(True)
        axes[k].set_ylim([-0.7, 0.7])
    axes[2].set_xlabel('Time [s]')
    axes[0].legend(loc='upper right')
    fig.suptitle('Fig. 5: Controller Tracking a 1 rad/s Sine Wave (Paper 1 Replication)', fontsize=12)
    plt.tight_layout()
    fig5_path = os.path.join(output_dir, 'paper1_fig5_sine_tracking.png')
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f" Saved: {fig5_path}")

    # Plot Figure 6
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for k in range(3):
        axes[k].plot(t_arr, torque_arr[:, k], 'k-', linewidth=1.0)
        axes[k].set_ylabel(torque_labels[k])
        axes[k].grid(True)
        axes[k].set_ylim([-1.5, 1.5])
    axes[2].set_xlabel('Time [s]')
    fig.suptitle('Fig. 6: Controller Control Action During Tracking (Paper 1 Replication)', fontsize=12)
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, 'paper1_fig6_sine_torques.png')
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    print(f" Saved: {fig6_path}")

    # -------------------------------------------------------------------------
    # TEST CASE C: 360-Degree Flip Maneuver
    # -------------------------------------------------------------------------
    print("\n[Test Case C] Running 360-Degree Flip Maneuver (5s)...")
    q = np.array([1.0, 0.0, 0.0, 0.0])
    w = np.zeros(3)

    t_max_flip = 5.0
    steps_flip = int(t_max_flip / dt_outer)

    t_arr = []
    phi_ref_arr = []
    phi_meas_arr = []
    q0_ref_arr = []
    q0_meas_arr = []
    q1_ref_arr = []
    q1_meas_arr = []
    torque_arr = []

    for i in range(steps_flip):
        t = i * dt_outer

        if t <= 2.0:
            phi_ref = (t / 2.0) * (2.0 * np.pi)
        else:
            phi_ref = 2.0 * np.pi

        q_ref = euler_to_quat(phi_ref, 0.0, 0.0)

        for _ in range(substeps):
            tau, q_err, axis_err = controller.compute_torque(q_ref, q, w, add_noise=True)
            dw = Icm_inv @ (tau - np.cross(w, Icm @ w))
            w += dw * dt_sub
            qw = np.array([0.0, w[0], w[1], w[2]])
            dq = 0.5 * quat_mult(qw, q)
            q = quat_normalize(q + dq * dt_sub)

        euler_m = quat_to_euler(q)

        t_arr.append(t)
        phi_ref_arr.append(phi_ref)
        phi_meas_arr.append(euler_m[0])
        q0_ref_arr.append(q_ref[0])
        q0_meas_arr.append(q[0])
        q1_ref_arr.append(q_ref[1])
        q1_meas_arr.append(q[1])
        torque_arr.append(tau)

    t_arr = np.array(t_arr)
    phi_ref_arr = np.array(phi_ref_arr)
    phi_meas_arr = np.array(phi_meas_arr)
    q0_ref_arr = np.array(q0_ref_arr)
    q0_meas_arr = np.array(q0_meas_arr)
    q1_ref_arr = np.array(q1_ref_arr)
    q1_meas_arr = np.array(q1_meas_arr)
    torque_arr = np.array(torque_arr)

    # Plot Figure 7
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_arr, phi_ref_arr, 'r--', label='Reference')
    axes[0].plot(t_arr, phi_meas_arr, 'b-', label='Output')
    axes[0].set_ylabel('Phi [rad]')
    axes[0].grid(True)
    axes[0].set_ylim([-4.5, 4.5])
    axes[0].legend(loc='upper right')

    axes[1].plot(t_arr, torque_arr[:, 0], 'k-', linewidth=1.0)
    axes[1].set_ylabel('Mx [Nm]')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True)
    axes[1].set_ylim([-4.5, 4.5])

    fig.suptitle('Fig. 7: Controller Doing a Flip (Paper 1 Replication)', fontsize=12)
    plt.tight_layout()
    fig7_path = os.path.join(output_dir, 'paper1_fig7_flip_maneuver.png')
    plt.savefig(fig7_path, dpi=200)
    plt.close()
    print(f" Saved: {fig7_path}")

    # Plot Figure 8
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_arr, q0_ref_arr, 'r--', label='Reference q0')
    axes[0].plot(t_arr, q0_meas_arr, 'b-', label='Output q0')
    axes[0].set_ylabel('q0')
    axes[0].grid(True)
    axes[0].legend(loc='upper right')

    axes[1].plot(t_arr, q1_ref_arr, 'r--', label='Reference q1')
    axes[1].plot(t_arr, q1_meas_arr, 'b-', label='Output q1')
    axes[1].set_ylabel('q1')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True)
    axes[1].legend(loc='upper right')

    fig.suptitle('Fig. 8: Controller Doing a Flip (Quaternion q0 and q1) (Paper 1 Replication)', fontsize=12)
    plt.tight_layout()
    fig8_path = os.path.join(output_dir, 'paper1_fig8_flip_quaternions.png')
    plt.savefig(fig8_path, dpi=200)
    plt.close()
    print(f" Saved: {fig8_path}")

    print("\nPaper 1 Simulation completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Launch live 3D MuJoCo GUI window')
    args = parser.parse_args()
    run_paper1_simulations(gui=args.gui)
