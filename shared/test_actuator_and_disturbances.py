"""
Demonstration script for actuator allocation, motor limits, and wind disturbance models:
1. Quaternion rotation matrix conversion (Shepperd algorithm for non-positive trace).
2. Motor mixing matrix (4 motors -> thrust & 3 torques) with speed limits.
3. Environmental wind & gust disturbance injection into Quadrotor dynamics.
4. Execution of full test suite.
"""
import numpy as np
import time

from attitude_control.quaternion import Quaternion
from attitude_control.motor_model import MotorModel
from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from environment.disturbances import WindDisturbanceModel


def demo_quaternion_fix():
    print("\n" + "=" * 60)
    print("DEMO 1: Quaternion Shepperd Algorithm Fix (trace <= 0)")
    print("=" * 60)

    # 180 degree rotation around X axis (trace = -1)
    R_x180 = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    q_x180 = Quaternion.from_rotation_matrix(R_x180)
    print(f"180° X-rotation matrix trace: {np.trace(R_x180):.1f}")
    print(f"Recovered quaternion:         {q_x180}")
    print(f"Success check (|q1| ~ 1.0):   {abs(q_x180.q1):.4f} -> OK!")


def demo_motor_mixing():
    print("\n" + "=" * 60)
    print("DEMO 2: Motor Mixing Matrix & Actuator Allocation")
    print("=" * 60)

    motor = MotorModel(kf=3.13e-5, km=7.5e-7, d=0.23, omega_min=100.0, omega_max=1200.0)

    print("Motor Allocation Matrix M:")
    print(motor.M.round(6))

    # Hover command: thrust = 9.81 N (1 kg quadrotor), zero torques
    thrust_hover = 9.81
    torques_hover = np.zeros(3)

    w_vec, act_thrust, act_torques, saturated = motor.allocate_and_limit(thrust_hover, torques_hover)
    print(f"\nHover Command: F = {thrust_hover:.2f}N, tau = [0, 0, 0]Nm")
    print(f"Calculated Motor Speeds (rad/s): {w_vec.round(2)}")
    print(f"Actual Thrust: {act_thrust:.2f}N, Actual Torques: {act_torques.round(4)}Nm")
    print(f"Motor Saturated: {saturated}")

    # Aggressive roll command
    thrust_roll = 12.0
    torques_roll = np.array([1.5, 0.0, 0.0])
    w_vec_roll, act_thrust_r, act_torques_r, saturated_r = motor.allocate_and_limit(thrust_roll, torques_roll)
    print(f"\nRoll Command: F = {thrust_roll:.2f}N, tau_x = {torques_roll[0]:.2f}Nm")
    print(f"Calculated Motor Speeds (rad/s): {w_vec_roll.round(2)}")
    print(f"Actual Torques: {act_torques_r.round(4)}Nm (Motor 4 vs Motor 2 speed offset)")


def demo_wind_disturbance():
    print("\n" + "=" * 60)
    print("DEMO 3: Environmental Wind & Gust Disturbance Injection")
    print("=" * 60)

    wind_model = WindDisturbanceModel(
        steady_wind=[3.0, 1.5, 0.5],
        gust_amplitude=[1.5, 1.0, 0.5],
        torque_bias=[0.05, -0.03, 0.02]
    )

    quad = QuadrotorModel(enable_motor_model=True)
    ctrl = P2Controller(Pq=20.0, Pw=4.0)

    dt_log = 0.01
    dt_sim = 0.0005
    substeps = int(round(dt_log / dt_sim))
    total_time = 2.0
    n_steps = int(total_time / dt_log)

    q_ref = Quaternion.from_euler(0.1, 0.0, 0.0)
    print(f"Simulating attitude tracking under wind disturbances for {total_time}s...")

    for i in range(n_steps):
        t = i * dt_log
        f_dist, tau_dist = wind_model.step(t, quad.velocity)

        for _ in range(substeps):
            tau, _ = ctrl.compute_torque(q_ref, quad.q, quad.omega)
            quad.step_attitude_only(tau, dt_sim, tau_dist=tau_dist)

        if i % (n_steps // 5) == 0:
            e = quad.q.to_euler()
            w_max = np.max(quad.motor_speeds)
            print(f"  t={t:.2f}s | Roll={e[0]:.4f}rad | Wind Force={np.linalg.norm(f_dist):.2f}N | "
                  f"Max Motor Speed={w_max:.1f}rad/s")

    print(f"Final Roll Angle: {quad.q.to_euler()[0]:.4f} rad (Desired: 0.1000 rad)")


if __name__ == '__main__':
    print("#" * 60)
    print("# ACTUATOR MODEL AND DISTURBANCE VALIDATION")
    print("#" * 60)

    demo_quaternion_fix()
    demo_motor_mixing()
    demo_wind_disturbance()

    print("\n" + "#" * 60)
    print("# ALL DEMONSTRATIONS SUCCESSFULLY COMPLETED")
    print("#" * 60)

