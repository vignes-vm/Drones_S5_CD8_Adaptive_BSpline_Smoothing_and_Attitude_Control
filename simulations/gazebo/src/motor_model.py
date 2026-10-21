"""
Motor Allocation & Dynamics Model for Gazebo Quadrotor.
Implements the 4-rotor '+' configuration control allocation matrix:
  [T, tau_x, tau_y, tau_z]^T = M * [f_0, f_1, f_2, f_3]^T
Handles individual motor thrust limits, saturation detection, and motor time constants.
"""
import numpy as np


class MotorModel:

    def __init__(self, d=0.23, c_tau_f=0.016, min_thrust=0.0, max_thrust=8.0, tau_motor=0.02):
        """
        d: Arm length from center to motor (m), default 0.23m
        c_tau_f: Drag torque to thrust ratio (m)
        min_thrust: Minimum thrust per motor (N)
        max_thrust: Maximum thrust per motor (N), 4x8N = 32N total (hover needs ~9.81N)
        tau_motor: Motor first-order lag time constant (s)
        """
        self.d = d
        self.c_tau_f = c_tau_f
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.tau_motor = tau_motor

        # '+' Configuration Forward Allocation Matrix:
        # Motor 0: +X (Front), CCW (-)
        # Motor 1: -Y (Right), CW  (+)
        # Motor 2: -X (Back),  CCW (-)
        # Motor 3: +Y (Left),  CW  (+)
        self.B = np.array([
            [ 1.0,        1.0,         1.0,        1.0        ],  # Total Thrust T
            [ 0.0,       -self.d,      0.0,        self.d     ],  # Roll Torque tau_x
            [ self.d,     0.0,        -self.d,     0.0        ],  # Pitch Torque tau_y
            [-self.c_tau_f, self.c_tau_f, -self.c_tau_f, self.c_tau_f] # Yaw Torque tau_z
        ])

        # Inverse Allocation Matrix (pseudo-inverse)
        self.B_inv = np.linalg.pinv(self.B)

        self.current_thrusts = np.zeros(4)
        self.is_saturated = False

    def allocate(self, thrust_cmd, torques_cmd):
        """
        Maps desired total thrust (N) and torques (Nm) to 4 individual motor thrusts (N).
        Clamps each motor thrust to [min_thrust, max_thrust].
        Returns:
          actual_thrust: Achieved total thrust (N)
          actual_torques: Achieved torques [tau_x, tau_y, tau_z] (Nm)
          motor_thrusts: Vector of 4 motor thrusts (N)
        """
        wrench_cmd = np.array([
            thrust_cmd,
            torques_cmd[0],
            torques_cmd[1],
            torques_cmd[2]
        ], dtype=float)

        raw_thrusts = self.B_inv @ wrench_cmd

        clamped_thrusts = np.clip(raw_thrusts, self.min_thrust, self.max_thrust)
        self.is_saturated = np.any(np.abs(clamped_thrusts - raw_thrusts) > 1e-4)

        achieved_wrench = self.B @ clamped_thrusts
        actual_thrust = achieved_wrench[0]
        actual_torques = achieved_wrench[1:4]

        return actual_thrust, actual_torques, clamped_thrusts

    def step_motor_dynamics(self, target_thrusts, dt):
        """Simulates 1st-order motor response: df/dt = (f_target - f) / tau_motor."""
        alpha = dt / (self.tau_motor + dt)
        self.current_thrusts += alpha * (target_thrusts - self.current_thrusts)
        self.current_thrusts = np.clip(self.current_thrusts, self.min_thrust, self.max_thrust)
        return self.current_thrusts.copy()
