"""
First-Order Quadrotor Actuator & Motor Model.
Implements first-order lag, speed bounds [w_min, w_max],
thrust coefficient c_T, and drag torque coefficient c_Q.
"""

from typing import Tuple
import numpy as np


class MotorModel:
    """
    Simulates four brushless DC motors with 1st-order response lag:
        d(omega_i)/dt = (omega_i,cmd - omega_i) / tau_m
    """
    def __init__(
        self,
        tau_m: float = 0.02,          # motor time constant in seconds
        omega_min: float = 100.0,      # minimum RPM (idle) in rad/s
        omega_max: float = 1200.0,     # maximum RPM in rad/s
        c_T: float = 1.105e-5,         # thrust coefficient: F_i = c_T * omega_i^2 (N/(rad/s)^2)
        c_Q: float = 1.779e-7,         # drag torque coefficient: Q_i = c_Q * omega_i^2 (N*m/(rad/s)^2)
        arm_length: float = 0.225      # distance from center to rotor axis in meters
    ):
        self.tau_m = tau_m
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.c_T = c_T
        self.c_Q = c_Q
        self.arm_length = arm_length

        # State: current motor speeds (4,)
        self.omega = np.full(4, 500.0, dtype=np.float64)

    def reset(self, initial_omega: float = 500.0) -> None:
        """Resets motor speeds."""
        self.omega = np.full(4, initial_omega, dtype=np.float64)

    def step(self, omega_cmd: np.ndarray, dt: float) -> np.ndarray:
        """
        Updates motor speeds towards omega_cmd via 1st-order Euler step.
        Clamps to [omega_min, omega_max].
        """
        omega_cmd_clamped = np.clip(omega_cmd, self.omega_min, self.omega_max)
        alpha = np.clip(dt / max(self.tau_m, 1e-4), 0.0, 1.0)
        self.omega += alpha * (omega_cmd_clamped - self.omega)
        self.omega = np.clip(self.omega, self.omega_min, self.omega_max)
        return self.omega.copy()

    def compute_forces_and_torques(self, omega: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes total collective thrust (N), body torques (N*m) [tau_x, tau_y, tau_z],
        and individual rotor thrusts.
        Rotor layout (Standard Plus / Cross Configuration):
            Rotor 1: +X (Front), CCW (c_Q > 0)
            Rotor 2: +Y (Right), CW  (c_Q < 0)
            Rotor 3: -X (Rear),  CCW (c_Q > 0)
            Rotor 4: -Y (Left),  CW  (c_Q < 0)
        """
        omega_sq = omega ** 2
        f_rotors = self.c_T * omega_sq
        total_thrust = float(np.sum(f_rotors))

        d = self.arm_length
        tau_x = d * (f_rotors[1] - f_rotors[3])  # rotor 2 (+Y) minus rotor 4 (-Y)
        tau_y = d * (f_rotors[2] - f_rotors[0])  # rotor 3 (-X) minus rotor 1 (+X)
        tau_z = self.c_Q * (omega_sq[0] - omega_sq[1] + omega_sq[2] - omega_sq[3])

        tau_body = np.array([tau_x, tau_y, tau_z], dtype=np.float64)
        return total_thrust, tau_body, f_rotors
