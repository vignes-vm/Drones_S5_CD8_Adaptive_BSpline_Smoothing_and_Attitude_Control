"""
Quaternion Attitude Controller Module
Implements the full non-linear P^2 quaternion attitude control scheme from Paper 1:
"Full Quaternion Based Attitude Control for a Quadrotor" (Fresk & Nikolakopoulos, 2013)
"""

import numpy as np
from src.quaternion_math import compute_quat_error, quat_normalize

class NonlinearP2QuaternionController:
    def __init__(self, Pq: float = 20.0, Pw: float = 4.0, max_torque: float = 4.0, noise_amp: float = 0.0):
        """
        Parameters from Paper 1:
        Pq = 20.0 (Outer loop proportional gain for attitude error)
        Pw = 4.0  (Inner loop proportional gain for angular velocity)
        max_torque = 4.0 Nm (Torque saturation limit per axis)
        noise_amp = 0.1 (Zero-mean noise amplitude affecting measurements)
        """
        self.Pq = Pq
        self.Pw = Pw
        self.max_torque = max_torque
        self.noise_amp = noise_amp

    def compute_torque(self, q_ref: np.ndarray, q_measured: np.ndarray, omega_measured: np.ndarray, add_noise: bool = True) -> tuple:
        """
        Computes control torque tau based on Paper 1 Equation (21):
        tau = Pq * Axis_err - Pw * omega

        Returns:
            tau: (3,) numpy array bounded by [-max_torque, +max_torque]
            q_err: (4,) unit error quaternion
            axis_err: (3,) vector axis error representation
        """
        q_m = q_measured.copy()
        omega_m = omega_measured.copy()

        if add_noise and self.noise_amp > 0:
            # Corrupt measurements with zero-mean noise of specified amplitude (Paper 1 Section V)
            q_noise = np.random.normal(0, self.noise_amp, size=4)
            q_m = quat_normalize(q_m + q_noise)
            omega_m += np.random.normal(0, self.noise_amp, size=3)

        # Compute error quaternion q_err = q_ref ⊗ q_m* with shortest path check (Eq 19)
        q_err = compute_quat_error(q_ref, q_m)

        # Compute axis error vector from quaternion representation (Eq 20)
        n_vec = np.linalg.norm(q_err[1:4])
        if n_vec > 1e-8:
            angle_err = 2.0 * np.arcsin(np.clip(n_vec, 0.0, 1.0))
            axis_err = angle_err * (q_err[1:4] / n_vec)
        else:
            axis_err = np.zeros(3)

        # P^2 Control Law (Eq 21)
        tau = self.Pq * axis_err - self.Pw * omega_m

        # Apply torque saturation bounds (+/- 4 Nm)
        tau_saturated = np.clip(tau, -self.max_torque, self.max_torque)

        return tau_saturated, q_err, axis_err
