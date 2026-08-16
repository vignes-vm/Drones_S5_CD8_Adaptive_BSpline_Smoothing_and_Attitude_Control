"""
Quadrotor Actuator & Motor Mixing Model.
Paper 2 — Equations (5)-(9): Motor allocation matrix mapping total thrust F
and roll/pitch/yaw torques tau = [tau_x, tau_y, tau_z]^T to individual motor speeds.
"""
import numpy as np


class MotorModel:
    """
    Quadrotor motor mixing model for 4 rotors in '+' configuration.
    
    Equations of motion mapping motor angular speeds omega_i (rad/s) to
    total thrust F and torques (tau_x, tau_y, tau_z):
        F     = kf * (w1^2 + w2^2 + w3^2 + w4^2)
        tau_x = d * kf * (w4^2 - w2^2)
        tau_y = d * kf * (w1^2 - w3^2)
        tau_z = km * (w2^2 + w4^2 - w1^2 - w3^2)
    """

    def __init__(self, kf=3.13e-5, km=7.5e-7, d=0.23,
                 omega_min=0.0, omega_max=1200.0):
        self.kf = float(kf)
        self.km = float(km)
        self.d  = float(d)

        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

        # Allocation matrix M such that [F, tau_x, tau_y, tau_z]^T = M @ [w1^2, w2^2, w3^2, w4^2]^T
        self.M = np.array([
            [ self.kf,          self.kf,          self.kf,          self.kf],
            [ 0.0,             -self.d*self.kf,   0.0,              self.d*self.kf],
            [ self.d*self.kf,   0.0,             -self.d*self.kf,   0.0],
            [-self.km,          self.km,         -self.km,          self.km]
        ], dtype=float)

        self.M_inv = np.linalg.inv(self.M)

    def compute_thrust_and_torques(self, omega_vec):
        """
        Forward mapping: motor speeds (rad/s) -> total thrust F and torques tau.
        """
        omega_sq = np.clip(np.asarray(omega_vec, dtype=float), self.omega_min, self.omega_max)**2
        wrench = self.M @ omega_sq
        total_thrust = wrench[0]
        torques = wrench[1:4]
        return total_thrust, torques

    def compute_motor_speeds(self, total_thrust, torques):
        """
        Inverse mapping: desired thrust F and torques tau -> motor speeds omega_i (rad/s).
        Clips output to motor bounds [omega_min, omega_max].
        """
        wrench = np.zeros(4)
        wrench[0] = max(0.0, float(total_thrust))
        wrench[1:4] = np.asarray(torques, dtype=float)

        omega_sq_desired = self.M_inv @ wrench
        
        # Check saturation before clipping
        w_min_sq = self.omega_min**2
        w_max_sq = self.omega_max**2

        is_saturated = bool(np.any(omega_sq_desired < w_min_sq) or np.any(omega_sq_desired > w_max_sq))
        
        omega_sq_clipped = np.clip(omega_sq_desired, w_min_sq, w_max_sq)
        omega_vec = np.sqrt(omega_sq_clipped)

        return omega_vec, is_saturated

    def allocate_and_limit(self, total_thrust, torques):
        """
        Calculates motor speeds for given thrust and torque commands,
        then evaluates actual achieved thrust and torques after speed clipping.
        """
        omega_vec, is_saturated = self.compute_motor_speeds(total_thrust, torques)
        actual_thrust, actual_torques = self.compute_thrust_and_torques(omega_vec)
        return omega_vec, actual_thrust, actual_torques, is_saturated
