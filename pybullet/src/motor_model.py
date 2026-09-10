"""
Quadrotor Actuator & Motor Mixing Model for PyBullet.
Paper 2 Equations (5)-(9): Motor allocation matrix mapping total thrust F
and roll/pitch/yaw torques tau = [tau_x, tau_y, tau_z]^T to individual motor speeds and rotor forces.
"""
import numpy as np


class MotorModel:

    def __init__(self, kf=3.13e-5, km=7.5e-7, d=0.23,
                 omega_min=0.0, omega_max=1200.0):
        self.kf = float(kf)
        self.km = float(km)
        self.d  = float(d)

        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

        # Allocation matrix M mapping [w1^2, w2^2, w3^2, w4^2]^T -> [F, tau_x, tau_y, tau_z]^T
        self.M = np.array([
            [ self.kf,          self.kf,          self.kf,          self.kf],
            [ 0.0,             -self.d*self.kf,   0.0,              self.d*self.kf],
            [ self.d*self.kf,   0.0,             -self.d*self.kf,   0.0],
            [-self.km,          self.km,         -self.km,          self.km]
        ], dtype=float)

        self.M_inv = np.linalg.inv(self.M)

    def compute_motor_speeds(self, total_thrust, torques):
        """
        Inverse mapping: desired thrust F and torques tau -> motor speeds omega_i (rad/s).
        Prioritizes control torques tau over total thrust F when motors approach saturation limits.
        """
        wrench_t = np.zeros(4)
        wrench_t[1:4] = np.asarray(torques, dtype=float)
        w_sq_t = self.M_inv @ wrench_t

        w_sq_f_single = max(0.0, float(total_thrust)) / (4.0 * self.kf)

        w_sq = w_sq_f_single + w_sq_t

        w_min_sq = self.omega_min**2
        w_max_sq = self.omega_max**2

        is_saturated = bool(np.any(w_sq < w_min_sq) or np.any(w_sq > w_max_sq))

        # Torque-prioritized derating of thrust contribution if max limit exceeded
        max_val = np.max(w_sq)
        if max_val > w_max_sq:
            overflow = max_val - w_max_sq
            w_sq_f_single = max(w_min_sq, w_sq_f_single - overflow)
            w_sq = w_sq_f_single + w_sq_t

        w_sq_clipped = np.clip(w_sq, w_min_sq, w_max_sq)
        omega_vec = np.sqrt(w_sq_clipped)

        return omega_vec, is_saturated

    def compute_rotor_forces_and_torques(self, omega_vec):
        """
        Calculates individual rotor thrust forces F_i and drag torques M_z,i.
        Rotor rotation directions: Motor 1 (-z), Motor 2 (+z), Motor 3 (-z), Motor 4 (+z).
        """
        omega_vec = np.clip(np.asarray(omega_vec, dtype=float), self.omega_min, self.omega_max)
        omega_sq = omega_vec**2

        rotor_forces = self.kf * omega_sq
        # Drag torque reaction on body
        rotor_torques = np.array([
            -self.km * omega_sq[0],
             self.km * omega_sq[1],
            -self.km * omega_sq[2],
             self.km * omega_sq[3]
        ], dtype=float)

        return rotor_forces, rotor_torques

    def compute_thrust_and_torques(self, omega_vec):
        """
        Forward mapping: motor speeds -> net total thrust F and body torques tau.
        """
        omega_sq = np.clip(np.asarray(omega_vec, dtype=float), self.omega_min, self.omega_max)**2
        wrench = self.M @ omega_sq
        return wrench[0], wrench[1:4]

    def allocate_and_limit(self, total_thrust, torques):
        """
        Full allocation step: desired (F, tau) -> motor speeds -> actual applied (F_act, tau_act).
        """
        omega_vec, is_saturated = self.compute_motor_speeds(total_thrust, torques)
        actual_thrust, actual_torques = self.compute_thrust_and_torques(omega_vec)
        return omega_vec, actual_thrust, actual_torques, is_saturated
