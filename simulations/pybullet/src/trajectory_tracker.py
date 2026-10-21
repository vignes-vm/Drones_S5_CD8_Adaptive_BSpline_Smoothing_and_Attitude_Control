"""
Combined Guidance & Control Trajectory Tracker for PyBullet Simulation.
Integrates Outer-Loop 3D Position PD Controller and Inner-Loop Paper 2 Quaternion P^2 Controller.
Translates position/velocity errors into target thrust F and target quaternion q_ref.
"""
import numpy as np
from .quaternion_math import QuaternionMath


class PyBulletTrajectoryTracker:

    def __init__(self, mass=1.0, g=9.81, Pq=2.5, Pw=0.08,
                 Kp_pos=None, Kd_pos=None, tau_limit=0.5):

        self.mass = float(mass)
        self.g    = float(g)

        # Paper 2 Attitude Controller Gains (Tuned for discrete 500Hz-1kHz sampling & quadrotor inertias)
        self.Pq = float(Pq)
        self.Pw = float(Pw)
        self.tau_limit = float(tau_limit)

        # 3D Position PD Gains
        self.Kp_pos = np.array(Kp_pos if Kp_pos is not None else [0.8, 0.8, 2.0], dtype=float)
        self.Kd_pos = np.array(Kd_pos if Kd_pos is not None else [1.0, 1.0, 1.8], dtype=float)

    def compute_position_control(self, p_des, v_des, p_meas, v_meas, a_des=None):
        """
        Outer Loop Position Controller:
        Translates position & velocity tracking errors into desired acceleration vector a_cmd
        and desired body Z-axis orientation vector z_body_des.
        """
        p_err = np.asarray(p_des, dtype=float) - np.asarray(p_meas, dtype=float)
        v_err = np.asarray(v_des, dtype=float) - np.asarray(v_meas, dtype=float)
        acc_feedforward = np.asarray(a_des, dtype=float) if a_des is not None else np.zeros(3)

        # Desired acceleration vector: a_cmd = a_ff + Kp*p_err + Kd*v_err + g*z_hat
        a_cmd = acc_feedforward + self.Kp_pos * p_err + self.Kd_pos * v_err + np.array([0.0, 0.0, self.g])

        a_norm = np.linalg.norm(a_cmd)
        if a_norm < 1e-6:
            a_cmd = np.array([0.0, 0.0, self.g])
            a_norm = self.g

        z_body_des = a_cmd / a_norm

        # Limit maximum tilt angle to 30 degrees (cos(30 deg) = 0.866) for flight safety
        if z_body_des[2] < 0.866:
            z_body_des[2] = 0.866
            z_body_des = z_body_des / np.linalg.norm(z_body_des)

        return a_cmd, z_body_des

    def compute_reference_quaternion(self, z_body_des, tangent=None, yaw_des=None):
        """
        Constructs desired rotation matrix R_des = [x_b | y_b | z_b] and reference quaternion q_ref.
        X-body axis aligns with path tangent C'(u) or specified yaw angle.
        """
        z_b = z_body_des / np.linalg.norm(z_body_des)

        if yaw_des is not None:
            heading = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        elif tangent is not None and np.linalg.norm(tangent) > 1e-4:
            heading = np.array(tangent, dtype=float)
            heading[2] = 0.0  # Project onto horizontal plane
            if np.linalg.norm(heading) < 1e-4:
                heading = np.array([1.0, 0.0, 0.0])
            else:
                heading = heading / np.linalg.norm(heading)
        else:
            heading = np.array([1.0, 0.0, 0.0])

        # Orthonormal frame construction
        y_b = np.cross(z_b, heading)
        y_norm = np.linalg.norm(y_b)
        if y_norm < 1e-6:
            y_b = np.array([0.0, 1.0, 0.0])
        else:
            y_b = y_b / y_norm

        x_b = np.cross(y_b, z_b)
        x_b = x_b / np.linalg.norm(x_b)

        R_des = np.column_stack([x_b, y_b, z_b])
        q_ref = QuaternionMath.from_rotation_matrix(R_des)
        return q_ref

    def compute_attitude_control(self, q_ref, q_meas, omega_meas):
        """
        Inner Loop Paper 2 Quaternion P^2 Attitude Controller in Body Frame:
        q_err_body = q_meas^-1 ⊗ q_ref (with shortest path sign correction)
        tau = sat(Pq * q_err_body[1:4] - Pw * omega_meas)
        """
        q_err_body = QuaternionMath.multiply(QuaternionMath.conjugate(q_meas), q_ref)
        if q_err_body[0] < 0.0:
            q_err_body = -q_err_body

        axis_err = q_err_body[1:4]  # Body frame error vector [q1, q2, q3]

        tau = self.Pq * axis_err - self.Pw * np.asarray(omega_meas, dtype=float)
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        return tau, axis_err

    def step_control(self, p_des, v_des, p_meas, v_meas, q_meas, omega_meas, tangent=None, a_des=None):
        """
        Full guidance + control step:
        Returns total_thrust_cmd, torques_cmd, q_ref, axis_err, pos_err_mag.
        """
        a_cmd, z_body_des = self.compute_position_control(p_des, v_des, p_meas, v_meas, a_des)
        q_ref = self.compute_reference_quaternion(z_body_des, tangent=tangent)
        torques_cmd, axis_err = self.compute_attitude_control(q_ref, q_meas, omega_meas)

        # Projected thrust scalar along current quadrotor body Z-axis
        R_meas = QuaternionMath.to_rotation_matrix(q_meas)
        z_b_curr = R_meas[:, 2]
        thrust_cmd = self.mass * np.dot(a_cmd, z_b_curr)
        thrust_cmd = np.clip(thrust_cmd, 0.2 * self.mass * self.g, 2.5 * self.mass * self.g)

        pos_err_mag = np.linalg.norm(np.asarray(p_des) - np.asarray(p_meas))
        return thrust_cmd, torques_cmd, q_ref, axis_err, pos_err_mag

