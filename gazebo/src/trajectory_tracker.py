"""
Gazebo Trajectory Tracking Guidance & Controller.
Integrates:
  1. Outer loop: 3D Position PD controller computing commanded collective thrust and desired tilt quaternion.
  2. Inner loop: Paper 2 Singularity-Free Quaternion P^2 attitude controller.
"""
import numpy as np
from .quaternion_math import QuaternionMath


class GazeboTrajectoryTracker:

    def __init__(self, mass=1.0, g=9.81, Pq=2.5, Pw=0.08, tau_limit=1.0,
                 Kp_pos=(1.2, 1.2, 2.5), Kd_pos=(1.5, 1.5, 2.0), max_tilt_deg=35.0):
        """
        mass: Vehicle mass [kg]
        g: Gravitational acceleration [m/s^2]
        Pq: Attitude proportional gain (tuned for 500Hz discrete control with I=6.5e-4)
        Pw: Angular rate damping gain
        tau_limit: Torque saturation bound [Nm]
        Kp_pos, Kd_pos: Position loop PD gains [X, Y, Z]
        max_tilt_deg: Maximum allowable roll/pitch tilt angle
        """
        self.mass = mass
        self.g = g
        self.Pq = Pq
        self.Pw = Pw
        self.tau_limit = tau_limit

        self.Kp_pos = np.array(Kp_pos, dtype=float)
        self.Kd_pos = np.array(Kd_pos, dtype=float)
        self.max_tilt = np.radians(max_tilt_deg)
        self.prev_q_des = np.array([1.0, 0.0, 0.0, 0.0])

    def compute_position_control(self, pos_target, pos_current, vel_target=None, vel_current=None, yaw_target=0.0):
        """
        Outer-loop position controller.
        Computes commanded collective thrust (N) and target attitude quaternion q_des.
        """
        if vel_target is None:
            vel_target = np.zeros(3)
        if vel_current is None:
            vel_current = np.zeros(3)

        pos_err = np.asarray(pos_target, dtype=float) - np.asarray(pos_current, dtype=float)
        vel_err = np.asarray(vel_target, dtype=float) - np.asarray(vel_current, dtype=float)

        # Virtual acceleration: a_des = Kp * e_p + Kd * e_v + g * z_hat
        a_fb = self.Kp_pos * pos_err + self.Kd_pos * vel_err
        a_total = a_fb + np.array([0.0, 0.0, self.g])

        norm_a = np.linalg.norm(a_total)
        if norm_a < 1e-4:
            z_b_des = np.array([0.0, 0.0, 1.0])
        else:
            z_b_des = a_total / norm_a

        # Tilt angle clamping (limit tilt to max_tilt for safety)
        cos_limit = np.cos(self.max_tilt)
        if z_b_des[2] < cos_limit:
            z_b_des[2] = cos_limit
            z_b_des = z_b_des / np.linalg.norm(z_b_des)

        # Commanded collective thrust
        thrust_cmd = self.mass * np.dot(a_total, z_b_des)
        thrust_cmd = np.clip(thrust_cmd, 0.1, 30.0)

        # Construct desired orthonormal rotation matrix R_des = [x_b, y_b, z_b]
        heading = np.array([np.cos(yaw_target), np.sin(yaw_target), 0.0])
        y_b_des = np.cross(z_b_des, heading)
        norm_y = np.linalg.norm(y_b_des)
        if norm_y < 1e-4:
            y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            y_b_des = y_b_des / norm_y

        x_b_des = np.cross(y_b_des, z_b_des)
        x_b_des = x_b_des / np.linalg.norm(x_b_des)

        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])
        q_des = QuaternionMath.from_rotation_matrix(R_des)

        if np.dot(q_des, self.prev_q_des) < 0.0:
            q_des = -q_des
        self.prev_q_des = q_des.copy()

        return thrust_cmd, q_des

    def compute_attitude_control(self, q_des, q_current, omega_current):
        """
        Inner-loop Paper 2 Quaternion P^2 Attitude Controller in Body Frame:
          q_err_body = q_current^-1 x q_des (with shortest-path sign correction)
          tau = clamp(Pq * q_err_body[1:4] - Pw * omega_current, -tau_limit, tau_limit)
        """
        # Body frame error: rotation from current to desired in body coordinates
        q_err_body = QuaternionMath.multiply(QuaternionMath.conjugate(q_current), q_des)
        if q_err_body[0] < 0.0:
            q_err_body = -q_err_body

        axis_err = q_err_body[1:4]  # Vector part [q1, q2, q3]

        tau = self.Pq * axis_err - self.Pw * np.asarray(omega_current, dtype=float)
        tau_saturated = np.clip(tau, -self.tau_limit, self.tau_limit)

        return tau_saturated, axis_err, q_err_body

    def compute_full_control(self, pos_target, pos_current, q_current, omega_current,
                            vel_target=None, vel_current=None, tangent=None, yaw_target=None):
        """
        Cascaded evaluation: outer position loop -> inner quaternion P^2 loop.
        Computes projected thrust along current vehicle body Z-axis and attitude torques.
        """
        if vel_target is None:
            vel_target = np.zeros(3)
        if vel_current is None:
            vel_current = np.zeros(3)

        pos_err = np.asarray(pos_target, dtype=float) - np.asarray(pos_current, dtype=float)
        vel_err = np.asarray(vel_target, dtype=float) - np.asarray(vel_current, dtype=float)

        # Virtual acceleration: a_cmd = Kp * e_p + Kd * e_v + g * z_hat
        a_fb = self.Kp_pos * pos_err + self.Kd_pos * vel_err
        a_total = a_fb + np.array([0.0, 0.0, self.g])

        norm_a = np.linalg.norm(a_total)
        if norm_a < 1e-4:
            z_b_des = np.array([0.0, 0.0, 1.0])
        else:
            z_b_des = a_total / norm_a

        # Tilt angle clamping (limit tilt to max_tilt for safe flight)
        cos_limit = np.cos(self.max_tilt)
        if z_b_des[2] < cos_limit:
            z_b_des[2] = cos_limit
            z_b_des = z_b_des / np.linalg.norm(z_b_des)

        # Extract horizontal heading
        if yaw_target is not None:
            heading = np.array([np.cos(yaw_target), np.sin(yaw_target), 0.0])
        elif tangent is not None and np.linalg.norm(tangent) > 1e-4:
            heading = np.array(tangent, dtype=float)
            heading[2] = 0.0
            h_norm = np.linalg.norm(heading)
            heading = heading / h_norm if h_norm > 1e-4 else np.array([1.0, 0.0, 0.0])
        else:
            heading = np.array([1.0, 0.0, 0.0])

        y_b_des = np.cross(z_b_des, heading)
        norm_y = np.linalg.norm(y_b_des)
        y_b_des = np.array([0.0, 1.0, 0.0]) if norm_y < 1e-4 else y_b_des / norm_y

        x_b_des = np.cross(y_b_des, z_b_des)
        x_b_des = x_b_des / np.linalg.norm(x_b_des)

        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])
        q_des = QuaternionMath.from_rotation_matrix(R_des)

        if np.dot(q_des, self.prev_q_des) < 0.0:
            q_des = -q_des
        self.prev_q_des = q_des.copy()

        # Inner loop attitude torque in body frame
        torques_cmd, axis_err, _ = self.compute_attitude_control(q_des, q_current, omega_current)

        # Projected thrust scalar along current quadrotor body Z-axis
        R_current = QuaternionMath.to_rotation_matrix(q_current)
        z_b_curr = R_current[:, 2]
        thrust_cmd = self.mass * np.dot(a_total, z_b_curr)
        thrust_cmd = np.clip(thrust_cmd, 0.2 * self.mass * self.g, 2.5 * self.mass * self.g)

        return thrust_cmd, torques_cmd, q_des, axis_err
