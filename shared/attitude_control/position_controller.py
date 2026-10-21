"""
Position and Altitude Controller for Quadrotor UAV.
Outer-loop controller converting 3D position & velocity errors into
desired total thrust magnitude F_thrust and desired attitude quaternion q_des.
"""
import numpy as np
from .quaternion import Quaternion


class PositionController:
    """
    Cascading Outer-Loop Position and Altitude Controller.
    
    Computes desired acceleration vector:
        a_des = Kp * (p_ref - p) + Kd * (v_ref - v) + [0, 0, g]^T
    
    From a_des, extracts:
      1. Total thrust magnitude: F_thrust = mass * ||a_des||
      2. Desired body orientation matrix R_des -> q_des
    """

    def __init__(self, Kp_pos=(2.0, 2.0, 4.0), Kd_pos=(1.5, 1.5, 2.5),
                 mass=1.0, g=9.81, max_tilt_deg=45.0, max_thrust=30.0):
        self.Kp = np.array(Kp_pos, dtype=float)
        self.Kd = np.array(Kd_pos, dtype=float)
        self.mass = float(mass)
        self.g = float(g)
        self.max_tilt_rad = np.radians(max_tilt_deg)
        self.max_thrust = float(max_thrust)

    def compute(self, pos_ref, pos, vel_ref=None, vel=None, yaw_ref=0.0):
        """
        Compute total thrust and desired quaternion for target position/velocity.

        Parameters:
        -----------
        pos_ref : array-like (3,)
            Desired 3D position [x, y, z] in world frame.
        pos : array-like (3,)
            Current 3D position [x, y, z] in world frame.
        vel_ref : array-like (3,), optional
            Desired 3D velocity [vx, vy, vz]. Defaults to zeros.
        vel : array-like (3,), optional
            Current 3D velocity [vx, vy, vz]. Defaults to zeros.
        yaw_ref : float, optional
            Desired yaw angle (rad). Defaults to 0.0.

        Returns:
        --------
        total_thrust : float
            Required total thrust force magnitude (N).
        q_des : Quaternion
            Desired target attitude quaternion.
        """
        pos_ref = np.asarray(pos_ref, dtype=float)
        pos     = np.asarray(pos, dtype=float)
        vel_ref = np.zeros(3) if vel_ref is None else np.asarray(vel_ref, dtype=float)
        vel     = np.zeros(3) if vel is None else np.asarray(vel, dtype=float)

        # Position and velocity errors
        pos_err = pos_ref - pos
        vel_err = vel_ref - vel

        # Desired acceleration in world frame (including gravity compensation)
        a_cmd = self.Kp * pos_err + self.Kd * vel_err
        a_des = a_cmd + np.array([0.0, 0.0, self.g])

        # Limit tilt angle away from vertical [0, 0, 1] to prevent inverted flight
        a_norm = np.linalg.norm(a_des)
        if a_norm < 1e-6:
            a_des = np.array([0.0, 0.0, self.g])
            a_norm = self.g

        z_b = a_des / a_norm

        # Check maximum tilt limit
        cos_tilt = np.dot(z_b, np.array([0.0, 0.0, 1.0]))
        min_cos_tilt = np.cos(self.max_tilt_rad)

        if cos_tilt < min_cos_tilt:
            # Clamp z_b within max tilt cone
            xy_norm = np.hypot(z_b[0], z_b[1])
            if xy_norm > 1e-6:
                sin_max = np.sin(self.max_tilt_rad)
                z_b[0] = (z_b[0] / xy_norm) * sin_max
                z_b[1] = (z_b[1] / xy_norm) * sin_max
                z_b[2] = min_cos_tilt
                z_b = z_b / np.linalg.norm(z_b)

        # Desired thrust magnitude
        total_thrust = self.mass * np.dot(a_des, z_b)
        total_thrust = np.clip(total_thrust, 0.0, self.max_thrust)

        # Build orthonormal rotation matrix from z_b and yaw_ref
        x_c = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        
        y_b = np.cross(z_b, x_c)
        y_norm = np.linalg.norm(y_b)

        if y_norm < 1e-6:
            # Singularity fallback if z_b is parallel to x_c
            x_c_alt = np.array([0.0, 1.0, 0.0])
            y_b = np.cross(z_b, x_c_alt)
            y_norm = np.linalg.norm(y_b)
            if y_norm < 1e-6:
                y_b = np.array([0.0, 1.0, 0.0])
                y_norm = 1.0

        y_b = y_b / y_norm
        x_b = np.cross(y_b, z_b)
        x_b = x_b / np.linalg.norm(x_b)

        # R matrix columns are body axes in world frame
        R_des = np.column_stack([x_b, y_b, z_b])

        q_des = Quaternion.from_rotation_matrix(R_des)
        return total_thrust, q_des
