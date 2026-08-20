"""
Cascaded Quadrotor Controller:
Combines Outer-Loop PID Position Controller with Inner-Loop Paper 1 Quaternion Attitude Controller.
Computes world-frame thrust and body-frame control torques for 6-DoF trajectory tracking.
"""

import numpy as np
from src.quaternion_math import quat_normalize, quat_to_rot_matrix
from src.quaternion_controller import NonlinearP2QuaternionController

class CascadedQuadrotorTracker:
    def __init__(self, mass: float = 0.5, gravity: float = 9.81, dt: float = 0.005):
        self.m = mass
        self.g = gravity
        self.dt = dt

        # Outer-loop Position PID Gains
        self.Kp_pos = np.array([2.0, 2.0, 3.0])
        self.Kd_pos = np.array([1.5, 1.5, 2.0])
        self.Ki_pos = np.array([0.05, 0.05, 0.1])

        self.e_p_integral = np.zeros(3)

        # Inner-loop Paper 1 Quaternion Controller
        self.att_controller = NonlinearP2QuaternionController(
            Pq=20.0,
            Pw=4.0,
            max_torque=4.0,
            noise_amp=0.0
        )

    def reset(self):
        self.e_p_integral = np.zeros(3)

    def compute_control(self, p_target: np.ndarray, v_target: np.ndarray, a_target: np.ndarray, state: dict) -> tuple:
        """
        Computes control torque tau and scalar thrust force F_thrust.
        """
        p_curr = state['pos']
        v_curr = state['vel']
        q_curr = state['quat']
        w_world = state['omega']

        # Position and Velocity Errors
        e_p = p_target - p_curr
        e_v = v_target - v_curr

        self.e_p_integral += e_p * self.dt
        self.e_p_integral = np.clip(self.e_p_integral, -1.0, 1.0)

        # Desired Translational Acceleration Command
        a_des = a_target + self.Kp_pos * e_p + self.Kd_pos * e_v + self.Ki_pos * self.e_p_integral

        # Acceleration Bounds for Tilt Safety
        a_des[0] = np.clip(a_des[0], -2.0, 2.0)
        a_des[1] = np.clip(a_des[1], -2.0, 2.0)
        a_des[2] = np.clip(a_des[2], -2.0, 2.0)

        # Desired World Thrust Vector: F_world = m * (a_des + g * z_hat)
        F_world = self.m * (a_des + np.array([0.0, 0.0, self.g]))

        R_curr = quat_to_rot_matrix(q_curr)
        cos_tilt = max(0.3, R_curr[2, 2])

        # Vertical Thrust Force (compensated for tilt angle)
        thrust_force = self.m * (a_des[2] + self.g) / cos_tilt
        thrust_force = np.clip(thrust_force, 0.1, 12.0)

        # Desired Body Z-Axis Direction
        z_b_des = F_world / np.linalg.norm(F_world)

        x_c = np.array([1.0, 0.0, 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        if np.linalg.norm(y_b_des) < 1e-6:
            y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            y_b_des /= np.linalg.norm(y_b_des)

        x_b_des = np.cross(y_b_des, z_b_des)
        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])

        # Convert R_des to desired quaternion q_ref
        tr = np.trace(R_des)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q0 = 0.25 * S
            q1 = (R_des[2, 1] - R_des[1, 2]) / S
            q2 = (R_des[0, 2] - R_des[2, 0]) / S
            q3 = (R_des[1, 0] - R_des[0, 1]) / S
        else:
            q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0
        q_ref = quat_normalize(np.array([q0, q1, q2, q3]))

        # Transform world angular velocity to body frame: w_body = R^T * w_world
        w_body = R_curr.T @ w_world

        # Compute body control torque via Paper 1 Quaternion Attitude Controller
        tau_body, _, _ = self.att_controller.compute_torque(q_ref, q_curr, w_body, add_noise=False)

        # Transform body torque to world torque for MuJoCo freejoint qfrc_applied
        tau_world = R_curr @ tau_body

        return tau_world, thrust_force
