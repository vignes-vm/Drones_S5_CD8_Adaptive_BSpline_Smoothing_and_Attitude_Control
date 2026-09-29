r"""
Paper 2: Full Quaternion-Based Attitude & Position Controller.
Implements:
1. Position PID tracking loop -> Virtual acceleration command.
2. Geometric Reference Attitude Generation -> Desired unit quaternion q_ref.
3. Paper 2 Quaternion P^2 Attitude Control:
      q_err = q_meas^* \otimes q_ref = [e_w, e_v]
      tau_cmd = 2 * K_q * sgn(e_w) * e_v - K_omega * omega_body
4. Actuator Mixer Matrix -> Individual rotor speed commands [w1, w2, w3, w4].
"""

from typing import Dict, Tuple
import numpy as np
from .quaternion_math import (
    quat_error_body,
    quat_norm,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    quat_conj,
    quat_mult
)
from .motor_model import MotorModel


class QuaternionP2Controller:
    """
    Unified Paper 2 Quadrotor Trajectory & Attitude Controller.
    """
    def __init__(
        self,
        mass: float = 1.0,               # Quadrotor mass (kg)
        gravity: float = 9.81,           # Gravity acceleration (m/s^2)
        inertia: np.ndarray = np.diag([0.0082, 0.0082, 0.0149]), # Inertia tensor (kg*m^2)
        motor_model: MotorModel = None,
        # Position gains
        kp_pos: np.ndarray = np.array([4.0, 4.0, 6.5]),
        kd_pos: np.ndarray = np.array([3.2, 3.2, 4.5]),
        ki_pos: np.ndarray = np.array([0.4, 0.4, 1.0]),
        # Attitude gains (Paper 2 P^2 Quaternion gains)
        kq_att: np.ndarray = np.array([12.0, 12.0, 6.0]),
        komega_att: np.ndarray = np.array([2.4, 2.4, 1.5])
    ):
        self.mass = mass
        self.gravity = gravity
        self.inertia = inertia
        self.motor = motor_model if motor_model is not None else MotorModel()

        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.ki_pos = ki_pos
        self.kq_att = kq_att
        self.komega_att = komega_att

        # Integrator state
        self.pos_integral = np.zeros(3, dtype=np.float64)
        self.integral_limit = 3.0

    def reset(self) -> None:
        """Resets integral errors and motor state."""
        self.pos_integral = np.zeros(3, dtype=np.float64)
        self.motor.reset()

    def compute_control(
        self,
        pos_meas: np.ndarray,
        vel_meas: np.ndarray,
        quat_meas: np.ndarray,
        omega_meas: np.ndarray,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray,
        acc_ref: np.ndarray,
        yaw_ref: float,
        dt: float
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes full position and Paper 2 quaternion P^2 attitude control loop.
        Returns:
            omega_cmd: (4,) desired motor speeds
            total_thrust: scalar desired collective thrust (N)
            tau_cmd: (3,) commanded body torques (N*m)
            q_ref: (4,) desired attitude quaternion
            q_err: (4,) body-frame quaternion error [e_w, e_x, e_y, e_z]
        """
        # 1. Position Tracking Error
        e_pos = pos_ref - pos_meas
        e_vel = vel_ref - vel_meas
        self.pos_integral += e_pos * dt
        self.pos_integral = np.clip(self.pos_integral, -self.integral_limit, self.integral_limit)

        # 2. Virtual Acceleration Command
        a_cmd = acc_ref + self.kp_pos * e_pos + self.kd_pos * e_vel + self.ki_pos * self.pos_integral
        # Add gravity compensation
        a_cmd += np.array([0.0, 0.0, self.gravity])

        # Clamp max tilt acceleration for aerodynamic stability under heavy wind
        a_xy_norm = np.linalg.norm(a_cmd[:2])
        max_tilt_acc = self.gravity * np.tan(np.radians(50.0))  # max 50 deg tilt
        if a_xy_norm > max_tilt_acc:
            a_cmd[:2] = a_cmd[:2] * (max_tilt_acc / a_xy_norm)

        # 3. Collective Thrust Command
        # Project commanded acceleration along body z-axis
        R_meas = quat_to_rot_matrix(quat_meas)
        z_B_meas = R_meas[:, 2]
        total_thrust = float(self.mass * np.dot(a_cmd, z_B_meas))
        total_thrust = max(total_thrust, 0.1)

        # 4. Desired Attitude Quaternion q_ref Generation
        z_B_cmd = a_cmd / np.linalg.norm(a_cmd)
        
        # Desired heading vector in horizontal plane
        x_C = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])
        
        # Intermediate y-axis perpendicular to z_B and x_C
        y_B_cmd = np.cross(z_B_cmd, x_C)
        y_B_norm = np.linalg.norm(y_B_cmd)
        if y_B_norm < 1e-6:
            y_B_cmd = np.array([0.0, 1.0, 0.0])
        else:
            y_B_cmd = y_B_cmd / y_B_norm

        # Orthogonal x-axis
        x_B_cmd = np.cross(y_B_cmd, z_B_cmd)

        R_cmd = np.column_stack([x_B_cmd, y_B_cmd, z_B_cmd])
        q_ref = rot_matrix_to_quat(R_cmd)

        # 5. Paper 2 Quaternion P^2 Attitude Error in Body Frame
        # q_err = q_meas^* \otimes q_ref = [e_w, e_x, e_y, e_z]
        q_err = quat_error_body(quat_meas, q_ref)
        e_w = q_err[0]
        e_v = q_err[1:4]  # vector part [e_x, e_y, e_z]

        # Commanded body torques
        tau_cmd = 2.0 * self.kq_att * np.sign(e_w) * e_v - self.komega_att * omega_meas

        # 6. Inverse Mixer Matrix: (T_cmd, tau_cmd) -> omega_cmd^2
        c_T = self.motor.c_T
        c_Q = self.motor.c_Q
        d = self.motor.arm_length

        # Plus configuration mixer:
        # T   = c_T * (w1^2 + w2^2 + w3^2 + w4^2)
        # tx  = d * c_T * (w2^2 - w4^2)
        # ty  = d * c_T * (w3^2 - w1^2)
        # tz  = c_Q * (w1^2 - w2^2 + w3^2 - w4^2)
        
        A_mix = np.array([
            [c_T,      c_T,      c_T,      c_T],
            [0.0,      d * c_T,  0.0,     -d * c_T],
            [-d * c_T, 0.0,      d * c_T,  0.0],
            [c_Q,     -c_Q,      c_Q,     -c_Q]
        ], dtype=np.float64)

        b_vec = np.array([total_thrust, tau_cmd[0], tau_cmd[1], tau_cmd[2]])
        
        try:
            omega_sq = np.linalg.solve(A_mix, b_vec)
        except np.linalg.LinAlgError:
            omega_sq = np.linalg.pinv(A_mix) @ b_vec

        omega_sq = np.maximum(omega_sq, 0.0)
        omega_cmd = np.sqrt(omega_sq)
        omega_cmd = np.clip(omega_cmd, self.motor.omega_min, self.motor.omega_max)

        return omega_cmd, total_thrust, tau_cmd, q_ref, q_err
