import numpy as np
from .quaternion import Quaternion
from .motor_model import MotorModel


class QuadrotorModel:
    """
    Quadrotor dynamics model.
    Paper 2 — Equation (18):
      q_dot     = 1/2 * q ⊗ [0, ω]^T
      omega_dot = I⁻¹(τ - ω × (Iω))
    """

    def __init__(self, enable_motor_model=False, motor_model=None):
        # Paper 2 physical inertia values
        self.Ixx = 6.5e-4
        self.Iyy = 6.5e-4
        self.Izz = 1.2e-3
        self.mass = 1.0
        self.g    = 9.81

        self.I     = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.I_inv = np.linalg.inv(self.I)

        # Actuator / Motor Model
        self.enable_motor_model = enable_motor_model
        self.motor_model = motor_model if motor_model is not None else MotorModel()

        # State variables
        self.q             = Quaternion()
        self.omega         = np.zeros(3)
        self.position      = np.zeros(3)
        self.velocity      = np.zeros(3)
        self.motor_speeds  = np.zeros(4)
        self.applied_torque = np.zeros(3)
        self.motor_saturated = False

    def reset(self, position=None, euler=None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        if euler is not None:
            self.q = Quaternion.from_euler(*euler)
        else:
            self.q = Quaternion()
        self.omega           = np.zeros(3)
        self.velocity        = np.zeros(3)
        self.motor_speeds    = np.zeros(4)
        self.applied_torque  = np.zeros(3)
        self.motor_saturated = False

    def step_attitude_only(self, tau, dt, total_thrust=None, tau_dist=None):
        """
        Integrate rotational dynamics using RK4 integration.
        Optionally uses motor model for actuator mixing and torque limits.
        """
        tau_cmd = np.clip(np.asarray(tau, dtype=float), -4.0, 4.0)

        if self.enable_motor_model:
            thrust_cmd = total_thrust if total_thrust is not None else (self.mass * self.g)
            motor_w, act_thrust, act_tau, saturated = self.motor_model.allocate_and_limit(
                thrust_cmd, tau_cmd
            )
            self.motor_speeds    = motor_w
            self.applied_torque  = act_tau
            self.motor_saturated = saturated
            tau_effective        = act_tau.copy()
        else:
            tau_effective        = tau_cmd.copy()
            self.applied_torque  = tau_cmd.copy()
            self.motor_saturated = False

        if tau_dist is not None:
            tau_effective += np.asarray(tau_dist, dtype=float)

        def derivatives(q_arr, omega_arr):
            omega_quat = Quaternion(0, omega_arr[0], omega_arr[1], omega_arr[2])
            q_temp     = Quaternion(*q_arr)
            dq         = -0.5 * omega_quat.multiply(q_temp).q
            gyro       = np.cross(omega_arr, self.I @ omega_arr)
            domega     = self.I_inv @ (tau_effective - gyro)
            return dq, domega

        # RK4 Integration
        q0 = self.q.q.copy()
        w0 = self.omega.copy()

        k1q, k1w = derivatives(q0, w0)
        k2q, k2w = derivatives(q0 + 0.5*dt*k1q, w0 + 0.5*dt*k1w)
        k3q, k3w = derivatives(q0 + 0.5*dt*k2q, w0 + 0.5*dt*k2w)
        k4q, k4w = derivatives(q0 + dt*k3q,     w0 + dt*k3w)

        new_q = q0 + (dt/6.0)*(k1q + 2*k2q + 2*k3q + k4q)
        new_w = w0 + (dt/6.0)*(k1w + 2*k2w + 2*k3w + k4w)

        self.q = Quaternion(*new_q)
        self.q.normalize()
        self.omega = new_w