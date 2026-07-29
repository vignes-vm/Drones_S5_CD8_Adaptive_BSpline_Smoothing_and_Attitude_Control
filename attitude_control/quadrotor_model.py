import numpy as np
from .quaternion import Quaternion

class QuadrotorModel:
    """
    Quadrotor dynamics model.
    Paper 2 — Equation (18)
    """

    def __init__(self):
        # Paper 2 inertia values
        self.Ixx = 6.5e-4
        self.Iyy = 6.5e-4
        self.Izz = 1.2e-3
        self.mass = 1.0
        self.g    = 9.81

        self.I     = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.I_inv = np.linalg.inv(self.I)

        # State
        self.q        = Quaternion()
        self.omega    = np.zeros(3)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

    def reset(self, position=None, euler=None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        if euler is not None:
            self.q = Quaternion.from_euler(*euler)
        else:
            self.q = Quaternion()
        self.omega    = np.zeros(3)
        self.velocity = np.zeros(3)

    def step_attitude_only(self, tau, dt):
        """
        Integrate ONLY rotational dynamics.
        Paper 2 Equation (18):
        q_dot    = -1/2 * [0,ω]^T ⊗ q
        omega_dot = I⁻¹τ - I⁻¹(ω × Iω)
        """
        tau = np.clip(tau, -4.0, 4.0)

        def derivatives(q_arr, omega_arr):
            omega_quat = Quaternion(0,
                                    omega_arr[0],
                                    omega_arr[1],
                                    omega_arr[2])
            q_temp = Quaternion(*q_arr)
            dq     = -0.5 * omega_quat.multiply(q_temp).q
            gyro   = np.cross(omega_arr,
                               self.I @ omega_arr)
            domega = self.I_inv @ (tau - gyro)
            return dq, domega

        # RK4
        q0 = self.q.q.copy()
        w0 = self.omega.copy()

        k1q, k1w = derivatives(q0, w0)
        k2q, k2w = derivatives(
            q0 + 0.5*dt*k1q, w0 + 0.5*dt*k1w)
        k3q, k3w = derivatives(
            q0 + 0.5*dt*k2q, w0 + 0.5*dt*k2w)
        k4q, k4w = derivatives(
            q0 + dt*k3q,     w0 + dt*k3w)

        new_q = q0 + (dt/6)*(k1q + 2*k2q + 2*k3q + k4q)
        new_w = w0 + (dt/6)*(k1w + 2*k2w + 2*k3w + k4w)

        self.q = Quaternion(*new_q)
        self.q.normalize()
        self.omega = new_w