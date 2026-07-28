import numpy as np
from .quaternion import Quaternion

class QuadrotorModel:

    def __init__(self):
        self.Ixx = 6.5e-4   # kg.m²
        self.Iyy = 6.5e-4   # kg.m²
        self.Izz = 1.2e-3   # kg.m²
        self.mass = 1.0     # kg (approximate)

        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.I_inv = np.linalg.inv(self.I)

        # State variables
        self.q = Quaternion()            # attitude quaternion
        self.omega = np.zeros(3)         # angular velocity [wx, wy, wz]
        self.position = np.zeros(3)      # world position [x, y, z]
        self.velocity = np.zeros(3)      # world velocity
        self.history = {
            't': [], 'q': [], 'omega': [],
            'position': [], 'euler': []
        }

    def reset(self, position=None, euler=None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        if euler is not None:
            self.q = Quaternion.from_euler(*euler)
        self.omega = np.zeros(3)
        self.velocity = np.zeros(3)

    def quaternion_derivative(self):
        omega_quat = Quaternion(0, self.omega[0],
                                   self.omega[1],
                                   self.omega[2])
        q_dot_q = omega_quat.multiply(self.q)
        return -0.5 * q_dot_q.q

    def omega_derivative(self, tau):
        gyroscopic = np.cross(self.omega, self.I @ self.omega)
        omega_dot = self.I_inv @ (tau - gyroscopic)
        return omega_dot

    def step(self, tau, dt, noise_std=0.0):
        if noise_std > 0:
            tau = tau + np.random.normal(0, noise_std, 3)

        tau = np.clip(tau, -4.0, 4.0)

        def derivatives(q_arr, omega_arr):
            # Temporary state
            q_temp = Quaternion(*q_arr)
            omega_quat = Quaternion(0, omega_arr[0],
                                      omega_arr[1],
                                      omega_arr[2])
            dq = -0.5 * omega_quat.multiply(q_temp).q
            gyro = np.cross(omega_arr, self.I @ omega_arr)
            domega = self.I_inv @ (tau - gyro)
            return dq, domega

        q0 = self.q.q.copy()
        w0 = self.omega.copy()

        k1q, k1w = derivatives(q0, w0)
        k2q, k2w = derivatives(q0 + 0.5*dt*k1q, w0 + 0.5*dt*k1w)
        k3q, k3w = derivatives(q0 + 0.5*dt*k2q, w0 + 0.5*dt*k2w)
        k4q, k4w = derivatives(q0 + dt*k3q,     w0 + dt*k3w)

        new_q = q0 + (dt/6)*(k1q + 2*k2q + 2*k3q + k4q)
        new_w = w0 + (dt/6)*(k1w + 2*k2w + 2*k3w + k4w)

        self.q = Quaternion(*new_q)
        self.q.normalize()
        self.omega = new_w

    def log(self, t):
        self.history['t'].append(t)
        self.history['q'].append(self.q.q.copy())
        self.history['omega'].append(self.omega.copy())
        self.history['euler'].append(self.q.to_euler())