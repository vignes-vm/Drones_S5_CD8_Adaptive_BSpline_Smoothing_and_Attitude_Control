r"""
6-DOF Quadrotor Dynamics and Numerical Integrator.
Solves the Newton-Euler rigid body equations:
    m * \ddot{p} = R * [0, 0, T]^T + [0, 0, -m*g]^T + F_aero
    I * \dot{\omega} = \tau_motors - \omega \times (I * \omega) + \tau_aero
    \dot{q} = 0.5 * q \otimes [0, \omega]
"""

from typing import Tuple
import numpy as np
from .quaternion_math import quat_norm, quat_to_rot_matrix, quat_mult


class QuadrotorDynamics:
    """
    Simulates high-precision 6-DOF quadrotor physical state:
    State vector: [p(3), v(3), q(4), omega(3)]
    """
    def __init__(
        self,
        mass: float = 1.0,
        gravity: float = 9.81,
        inertia: np.ndarray = np.diag([0.0082, 0.0082, 0.0149])
    ):
        self.mass = mass
        self.gravity = gravity
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(inertia)

        # State initialization
        self.pos = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

    def reset(self, pos: np.ndarray = None, quat: np.ndarray = None) -> None:
        """Resets dynamic state."""
        self.pos = np.array(pos if pos is not None else [1.0, 1.0, 1.0], dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.quat = np.array(quat if quat is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

    def step_rk4(
        self,
        thrust: float,
        tau_motors: np.ndarray,
        f_aero_world: np.ndarray,
        tau_aero_body: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        4th-order Runge-Kutta integration step for 6-DOF dynamics.
        """
        def derivative(p, v, q, w):
            q_u = quat_norm(q)
            R = quat_to_rot_matrix(q_u)
            
            # Position derivative
            dp = v
            
            # Velocity derivative
            thrust_world = R @ np.array([0.0, 0.0, thrust])
            gravity_world = np.array([0.0, 0.0, -self.mass * self.gravity])
            dv = (thrust_world + gravity_world + f_aero_world) / self.mass
            
            # Quaternion derivative: \dot{q} = 0.5 * q \otimes [0, w]
            q_w = np.array([0.0, w[0], w[1], w[2]])
            dq = 0.5 * quat_mult(q_u, q_w)
            
            # Angular velocity derivative: \dot{w} = I^{-1} * (tau - w \times (I*w))
            w_cross_Iw = np.cross(w, self.inertia @ w)
            dw = self.inv_inertia @ (tau_motors + tau_aero_body - w_cross_Iw)
            
            return dp, dv, dq, dw

        # RK4 stages
        p0, v0, q0, w0 = self.pos, self.vel, self.quat, self.omega
        
        dp1, dv1, dq1, dw1 = derivative(p0, v0, q0, w0)
        
        p1 = p0 + 0.5 * dt * dp1
        v1 = v0 + 0.5 * dt * dv1
        q1 = quat_norm(q0 + 0.5 * dt * dq1)
        w1 = w0 + 0.5 * dt * dw1
        dp2, dv2, dq2, dw2 = derivative(p1, v1, q1, w1)
        
        p2 = p0 + 0.5 * dt * dp2
        v2 = v0 + 0.5 * dt * dv2
        q2 = quat_norm(q0 + 0.5 * dt * dq2)
        w2 = w0 + 0.5 * dt * dw2
        dp3, dv3, dq3, dw3 = derivative(p2, v2, q2, w2)
        
        p3 = p0 + dt * dp3
        v3 = v0 + dt * dv3
        q3 = quat_norm(q0 + dt * dq3)
        w3 = w0 + dt * dw3
        dp4, dv4, dq4, dw4 = derivative(p3, v3, q3, w3)

        self.pos = p0 + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        self.vel = v0 + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        self.quat = quat_norm(q0 + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4))
        self.omega = w0 + (dt / 6.0) * (dw1 + 2*dw2 + 2*dw3 + dw4)

        return self.pos.copy(), self.vel.copy(), self.quat.copy(), self.omega.copy()
