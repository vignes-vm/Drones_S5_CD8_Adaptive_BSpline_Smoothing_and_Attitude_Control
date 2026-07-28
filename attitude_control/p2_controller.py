import numpy as np
from .quaternion import Quaternion

class P2Controller:

    def __init__(self, Pq=20.0, Pw=4.0):
        self.Pq = Pq   # Outer loop gain — attitude
        self.Pw = Pw   # Inner loop gain — angular rate

    def compute_error(self, q_ref, q_measured):
        q_err = q_ref.multiply(q_measured.conjugate())

        if q_err.q0 < 0:
            q_err = Quaternion(
                -q_err.q0,
                -q_err.q1,
                -q_err.q2,
                -q_err.q3
            )
        return q_err

    def compute_torque(self, q_ref, q_measured, omega):
        q_err = self.compute_error(q_ref, q_measured)

        axis_err = np.array([q_err.q1, q_err.q2, q_err.q3])

        tau = -self.Pq * axis_err - self.Pw * omega
        
        tau = np.clip(tau, -4.0, 4.0)

        return tau, axis_err