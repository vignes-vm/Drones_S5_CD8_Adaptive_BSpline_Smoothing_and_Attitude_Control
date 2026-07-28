import numpy as np

class Quaternion:

    def __init__(self, q0=1.0, q1=0.0, q2=0.0, q3=0.0):
        self.q = np.array([q0, q1, q2, q3], dtype=float)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.q)
        if norm > 1e-10:
            self.q = self.q / norm

    @property
    def q0(self): return self.q[0]
    @property
    def q1(self): return self.q[1]
    @property
    def q2(self): return self.q[2]
    @property
    def q3(self): return self.q[3]

    def conjugate(self):
        return Quaternion(self.q0, -self.q1, -self.q2, -self.q3)

    def inverse(self):
        return self.conjugate()

    def multiply(self, other):
        p = self.q
        q = other.q
        result = np.array([
            p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
            p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
            p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
            p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
        ])
        return Quaternion(*result)

    def Q_matrix(self):
        q = self.q
        return np.array([
            [ q[0], -q[1], -q[2], -q[3]],
            [ q[1],  q[0], -q[3],  q[2]],
            [ q[2],  q[3],  q[0], -q[1]],
            [ q[3], -q[2],  q[1],  q[0]]
        ])

    def to_rotation_matrix(self):
        q0, q1, q2, q3 = self.q
        R = np.array([
            [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3),         q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2),         2*(q2*q3+q0*q1),         q0**2-q1**2-q2**2+q3**2]
        ])
        return R

    def to_euler(self):
        q0, q1, q2, q3 = self.q
        phi   = np.arctan2(2*(q0*q1 + q2*q3), q0**2 - q1**2 - q2**2 + q3**2)
        theta = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1, 1))
        psi   = np.arctan2(2*(q0*q3 + q1*q2), q0**2 + q1**2 - q2**2 - q3**2)
        return np.array([phi, theta, psi])

    @staticmethod
    def from_euler(phi, theta, psi):
        # Paper 2 Equation (15) — Euler to quaternion
        q0 = (np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2)
            + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2))
        q1 = (np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2)
            - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2))
        q2 = (np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2)
            + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2))
        q3 = (np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2)
            - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2))
        return Quaternion(q0, q1, q2, q3)

    @staticmethod
    def from_rotation_matrix(R):
        # Convert 3x3 rotation matrix to quaternion
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q0 = 0.25 / s
            q1 = (R[2,1] - R[1,2]) * s
            q2 = (R[0,2] - R[2,0]) * s
            q3 = (R[1,0] - R[0,1]) * s
        else:
            q0 = 0.5 * np.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2]))
            q1 = 0.5 * np.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2]))
            q2 = 0.5 * np.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2]))
            q3 = 0.5 * np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2]))
        return Quaternion(q0, q1, q2, q3)

    def error(self, q_ref):
        return q_ref.multiply(self.conjugate())

    def __repr__(self):
        return f"Quaternion([{self.q0:.4f}, {self.q1:.4f}, {self.q2:.4f}, {self.q3:.4f}])"