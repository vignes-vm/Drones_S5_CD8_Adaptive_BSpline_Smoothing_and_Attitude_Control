"""
Quaternion math module supporting Hamilton [q0, q1, q2, q3] and PyBullet [x, y, z, w] conventions.
Implements Paper 2 quaternion kinematics, shortest path sign correction, and Shepperd's algorithm.
"""
import numpy as np


class QuaternionMath:

    @staticmethod
    def normalize(q):
        q = np.asarray(q, dtype=float)
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    @staticmethod
    def hamilton_to_pybullet(q):
        """[q0, q1, q2, q3] -> [x, y, z, w] = [q1, q2, q3, q0]"""
        q = QuaternionMath.normalize(q)
        return np.array([q[1], q[2], q[3], q[0]], dtype=float)

    @staticmethod
    def pybullet_to_hamilton(q_pb):
        """[x, y, z, w] -> [q0, q1, q2, q3] = [w, x, y, z]"""
        q_pb = np.asarray(q_pb, dtype=float)
        norm = np.linalg.norm(q_pb)
        if norm > 1e-12:
            q_pb = q_pb / norm
        else:
            q_pb = np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([q_pb[3], q_pb[0], q_pb[1], q_pb[2]], dtype=float)

    @staticmethod
    def conjugate(q):
        q = QuaternionMath.normalize(q)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def multiply(p, q):
        """Hamilton product p ⊗ q"""
        p = QuaternionMath.normalize(p)
        q = QuaternionMath.normalize(q)
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = q
        return QuaternionMath.normalize([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def compute_error(q_ref, q_meas):
        """
        Quaternion error q_err = q_ref ⊗ q_meas^-1.
        Includes shortest-path sign correction (q_err <- -q_err if q_err[0] < 0).
        """
        q_ref = QuaternionMath.normalize(q_ref)
        q_meas = QuaternionMath.normalize(q_meas)
        q_err = QuaternionMath.multiply(q_ref, QuaternionMath.conjugate(q_meas))

        if q_err[0] < 0.0:
            q_err = -q_err

        return QuaternionMath.normalize(q_err)

    @staticmethod
    def from_euler(phi, theta, psi):
        """Paper 2 Eq. (15): Euler angles (roll, pitch, yaw) -> Hamilton quaternion [q0, q1, q2, q3]"""
        c_phi = np.cos(phi / 2.0)
        s_phi = np.sin(phi / 2.0)
        c_th  = np.cos(theta / 2.0)
        s_th  = np.sin(theta / 2.0)
        c_ps  = np.cos(psi / 2.0)
        s_ps  = np.sin(psi / 2.0)

        q0 = c_phi * c_th * c_ps + s_phi * s_th * s_ps
        q1 = s_phi * c_th * c_ps - c_phi * s_th * s_ps
        q2 = c_phi * s_th * c_ps + s_phi * c_th * s_ps
        q3 = c_phi * c_th * s_ps - s_phi * s_th * c_ps

        return QuaternionMath.normalize([q0, q1, q2, q3])

    @staticmethod
    def to_euler(q):
        """Hamilton quaternion [q0, q1, q2, q3] -> Euler angles [phi, theta, psi]"""
        q0, q1, q2, q3 = QuaternionMath.normalize(q)

        phi   = np.arctan2(2.0 * (q0*q1 + q2*q3), q0**2 - q1**2 - q2**2 + q3**2)
        theta = np.arcsin(np.clip(2.0 * (q0*q2 - q3*q1), -1.0, 1.0))
        psi   = np.arctan2(2.0 * (q0*q3 + q1*q2), q0**2 + q1**2 - q2**2 - q3**2)

        return np.array([phi, theta, psi], dtype=float)

    @staticmethod
    def to_rotation_matrix(q):
        """Hamilton quaternion -> 3x3 Rotation matrix R"""
        q0, q1, q2, q3 = QuaternionMath.normalize(q)
        return np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2.0*(q1*q2 - q0*q3),         2.0*(q1*q3 + q0*q2)],
            [2.0*(q1*q2 + q0*q3),         q0**2 - q1**2 + q2**2 - q3**2, 2.0*(q2*q3 - q0*q1)],
            [2.0*(q1*q3 - q0*q2),         2.0*(q2*q3 + q0*q1),         q0**2 - q1**2 - q2**2 + q3**2]
        ], dtype=float)

    @staticmethod
    def from_rotation_matrix(R):
        """3x3 Rotation matrix R -> Hamilton quaternion [q0, q1, q2, q3] (Shepperd algorithm)"""
        R = np.asarray(R, dtype=float)
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            q0 = 0.25 * s
            q1 = (R[2,1] - R[1,2]) / s
            q2 = (R[0,2] - R[2,0]) / s
            q3 = (R[1,0] - R[0,1]) / s
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q0 = (R[2,1] - R[1,2]) / s
            q1 = 0.25 * s
            q2 = (R[0,1] + R[1,0]) / s
            q3 = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q0 = (R[0,2] - R[2,0]) / s
            q1 = (R[0,1] + R[1,0]) / s
            q2 = 0.25 * s
            q3 = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q0 = (R[1,0] - R[0,1]) / s
            q1 = (R[0,2] + R[2,0]) / s
            q2 = (R[1,2] + R[2,1]) / s
            q3 = 0.25 * s
        return QuaternionMath.normalize([q0, q1, q2, q3])

    @staticmethod
    def slerp(q1, q2, t):
        """Spherical linear interpolation (SLERP) between two quaternions."""
        q1 = QuaternionMath.normalize(q1)
        q2 = QuaternionMath.normalize(q2)

        dot = np.dot(q1, q2)

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.9995:
            res = q1 + t * (q2 - q1)
            return QuaternionMath.normalize(res)

        theta_0 = np.arccos(dot)
        theta   = theta_0 * t
        sin_t   = np.sin(theta)
        sin_t0  = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_t / sin_t0
        s2 = sin_t / sin_t0

        return QuaternionMath.normalize(s1 * q1 + s2 * q2)
