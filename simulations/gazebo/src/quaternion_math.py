"""
Quaternion Mathematics Module for Gazebo Simulation.
Supports Hamilton quaternion algebra [w, x, y, z] and Gazebo/ROS format [x, y, z, w],
Shepperd's rotation-to-quaternion algorithm, Euler conversions, and shortest-path sign correction.
"""
import numpy as np


class QuaternionMath:

    @staticmethod
    def normalize(q):
        """Normalizes quaternion [w, x, y, z]."""
        q = np.asarray(q, dtype=float)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

    @staticmethod
    def conjugate(q):
        """Quaternion conjugate of [w, x, y, z]."""
        q = np.asarray(q, dtype=float)
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def multiply(q1, q2):
        """
        Hamilton product of two quaternions q1 x q2.
        Both in [w, x, y, z] format.
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def error(q_ref, q_meas):
        """
        Calculates attitude error quaternion:
        q_err = q_ref x conj(q_meas)
        Enforces shortest-path rotation by requiring scalar part w >= 0.
        """
        q_ref = QuaternionMath.normalize(q_ref)
        q_meas = QuaternionMath.normalize(q_meas)
        q_err = QuaternionMath.multiply(q_ref, QuaternionMath.conjugate(q_meas))
        if q_err[0] < 0.0:
            q_err = -q_err
        return QuaternionMath.normalize(q_err)

    @staticmethod
    def from_euler(roll, pitch, yaw):
        """
        Converts Euler angles (roll phi, pitch theta, yaw psi) in ZYX order to [w, x, y, z].
        """
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return QuaternionMath.normalize([w, x, y, z])

    @staticmethod
    def to_euler(q):
        """
        Converts quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
        """
        q = QuaternionMath.normalize(q)
        w, x, y, z = q

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if np.abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @staticmethod
    def to_rotation_matrix(q):
        """
        Converts quaternion [w, x, y, z] to a 3x3 orthonormal rotation matrix R.
        v_world = R @ v_body
        """
        q = QuaternionMath.normalize(q)
        w, x, y, z = q
        return np.array([
            [1.0 - 2.0*(y*y + z*z),  2.0*(x*y - z*w),        2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),        2.0*(y*z + x*w),        1.0 - 2.0*(x*x + y*y)]
        ])

    @staticmethod
    def from_rotation_matrix(R):
        """
        Converts 3x3 rotation matrix to quaternion [w, x, y, z] using Shepperd's algorithm.
        """
        R = np.asarray(R, dtype=float)
        tr = np.trace(R)

        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

        q = np.array([w, x, y, z])
        if q[0] < 0.0:
            q = -q
        return QuaternionMath.normalize(q)

    @staticmethod
    def slerp(q1, q2, t):
        """Spherical linear interpolation between q1 and q2 with ratio t in [0, 1]."""
        q1 = QuaternionMath.normalize(q1)
        q2 = QuaternionMath.normalize(q2)
        dot = np.dot(q1, q2)

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            res = q1 + t * (q2 - q1)
            return QuaternionMath.normalize(res)

        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return QuaternionMath.normalize(s0 * q1 + s1 * q2)

    @staticmethod
    def hamilton_to_gazebo(q_hamilton):
        """Converts [w, x, y, z] to Gazebo/ROS format [x, y, z, w]."""
        w, x, y, z = q_hamilton
        return np.array([x, y, z, w])

    @staticmethod
    def gazebo_to_hamilton(q_gazebo):
        """Converts Gazebo/ROS format [x, y, z, w] to [w, x, y, z]."""
        x, y, z, w = q_gazebo
        return np.array([w, x, y, z])
