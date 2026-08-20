"""
Quaternion Mathematics Module
Implements fundamental quaternion operations according to Paper 1:
"Full Quaternion Based Attitude Control for a Quadrotor" (Fresk & Nikolakopoulos, 2013)
Uses MuJoCo's native mju_quat2Mat for exact rotation transformations.
"""

import numpy as np
import mujoco

def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Computes quaternion product p ⊗ q.
    q = [q0, q1, q2, q3]^T where q0 is scalar part, [q1, q2, q3] is vector part.
    Eq (1-2) & Kronecker product in Paper 1.
    """
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q

    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0
    ], dtype=np.float64)

def quat_conj(q: np.ndarray) -> np.ndarray:
    """Returns conjugate q* = [q0, -q1, -q2, -q3]^T (Eq 4)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_norm(q: np.ndarray) -> float:
    """Returns norm ||q|| (Eq 3)."""
    return float(np.linalg.norm(q))

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternion to unit length."""
    n = quat_norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Converts Euler angles (phi, theta, psi) in radians to Quaternion [q0, q1, q2, q3]^T.
    Paper 1 Equation (15).
    """
    cp = np.cos(phi / 2.0)
    sp = np.sin(phi / 2.0)
    ct = np.cos(theta / 2.0)
    st = np.sin(theta / 2.0)
    cy = np.cos(psi / 2.0)
    sy = np.sin(psi / 2.0)

    q0 = cp * ct * cy + sp * st * sy
    q1 = sp * ct * cy - cp * st * sy
    q2 = cp * st * cy + sp * ct * sy
    q3 = cp * ct * sy - sp * st * cy

    return quat_normalize(np.array([q0, q1, q2, q3], dtype=np.float64))

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Converts Quaternion [q0, q1, q2, q3]^T to Euler angles [phi, theta, psi]^T in radians.
    Paper 1 Equation (16).
    """
    q0, q1, q2, q3 = quat_normalize(q)

    # phi = atan2(2(q0 q1 + q2 q3), q0^2 - q1^2 - q2^2 + q3^2)
    phi = np.arctan2(2.0 * (q0 * q1 + q2 * q3), q0**2 - q1**2 - q2**2 + q3**2)

    # theta = asin(2(q0 q2 - q3 q1))
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.arcsin(sin_theta)

    # psi = atan2(2(q0 q3 + q1 q2), q0^2 + q1^2 - q2^2 - q3^2)
    psi = np.arctan2(2.0 * (q0 * q3 + q1 * q2), q0**2 + q1**2 - q2**2 - q3**2)

    return np.array([phi, theta, psi], dtype=np.float64)

def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion to 3x3 Direction Cosine / Rotation Matrix R(q)
    using MuJoCo's native mju_quat2Mat algorithm.
    """
    res = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(res, quat_normalize(q))
    return res.reshape(3, 3)

def compute_quat_error(q_ref: np.ndarray, q_m: np.ndarray) -> np.ndarray:
    """
    Computes unit error quaternion q_err = q_ref ⊗ q_m* (Eq 19 in Paper 1).
    Applies shortest-path logic: if q_err[0] < 0, q_err = -q_err.
    Returns q_err.
    """
    q_ref_u = quat_normalize(q_ref)
    q_m_u = quat_normalize(q_m)

    q_err = quat_mult(q_ref_u, quat_conj(q_m_u))

    # Shortest path check (Paper 1 Section IV):
    if q_err[0] < 0.0:
        q_err = -q_err

    return q_err
