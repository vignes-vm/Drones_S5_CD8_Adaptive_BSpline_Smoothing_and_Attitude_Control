r"""
Quaternion Mathematics Module for Variable Wind Quadrotor Simulation.
Implements Hamilton product, conjugate, rotation matrices, conversions,
and Paper 2 body-frame quaternion error formulation:
    q_err = q_meas^* \otimes q_ref
"""

import numpy as np


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r"""
    Hamilton quaternion multiplication q = q1 \otimes q2.
    Format: [w, x, y, z] (scalar-first).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float64)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q^* = [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_norm(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_error_body(q_meas: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    r"""
    Paper 2 Body-Frame Quaternion Error:
    q_err = q_meas^* \otimes q_ref = [e_w, e_x, e_y, e_z]
    where e_v = [e_x, e_y, e_z] represents the rotation error in body coordinates.
    """
    q_meas_norm = quat_norm(q_meas)
    q_ref_norm = quat_norm(q_ref)
    
    # Enforce shortest-path convention (antipodal consistency)
    if np.dot(q_meas_norm, q_ref_norm) < 0:
        q_ref_norm = -q_ref_norm
        
    q_err = quat_mult(quat_conj(q_meas_norm), q_ref_norm)
    return quat_norm(q_err)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [w, x, y, z] to 3x3 rotation matrix R (body -> world).
    """
    q = quat_norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Converts 3x3 rotation matrix to unit quaternion [w, x, y, z] via Shepperd's method.
    """
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return quat_norm(np.array([w, x, y, z], dtype=np.float64))


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler ZYX angles (roll, pitch, yaw in radians) to quaternion [w, x, y, z]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_norm(np.array([w, x, y, z], dtype=np.float64))


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians."""
    q = quat_norm(q)
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)
