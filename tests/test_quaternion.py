"""
Unit tests for Quaternion algebra and rotation conversions.
"""
import pytest
import numpy as np
from attitude_control.quaternion import Quaternion


def test_quaternion_initialization_and_normalization():
    q = Quaternion(2.0, 0.0, 0.0, 0.0)
    assert np.isclose(q.q0, 1.0)
    assert np.isclose(np.linalg.norm(q.q), 1.0)

    q2 = Quaternion(1.0, 2.0, 3.0, 4.0)
    assert np.isclose(np.linalg.norm(q2.q), 1.0)


def test_quaternion_conjugate_and_inverse():
    q = Quaternion(0.7071, 0.7071, 0.0, 0.0)
    q_conj = q.conjugate()
    q_inv  = q.inverse()

    assert np.isclose(q_conj.q0, q.q0)
    assert np.isclose(q_conj.q1, -q.q1)
    assert np.isclose(q_inv.q, q_conj.q).all()

    identity = q.multiply(q_inv)
    assert np.isclose(identity.q0, 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(identity.q[1:]), 0.0, atol=1e-6)


def test_euler_conversion_roundtrip():
    phi, theta, psi = 0.3, -0.2, 0.8
    q = Quaternion.from_euler(phi, theta, psi)
    euler_recovered = q.to_euler()

    assert np.isclose(euler_recovered[0], phi, atol=1e-5)
    assert np.isclose(euler_recovered[1], theta, atol=1e-5)
    assert np.isclose(euler_recovered[2], psi, atol=1e-5)


def test_from_rotation_matrix_positive_trace():
    # Identity matrix (trace = 3 > 0)
    R_id = np.eye(3)
    q_id = Quaternion.from_rotation_matrix(R_id)
    assert np.isclose(q_id.q0, 1.0, atol=1e-6)
    assert np.allclose(q_id.q[1:], 0.0, atol=1e-6)

    # Small rotation
    phi, theta, psi = 0.1, 0.2, -0.15
    q_orig = Quaternion.from_euler(phi, theta, psi)
    R = q_orig.to_rotation_matrix()
    q_rec = Quaternion.from_rotation_matrix(R)

    dot = np.dot(q_orig.q, q_rec.q)
    assert np.isclose(abs(dot), 1.0, atol=1e-5)


def test_from_rotation_matrix_non_positive_trace():
    """
    Test Shepperd's algorithm fix for non-positive trace matrices
    (e.g., 180-degree rotations around principal axes where trace <= 0).
    """
    # 180-degree rotation around X axis: R = diag(1, -1, -1), trace = -1 <= 0
    R_x180 = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    q_x180 = Quaternion.from_rotation_matrix(R_x180)
    assert np.isclose(abs(q_x180.q1), 1.0, atol=1e-5)
    assert np.isclose(q_x180.q0, 0.0, atol=1e-5)

    # 180-degree rotation around Y axis: R = diag(-1, 1, -1), trace = -1 <= 0
    R_y180 = np.array([
        [-1.0, 0.0,  0.0],
        [ 0.0, 1.0,  0.0],
        [ 0.0, 0.0, -1.0]
    ])
    q_y180 = Quaternion.from_rotation_matrix(R_y180)
    assert np.isclose(abs(q_y180.q2), 1.0, atol=1e-5)
    assert np.isclose(q_y180.q0, 0.0, atol=1e-5)

    # 180-degree rotation around Z axis: R = diag(-1, -1, 1), trace = -1 <= 0
    R_z180 = np.array([
        [-1.0,  0.0, 0.0],
        [ 0.0, -1.0, 0.0],
        [ 0.0,  0.0, 1.0]
    ])
    q_z180 = Quaternion.from_rotation_matrix(R_z180)
    assert np.isclose(abs(q_z180.q3), 1.0, atol=1e-5)
    assert np.isclose(q_z180.q0, 0.0, atol=1e-5)


def test_quaternion_error_computation():
    q_ref = Quaternion.from_euler(0.5, 0.0, 0.0)
    q_meas = Quaternion.from_euler(0.2, 0.0, 0.0)
    q_err = q_meas.error(q_ref)

    # Error should equal rotation from meas to ref (~0.3 rad roll)
    euler_err = q_err.to_euler()
    assert np.isclose(euler_err[0], 0.3, atol=1e-3)
