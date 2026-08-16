"""
Unit tests for Motor mixing matrix and Actuator model.
"""
import pytest
import numpy as np
from attitude_control.motor_model import MotorModel


def test_motor_model_initialization():
    motor = MotorModel(kf=3.13e-5, km=7.5e-7, d=0.23, omega_min=100.0, omega_max=1200.0)
    assert motor.M.shape == (4, 4)
    assert np.linalg.matrix_rank(motor.M) == 4


def test_motor_model_forward_inverse_kinematics():
    motor = MotorModel()
    
    # Target hover thrust (9.81 N) and zero torques
    target_thrust = 9.81
    target_torques = np.zeros(3)

    omega_vec, is_saturated = motor.compute_motor_speeds(target_thrust, target_torques)
    assert not is_saturated
    assert len(omega_vec) == 4
    assert np.all(omega_vec > 0)
    assert np.all(np.isclose(omega_vec[0], omega_vec))  # All motors equal speed for hover

    actual_thrust, actual_torques = motor.compute_thrust_and_torques(omega_vec)
    assert np.isclose(actual_thrust, target_thrust, atol=1e-4)
    assert np.allclose(actual_torques, target_torques, atol=1e-4)


def test_motor_model_saturation():
    motor = MotorModel(omega_min=0.0, omega_max=500.0)

    # Request extreme roll torque that exceeds max motor speed 500 rad/s
    extreme_thrust = 20.0
    extreme_torques = np.array([10.0, 0.0, 0.0])

    omega_vec, actual_thrust, actual_torques, is_saturated = motor.allocate_and_limit(
        extreme_thrust, extreme_torques
    )
    
    assert is_saturated
    assert np.all(omega_vec <= 500.0)
    assert np.all(omega_vec >= 0.0)
