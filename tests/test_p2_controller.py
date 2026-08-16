"""
Unit tests for P^2 Quaternion Attitude Controller and Quadrotor Dynamics.
"""
import pytest
import numpy as np
from attitude_control.quaternion import Quaternion
from attitude_control.p2_controller import P2Controller
from attitude_control.quadrotor_model import QuadrotorModel
from environment.disturbances import WindDisturbanceModel


def test_p2_controller_zero_error():
    ctrl = P2Controller(Pq=20.0, Pw=4.0)
    q_ref = Quaternion()
    q_meas = Quaternion()
    omega = np.zeros(3)

    tau, axis_err = ctrl.compute_torque(q_ref, q_meas, omega)
    assert np.allclose(tau, 0.0)
    assert np.allclose(axis_err, 0.0)


def test_p2_controller_torque_saturation():
    ctrl = P2Controller(Pq=20.0, Pw=4.0)
    # 90 degree roll error -> large proportional torque demand
    q_ref = Quaternion.from_euler(np.pi/2, 0.0, 0.0)
    q_meas = Quaternion()
    omega = np.zeros(3)

    tau, axis_err = ctrl.compute_torque(q_ref, q_meas, omega)
    assert np.all(np.abs(tau) <= 4.0)
    assert tau[0] < -3.5  # Strong negative roll torque command


def test_quadrotor_model_rk4_integration():
    quad = QuadrotorModel()
    quad.reset(euler=(0.0, 0.0, 0.0))
    dt = 0.001
    tau = np.array([0.5, 0.0, 0.0])  # Roll torque

    for _ in range(100):
        quad.step_attitude_only(tau, dt)

    assert quad.omega[0] > 0.0  # Positive roll rate buildup
    assert quad.q.q0 < 1.0       # Rotation away from identity


def test_wind_disturbance_injection():
    dist_model = WindDisturbanceModel(steady_wind=[5.0, 0.0, 0.0], torque_bias=[0.1, 0.0, 0.0])
    quad = QuadrotorModel()
    dt = 0.001
    t = 0.5
    force_dist, torque_dist = dist_model.step(t)

    assert np.linalg.norm(force_dist) > 0.0
    assert np.linalg.norm(torque_dist) > 0.0

    # Step quadrotor with external disturbance torque
    quad.step_attitude_only(np.zeros(3), dt, tau_dist=torque_dist)
    assert np.linalg.norm(quad.omega) > 0.0
