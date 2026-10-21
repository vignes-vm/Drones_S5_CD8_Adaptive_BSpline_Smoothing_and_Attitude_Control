"""
PyBullet Integration Unit Tests.
Tests quaternion math, motor model, quadrotor body dynamics, position/attitude controller, and wind model.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pybullet.src.quaternion_math import QuaternionMath
from pybullet.src.motor_model import MotorModel
from pybullet.src.trajectory_tracker import PyBulletTrajectoryTracker
from pybullet.src.wind_model import WindDisturbanceModel


class TestPyBulletSimulation(unittest.TestCase):

    def test_quaternion_conversions_and_errors(self):
        phi, theta, psi = 0.2, -0.1, 0.5
        q_ham = QuaternionMath.from_euler(phi, theta, psi)
        self.assertAlmostEqual(np.linalg.norm(q_ham), 1.0, places=5)

        q_pb = QuaternionMath.hamilton_to_pybullet(q_ham)
        q_rec = QuaternionMath.pybullet_to_hamilton(q_pb)
        self.assertTrue(np.allclose(q_ham, q_rec, atol=1e-5))

        euler_rec = QuaternionMath.to_euler(q_ham)
        self.assertTrue(np.allclose([phi, theta, psi], euler_rec, atol=1e-4))

        # Test Shepperd's algorithm for trace <= 0
        R_x180 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        q_x180 = QuaternionMath.from_rotation_matrix(R_x180)
        self.assertAlmostEqual(abs(q_x180[1]), 1.0, places=4)

    def test_motor_model_matrix_solve(self):
        motor = MotorModel()
        thrust_hover = 9.81
        torques_zero = np.zeros(3)

        w_vec, saturated = motor.compute_motor_speeds(thrust_hover, torques_zero)
        self.assertFalse(saturated)
        self.assertEqual(len(w_vec), 4)
        self.assertTrue(np.all(w_vec > 0))

        act_thrust, act_torques = motor.compute_thrust_and_torques(w_vec)
        self.assertAlmostEqual(act_thrust, thrust_hover, places=4)
        self.assertTrue(np.allclose(act_torques, torques_zero, atol=1e-4))

    def test_trajectory_tracker_pd_and_p2_control(self):
        tracker = PyBulletTrajectoryTracker(mass=1.0, g=9.81, Pq=20.0, Pw=4.0)

        p_des = np.array([10.0, 5.0, 20.0])
        v_des = np.zeros(3)
        p_meas = np.array([0.0, 0.0, 10.0])
        v_meas = np.zeros(3)
        q_meas = QuaternionMath.from_euler(0.0, 0.0, 0.0)
        omega_meas = np.zeros(3)

        thrust_cmd, torques_cmd, q_ref, axis_err, pos_err = tracker.step_control(
            p_des, v_des, p_meas, v_meas, q_meas, omega_meas
        )

        self.assertGreater(thrust_cmd, 9.81)  # Required thrust > hover thrust to gain altitude
        self.assertLessEqual(np.max(np.abs(torques_cmd)), 4.0)
        self.assertGreater(pos_err, 0.0)

    def test_wind_model_disturbance(self):
        wind = WindDisturbanceModel(steady_wind=[4.0, 2.0, 0.0])
        f_dist, tau_dist = wind.step(t=1.0, quad_velocity=[1.0, 0.0, 0.0])

        self.assertGreater(np.linalg.norm(f_dist), 0.0)
        self.assertGreater(np.linalg.norm(tau_dist), 0.0)


if __name__ == '__main__':
    unittest.main()
