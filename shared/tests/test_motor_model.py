"""
Unit tests for Motor mixing matrix and Actuator model.
"""
import unittest
import numpy as np
from attitude_control.motor_model import MotorModel


class TestMotorModel(unittest.TestCase):

    def test_motor_model_initialization(self):
        motor = MotorModel(kf=3.13e-5, km=7.5e-7, d=0.23, omega_min=100.0, omega_max=1200.0)
        self.assertEqual(motor.M.shape, (4, 4))
        self.assertEqual(np.linalg.matrix_rank(motor.M), 4)

    def test_motor_model_forward_inverse_kinematics(self):
        motor = MotorModel()
        
        target_thrust = 9.81
        target_torques = np.zeros(3)

        omega_vec, is_saturated = motor.compute_motor_speeds(target_thrust, target_torques)
        self.assertFalse(is_saturated)
        self.assertEqual(len(omega_vec), 4)
        self.assertTrue(np.all(omega_vec > 0))
        self.assertTrue(np.all(np.isclose(omega_vec[0], omega_vec)))

        actual_thrust, actual_torques = motor.compute_thrust_and_torques(omega_vec)
        self.assertTrue(np.isclose(actual_thrust, target_thrust, atol=1e-4))
        self.assertTrue(np.allclose(actual_torques, target_torques, atol=1e-4))

    def test_motor_model_saturation(self):
        motor = MotorModel(omega_min=0.0, omega_max=500.0)

        extreme_thrust = 20.0
        extreme_torques = np.array([10.0, 0.0, 0.0])

        omega_vec, actual_thrust, actual_torques, is_saturated = motor.allocate_and_limit(
            extreme_thrust, extreme_torques
        )
        
        self.assertTrue(is_saturated)
        self.assertTrue(np.all(omega_vec <= 500.0))
        self.assertTrue(np.all(omega_vec >= 0.0))


if __name__ == '__main__':
    unittest.main()
