"""
Unit tests for PositionController and 6-DOF Quadrotor Dynamics.
"""
import unittest
import numpy as np
from attitude_control.quaternion import Quaternion
from attitude_control.position_controller import PositionController
from attitude_control.p2_controller import P2Controller
from attitude_control.quadrotor_model import QuadrotorModel


class TestPositionController(unittest.TestCase):

    def test_hover_thrust_and_attitude(self):
        p_ctrl = PositionController(mass=1.0, g=9.81)
        pos = np.array([10.0, 20.0, 5.0])
        pos_ref = np.array([10.0, 20.0, 5.0])

        thrust, q_des = p_ctrl.compute(pos_ref, pos)

        # Hover thrust should equal mass * g = 9.81 N
        self.assertAlmostEqual(thrust, 9.81, places=2)

        # Quaternion should be identity (level flight)
        euler = q_des.to_euler()
        self.assertTrue(np.allclose(euler, 0.0, atol=1e-3))

    def test_z_altitude_error_increases_thrust(self):
        p_ctrl = PositionController(mass=1.0, g=9.81)
        pos = np.array([0.0, 0.0, 0.0])
        pos_ref = np.array([0.0, 0.0, 5.0])  # Target 5m higher

        thrust, q_des = p_ctrl.compute(pos_ref, pos)

        # Thrust must exceed hover thrust (9.81 N) to climb
        self.assertGreater(thrust, 9.81)

    def test_xy_position_error_tilts_drone(self):
        p_ctrl = PositionController(mass=1.0, g=9.81)
        pos = np.array([0.0, 0.0, 10.0])
        pos_ref = np.array([5.0, 0.0, 10.0])  # Target 5m ahead (+x)

        thrust, q_des = p_ctrl.compute(pos_ref, pos)
        euler = q_des.to_euler()

        # Pitch (euler[1]) should tilt forward to accelerate along +X
        self.assertNotEqual(euler[1], 0.0)

    def test_quadrotor_step_6dof_movement(self):
        quad = QuadrotorModel()
        quad.reset(position=[0.0, 0.0, 0.0])
        p_ctrl = PositionController(mass=1.0, g=9.81)
        att_ctrl = P2Controller(Pq=20.0, Pw=4.0)

        dt = 0.001
        pos_ref = np.array([0.0, 0.0, 10.0])  # Climb to 10m

        # Run closed-loop for 500 steps
        for _ in range(500):
            thrust, q_des = p_ctrl.compute(pos_ref, quad.position, vel=quad.velocity)
            tau, _ = att_ctrl.compute_torque(q_des, quad.q, quad.omega)
            quad.step_6dof(tau, total_thrust=thrust, dt=dt)

        # Position Z should have increased above 0.0
        self.assertGreater(quad.position[2], 0.0)
        self.assertGreater(quad.velocity[2], 0.0)


if __name__ == '__main__':
    unittest.main()
