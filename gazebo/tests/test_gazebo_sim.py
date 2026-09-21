"""
Automated Unit Tests for Gazebo Simulation Stack.
Tests QuaternionMath, MotorModel, GazeboEnvironment SDF Generator,
GazeboQuadrotor 6-DOF physics, and GazeboTrajectoryTracker dual-loop control.
"""
import os
import sys
import unittest
import numpy as np

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from environment.obstacles import Environment3D
from gazebo.src.quaternion_math import QuaternionMath
from gazebo.src.motor_model import MotorModel
from gazebo.src.wind_model import WindDisturbanceModel
from gazebo.src.gazebo_env import GazeboEnvironment, generate_gazebo_world_sdf
from gazebo.src.quadrotor_gazebo import GazeboQuadrotor
from gazebo.src.trajectory_tracker import GazeboTrajectoryTracker


class TestGazeboQuaternionMath(unittest.TestCase):

    def test_normalize_and_conjugate(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = QuaternionMath.normalize(q)
        np.testing.assert_allclose(q_norm, [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(q_norm), 1.0)

        q_conj = QuaternionMath.conjugate([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(q_conj, [1.0, -2.0, -3.0, -4.0])

    def test_multiply_identity(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        q_rot = QuaternionMath.normalize([0.7071, 0.7071, 0.0, 0.0])
        res = QuaternionMath.multiply(q_id, q_rot)
        np.testing.assert_allclose(res, q_rot, atol=1e-6)

    def test_euler_roundtrip(self):
        roll, pitch, yaw = 0.25, -0.15, 1.20
        q = QuaternionMath.from_euler(roll, pitch, yaw)
        euler_back = QuaternionMath.to_euler(q)
        np.testing.assert_allclose([roll, pitch, yaw], euler_back, atol=1e-5)

    def test_rotation_matrix_roundtrip(self):
        q_orig = QuaternionMath.normalize([0.5, 0.5, 0.5, 0.5])
        R = QuaternionMath.to_rotation_matrix(q_orig)
        # Check orthogonality: R @ R.T = I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=5)
        # Convert back
        q_rec = QuaternionMath.from_rotation_matrix(R)
        np.testing.assert_allclose(np.abs(q_orig), np.abs(q_rec), atol=1e-5)

    def test_error_quaternion_shortest_path(self):
        # Even if reference is -q (representing same physical orientation),
        # error quaternion should enforce w >= 0
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-1.0, 0.0, 0.0, 0.0])
        q_err = QuaternionMath.error(q1, q2)
        self.assertGreaterEqual(q_err[0], 0.0)
        np.testing.assert_allclose(q_err, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_gazebo_format_conversion(self):
        q_ham = np.array([1.0, 0.2, 0.3, 0.4])
        q_gz = QuaternionMath.hamilton_to_gazebo(q_ham)
        np.testing.assert_allclose(q_gz, [0.2, 0.3, 0.4, 1.0])
        q_ham_rec = QuaternionMath.gazebo_to_hamilton(q_gz)
        np.testing.assert_allclose(q_ham, q_ham_rec)


class TestGazeboMotorModel(unittest.TestCase):

    def setUp(self):
        self.motor = MotorModel(d=0.23, min_thrust=0.0, max_thrust=8.0)

    def test_hover_thrust_allocation(self):
        # Quadrotor hover thrust = 1.0 kg * 9.81 = 9.81 N total, torques = 0
        actual_t, actual_tau, rotor_f = self.motor.allocate(9.81, np.zeros(3))
        self.assertAlmostEqual(actual_t, 9.81, places=3)
        np.testing.assert_allclose(actual_tau, [0.0, 0.0, 0.0], atol=1e-6)
        # All 4 rotors should produce equal thrust 9.81 / 4 = 2.4525 N
        for f in rotor_f:
            self.assertAlmostEqual(f, 9.81 / 4.0, places=3)
        self.assertFalse(self.motor.is_saturated)

    def test_roll_torque_allocation(self):
        # Positive roll torque requires motor 3 (+Y) to increase, motor 1 (-Y) to decrease
        actual_t, actual_tau, rotor_f = self.motor.allocate(9.81, np.array([0.2, 0.0, 0.0]))
        self.assertAlmostEqual(actual_t, 9.81, places=3)
        self.assertAlmostEqual(actual_tau[0], 0.2, places=3)
        self.assertGreater(rotor_f[3], rotor_f[1])

    def test_saturation_clamping(self):
        # Excessive thrust command (100 N > 4 * 8 N = 32 N)
        actual_t, _, rotor_f = self.motor.allocate(100.0, np.zeros(3))
        self.assertTrue(self.motor.is_saturated)
        self.assertAlmostEqual(actual_t, 32.0, places=3)
        np.testing.assert_allclose(rotor_f, [8.0, 8.0, 8.0, 8.0])


class TestGazeboEnvironmentAndSDF(unittest.TestCase):

    def test_generate_gazebo_world_sdf(self):
        env_3d = Environment3D()
        env_3d.add_random_obstacles(n_cylinders=3, n_cuboids=3, seed=42)

        output_path = os.path.join(os.path.dirname(__file__), 'test_generated_world.sdf')
        generated_file = generate_gazebo_world_sdf(env_3d, output_path, world_name="test_world")

        self.assertTrue(os.path.exists(generated_file))
        with open(generated_file, 'r') as f:
            content = f.read()

        self.assertIn('<sdf version="1.8">', content)
        self.assertIn('<world name="test_world">', content)
        self.assertIn('cylinder_obs_', content)
        self.assertIn('cuboid_obs_', content)
        self.assertIn('model://quadrotor', content)

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)


class TestGazeboQuadrotorPhysics(unittest.TestCase):

    def setUp(self):
        self.env = GazeboEnvironment(dt=0.002)
        self.drone = GazeboQuadrotor(
            gazebo_environment=self.env,
            start_position=(0, 0, 10),
            start_orientation_euler=(0, 0, 0)
        )

    def test_quadrotor_parameters(self):
        self.assertEqual(self.drone.mass, 1.0)
        self.assertEqual(self.drone.Ixx, 6.5e-4)
        self.assertEqual(self.drone.Iyy, 6.5e-4)
        self.assertEqual(self.drone.Izz, 1.2e-3)
        self.assertEqual(self.drone.d, 0.23)

    def test_hover_equilibrium(self):
        # Hover with total thrust = mg
        hover_thrust = self.drone.mass * self.drone.g
        for _ in range(50):
            pos, vel, q, omega = self.drone.step(hover_thrust, np.zeros(3))

        # Position and velocity should remain virtually stationary
        self.assertAlmostEqual(pos[2], 10.0, delta=0.05)
        self.assertAlmostEqual(vel[2], 0.0, delta=0.05)

    def test_attitude_only_roll_step(self):
        # Step torque in roll (Mx = 0.5 Nm)
        q_init = self.drone.q.copy()
        for _ in range(20):
            q, omega = self.drone.step_attitude_only(np.array([0.5, 0.0, 0.0]), dt=0.002)

        # Angular rate in roll (p) should be positive
        self.assertGreater(omega[0], 0.0)
        # Roll angle should increase
        euler = self.drone.get_euler()
        self.assertGreater(euler[0], 0.0)


class TestGazeboTrajectoryTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = GazeboTrajectoryTracker(
            mass=1.0, g=9.81, Pq=20.0, Pw=4.0, tau_limit=4.0
        )

    def test_position_control_at_hover(self):
        pos_target = np.array([0.0, 0.0, 5.0])
        pos_current = np.array([0.0, 0.0, 5.0])
        thrust_cmd, q_des = self.tracker.compute_position_control(
            pos_target, pos_current
        )
        # Should command hover thrust ~ 9.81 N
        self.assertAlmostEqual(thrust_cmd, 9.81, places=1)
        # Attitude should be level [1, 0, 0, 0]
        np.testing.assert_allclose(q_des, [1.0, 0.0, 0.0, 0.0], atol=1e-3)

    def test_attitude_control_law_and_saturation(self):
        q_des = QuaternionMath.from_euler(0.5, 0.0, 0.0)
        q_curr = QuaternionMath.from_euler(0.0, 0.0, 0.0)
        omega = np.zeros(3)

        tau, axis_err, _ = self.tracker.compute_attitude_control(q_des, q_curr, omega)
        # Body frame restoring torque should rotate towards desired positive roll
        self.assertGreater(tau[0], 0.0)
        # Torque saturation limit must be respected
        self.assertLessEqual(np.max(np.abs(tau)), self.tracker.tau_limit)


if __name__ == '__main__':
    unittest.main()
