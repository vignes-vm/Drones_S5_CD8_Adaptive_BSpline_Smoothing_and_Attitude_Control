"""
Unit tests for Trajectory Converter, SLERP interpolation, and Path Planning integration.
"""
import unittest
import numpy as np
from attitude_control.quaternion import Quaternion
from path_planning.bspline import BSplinePath
from integration.trajectory_converter import TrajectoryConverter


class TestIntegration(unittest.TestCase):

    def test_trajectory_converter_tangent_to_quaternion(self):
        converter = TrajectoryConverter()
        tangent_x = np.array([10.0, 0.0, 0.0])
        q_ref = converter.path_tangent_to_quaternion(tangent_x)
        self.assertTrue(np.isclose(q_ref.q0, 1.0, atol=1e-3))

        tangent_y = np.array([0.0, 10.0, 0.0])
        q_ref_y = converter.path_tangent_to_quaternion(tangent_y)
        euler_y = q_ref_y.to_euler()
        self.assertTrue(np.isclose(euler_y[2], np.pi/2, atol=1e-3))

    def test_slerp_interpolation(self):
        q1 = Quaternion.from_euler(0.0, 0.0, 0.0)
        q2 = Quaternion.from_euler(0.0, 0.0, np.pi/2)

        q_half = TrajectoryConverter.slerp_static(q1, q2, 0.5)
        euler_half = q_half.to_euler()
        self.assertTrue(np.isclose(euler_half[2], np.pi/4, atol=1e-3))

    def test_bspline_and_trajectory_converter_integration(self):
        waypoints = np.array([
            [0.0, 0.0, 20.0],
            [100.0, 50.0, 40.0],
            [200.0, 100.0, 60.0]
        ])
        spline = BSplinePath(waypoints)
        converter = TrajectoryConverter()

        pos, q_ref = converter.get_reference_at(spline, 0.5)
        self.assertEqual(len(pos), 3)
        self.assertIsInstance(q_ref, Quaternion)
        self.assertTrue(np.isclose(np.linalg.norm(q_ref.q), 1.0))


if __name__ == '__main__':
    unittest.main()
