"""
Variable Wind Quadrotor Simulation Package.
Combines Paper 1 (3D A* + GA Adaptive B-Spline) and Paper 2 (Quaternion P^2 Control)
under piecewise distance-dependent variable wind conditions across MuJoCo, PyBullet, and Gazebo.
"""

from .quaternion_math import (
    quat_mult,
    quat_conj,
    quat_norm,
    quat_error_body,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    euler_to_quat,
    quat_to_euler
)
from .variable_wind_field import VariableSpatialWindField, SpatialWindZone
from .obstacle_planner import Obstacle3D, ObstacleEnvironment, AStar3DPlanner, GAAdaptiveBSplineSmoother, TrajectoryGenerator
from .motor_model import MotorModel
from .quaternion_controller import QuaternionP2Controller
from .quadrotor_dynamics import QuadrotorDynamics

__all__ = [
    "quat_mult",
    "quat_conj",
    "quat_norm",
    "quat_error_body",
    "quat_to_rot_matrix",
    "rot_matrix_to_quat",
    "euler_to_quat",
    "quat_to_euler",
    "VariableSpatialWindField",
    "SpatialWindZone",
    "Obstacle3D",
    "ObstacleEnvironment",
    "AStar3DPlanner",
    "GAAdaptiveBSplineSmoother",
    "TrajectoryGenerator",
    "MotorModel",
    "QuaternionP2Controller",
    "QuadrotorDynamics"
]
