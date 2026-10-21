"""
PyBullet Integration Package for UAV Navigation & Quaternion Attitude Control.
Combines Paper 1 (3D A* + Adaptive B-Spline Path Smoothing) and Paper 2 (Quaternion P^2 Attitude Control).
"""
from .src.quaternion_math import QuaternionMath
from .src.motor_model import MotorModel
from .src.quadrotor_pybullet import PyBulletQuadrotor
from .src.pybullet_env import PyBulletEnvironment
from .src.trajectory_tracker import PyBulletTrajectoryTracker
from .src.wind_model import WindDisturbanceModel

__all__ = [
    'QuaternionMath',
    'MotorModel',
    'PyBulletQuadrotor',
    'PyBulletEnvironment',
    'PyBulletTrajectoryTracker',
    'WindDisturbanceModel'
]
