"""
Gazebo Quadrotor Body Wrapper.
Interfaces with Gazebo physics, maintains state (position, velocity, orientation quaternion,
angular rate), applies motor allocation, and steps 6-DOF and attitude-only dynamics.
"""
import numpy as np
from .quaternion_math import QuaternionMath
from .motor_model import MotorModel


class GazeboQuadrotor:

    def __init__(self, gazebo_environment=None, start_position=(0.0, 0.0, 1.0),
                 start_orientation_euler=(0.0, 0.0, 0.0), enable_motor_model=True):
        """
        Quadrotor matching Paper 2 physical properties:
        mass = 1.0 kg
        Ixx = 6.5e-4 kg*m^2, Iyy = 6.5e-4 kg*m^2, Izz = 1.2e-3 kg*m^2
        arm length d = 0.23 m
        """
        self.env = gazebo_environment
        self.dt = self.env.dt if self.env else 0.002

        self.mass = 1.0
        self.Ixx = 6.5e-4
        self.Iyy = 6.5e-4
        self.Izz = 1.2e-3
        self.d = 0.23
        self.g = 9.81
        self.inertia = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.inv_inertia = np.linalg.inv(self.inertia)

        self.enable_motor_model = enable_motor_model
        self.motor_model = MotorModel(d=self.d)

        self.start_pos = np.array(start_position, dtype=float)
        self.start_euler = np.array(start_orientation_euler, dtype=float)
        self.start_quat = QuaternionMath.from_euler(*self.start_euler)

        # Vehicle state
        self.position = self.start_pos.copy()
        self.velocity = np.zeros(3, dtype=float)
        self.q = self.start_quat.copy()  # Hamilton [w, x, y, z]
        self.omega = np.zeros(3, dtype=float)

        # Actuator state
        self.applied_thrust = 0.0
        self.applied_torques = np.zeros(3, dtype=float)
        self.motor_thrusts = np.zeros(4, dtype=float)
        self.is_saturated = False

        self.reset(start_position, start_orientation_euler)

    def reset(self, position=None, orientation_euler=None):
        """Resets quadrotor state to initial conditions."""
        if position is not None:
            self.position = np.array(position, dtype=float)
        if orientation_euler is not None:
            self.q = QuaternionMath.from_euler(*orientation_euler)

        self.velocity = np.zeros(3, dtype=float)
        self.omega = np.zeros(3, dtype=float)
        self.applied_thrust = 0.0
        self.applied_torques = np.zeros(3, dtype=float)

        if self.env and self.env.fallback:
            self.env.fallback.reset(self.position, self.q)

    def get_euler(self):
        """Returns current [roll, pitch, yaw] Euler angles in radians."""
        return QuaternionMath.to_euler(self.q)

    def step(self, total_thrust_cmd, torques_cmd, wind_force_world=None, wind_torque_body=None):
        """
        Applies control inputs and steps 6-DOF quadrotor flight dynamics.
        total_thrust_cmd: Scalar collective thrust (N)
        torques_cmd: 3-element vector [tau_x, tau_y, tau_z] (Nm)
        """
        # Motor allocation & saturation limit
        if self.enable_motor_model:
            actual_thrust, actual_torques, motor_thrusts = self.motor_model.allocate(
                total_thrust_cmd, torques_cmd
            )
            self.motor_thrusts = motor_thrusts
            self.is_saturated = self.motor_model.is_saturated
        else:
            actual_thrust = max(0.0, total_thrust_cmd)
            actual_torques = np.clip(torques_cmd, -4.0, 4.0)
            self.motor_thrusts = np.full(4, actual_thrust / 4.0)
            self.is_saturated = np.any(np.abs(torques_cmd) > 4.0)

        self.applied_thrust = actual_thrust
        self.applied_torques = actual_torques

        # Step physics via environment engine
        if self.env and self.env.fallback:
            pos, vel, q, omega = self.env.fallback.step(
                actual_thrust, actual_torques, wind_force_world, wind_torque_body
            )
            self.position = pos
            self.velocity = vel
            self.q = q
            self.omega = omega

        return self.position, self.velocity, self.q, self.omega

    def step_attitude_only(self, torques_cmd, dt=None):
        """
        Steps rotational-only dynamics for Paper 2 benchmark recreation (Step, Sine, Flip).
        """
        if dt is None:
            dt = self.dt

        tau = np.clip(torques_cmd, -4.0, 4.0)
        self.applied_torques = tau

        # I * w_dot = tau - w x (I * w)
        gyro = np.cross(self.omega, self.inertia @ self.omega)
        omega_dot = self.inv_inertia @ (tau - gyro)

        # q_dot = 0.5 * q x [0, omega]
        omega_quat = np.array([0.0, self.omega[0], self.omega[1], self.omega[2]])
        q_dot = 0.5 * QuaternionMath.multiply(self.q, omega_quat)

        # Integration
        self.omega += omega_dot * dt
        self.q = QuaternionMath.normalize(self.q + q_dot * dt)

        return self.q, self.omega
