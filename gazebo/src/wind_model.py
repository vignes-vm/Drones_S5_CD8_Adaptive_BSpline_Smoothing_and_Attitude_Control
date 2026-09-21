"""
Wind & Gust Disturbance Model for Gazebo Simulation.
Provides steady wind, sinusoidal gusts, and aerodynamic drag force/torque calculations.
"""
import numpy as np


class WindDisturbanceModel:

    def __init__(self, steady_wind=(2.0, 1.0, 0.0), gust_amplitude=(1.5, 1.5, 0.5),
                 gust_frequency=(0.3, 0.2, 0.5), cd_lin=0.15, cd_rot=0.01):
        """
        steady_wind: Constant wind velocity in world frame [m/s]
        gust_amplitude: Amplitude of sinusoidal gust oscillations [m/s]
        gust_frequency: Frequency of sinusoidal gusts [Hz]
        cd_lin: Quadratic aerodynamic linear drag coefficient [kg/m]
        cd_rot: Rotational aerodynamic damping coefficient [N*m*s/rad]
        """
        self.steady_wind = np.array(steady_wind, dtype=float)
        self.gust_amplitude = np.array(gust_amplitude, dtype=float)
        self.gust_frequency = np.array(gust_frequency, dtype=float)
        self.cd_lin = cd_lin
        self.cd_rot = cd_rot

    def get_wind_velocity(self, t):
        """Calculates instantaneous world-frame wind velocity vector at time t."""
        gust = self.gust_amplitude * np.sin(2.0 * np.pi * self.gust_frequency * t)
        noise = np.random.normal(0.0, 0.1, 3)
        return self.steady_wind + gust + noise

    def compute_aerodynamic_wrench(self, drone_velocity, drone_omega, t):
        """
        Computes aerodynamic drag force and disturbance torque acting on the vehicle.
        Returns:
          force_world: [Fx, Fy, Fz] in world frame (N)
          torque_body: [Tx, Ty, Tz] in body frame (Nm)
        """
        v_wind = self.get_wind_velocity(t)
        v_rel = drone_velocity - v_wind
        speed_rel = np.linalg.norm(v_rel)

        # Quadratic drag: F_drag = -cd_lin * |v_rel| * v_rel
        force_world = -self.cd_lin * speed_rel * v_rel

        # Rotational damping torque: tau_drag = -cd_rot * omega
        torque_body = -self.cd_rot * drone_omega

        return force_world, torque_body
