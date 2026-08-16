"""
Environmental Wind & Aerodynamic Disturbance Model.
Implements steady wind vectors, sinusoidal turbulence gusts,
aerodynamic drag force, and external torque disturbances.
"""
import numpy as np


class WindDisturbanceModel:
    """
    Environmental disturbance generator simulating wind forces and gust torques.
    """

    def __init__(self, steady_wind=None, gust_amplitude=None, gust_frequency=None,
                 torque_bias=None, torque_noise_std=0.01, air_density=1.225,
                 cd_drag=0.5, reference_area=0.1):
        
        self.steady_wind = np.array(steady_wind if steady_wind is not None else [2.0, 1.0, 0.5], dtype=float)
        self.gust_amplitude = np.array(gust_amplitude if gust_amplitude is not None else [1.5, 1.0, 0.5], dtype=float)
        self.gust_frequency = np.array(gust_frequency if gust_frequency is not None else [0.5, 0.8, 1.2], dtype=float)
        
        self.torque_bias = np.array(torque_bias if torque_bias is not None else [0.05, -0.03, 0.02], dtype=float)
        self.torque_noise_std = float(torque_noise_std)
        
        self.rho = float(air_density)
        self.Cd  = float(cd_drag)
        self.A   = float(reference_area)

    def get_wind_velocity(self, t):
        """
        Calculates total wind velocity vector at time t (steady wind + turbulent gust).
        """
        gust = self.gust_amplitude * np.sin(self.gust_frequency * t)
        return self.steady_wind + gust

    def get_wind_force(self, t, quad_velocity=None):
        """
        Calculates drag force F = 0.5 * rho * Cd * A * v_relative^2.
        """
        v_wind = self.get_wind_velocity(t)
        v_rel  = v_wind if quad_velocity is None else (v_wind - np.asarray(quad_velocity, dtype=float))
        v_norm = np.linalg.norm(v_rel)
        if v_norm < 1e-6:
            return np.zeros(3)
        
        drag_mag = 0.5 * self.rho * self.Cd * self.A * (v_norm ** 2)
        return drag_mag * (v_rel / v_norm)

    def get_disturbance_torque(self, t):
        """
        Calculates external aerodynamic disturbance torque (constant bias + gust torque + noise).
        """
        gust_torque = 0.02 * np.sin(self.gust_frequency * t + np.pi/4)
        noise = np.random.normal(0.0, self.torque_noise_std, size=3)
        return self.torque_bias + gust_torque + noise

    def step(self, t, quad_velocity=None):
        """
        Returns (force_dist, torque_dist) at time t.
        """
        force_dist  = self.get_wind_force(t, quad_velocity)
        torque_dist = self.get_disturbance_torque(t)
        return force_dist, torque_dist
