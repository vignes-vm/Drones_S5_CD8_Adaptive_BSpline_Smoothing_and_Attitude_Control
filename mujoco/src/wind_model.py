"""
Wind Disturbance & Aerodynamic Turbulence Engine
Simulates realistic atmospheric disturbances on a 0.5kg quadrotor in MuJoCo:
1. Continuous mean crosswind vector (v_wind)
2. Stochastic Dryden turbulence gusts (d_gust)
3. Physical Aerodynamic drag force F_drag = 0.5 * rho * Cd * A * ||v_rel|| * v_rel
4. Aerodynamic torque imbalances tau_dist
"""

import numpy as np

class WindDisturbanceModel:
    def __init__(self, wind_speed: float = 3.5, wind_direction_deg: float = 45.0, gust_intensity: float = 0.5, dt: float = 0.002):
        """
        Parameters:
        - wind_speed: Mean crosswind speed in m/s (e.g. 3.5 m/s)
        - wind_direction_deg: Angle of wind in xy-plane (degrees)
        - gust_intensity: Turbulence fluctuation magnitude (m/s)
        - dt: Physics simulation timestep
        """
        self.wind_speed = wind_speed
        self.gust_intensity = gust_intensity
        self.dt = dt

        # Mean wind vector in 3D (m/s)
        rad = np.radians(wind_direction_deg)
        self.v_mean = np.array([
            wind_speed * np.cos(rad),
            wind_speed * np.sin(rad),
            0.2 * np.sin(2 * rad)  # mild vertical component
        ], dtype=np.float64)

        # Physical constants for 0.5kg micro-quadrotor airframe
        self.rho = 1.225       # Air density (kg/m^3)
        self.Cd = 1.0          # Drag coefficient
        self.area = 0.04       # Effective frontal area (m^2)

        self.gust_state = np.zeros(3)

    def update_gust(self) -> np.ndarray:
        """Updates stochastic wind gust using first-order Gauss-Markov process."""
        tau_gust = 1.5  # Gust correlation time (s)
        noise = np.random.normal(0, self.gust_intensity, size=3)
        alpha = np.exp(-self.dt / tau_gust)
        self.gust_state = alpha * self.gust_state + (1 - alpha) * noise
        return self.gust_state

    def compute_disturbance(self, quad_vel: np.ndarray) -> tuple:
        """
        Computes realistic aerodynamic drag force and torque disturbance:
        F_drag: (3,) force vector in world frame (N)
        tau_dist: (3,) torque disturbance vector in body frame (Nm)
        """
        gust = self.update_gust()
        v_total_wind = self.v_mean + gust

        # Relative air velocity
        v_rel = v_total_wind - quad_vel
        speed_rel = np.linalg.norm(v_rel)

        # Aerodynamic drag force: F_drag = 0.5 * rho * Cd * A * ||v_rel|| * v_rel
        F_drag = 0.5 * self.rho * self.Cd * self.area * speed_rel * v_rel

        # Cap drag force to realistic limits for 0.5kg drone (max ~1.5 N)
        F_drag = np.clip(F_drag, -1.5, 1.5)

        # Aerodynamic torque disturbance due to rotor drag imbalance under crosswind
        tau_dist = np.random.normal(0, 0.005, size=3)
        tau_dist[0] += 0.01 * v_rel[1]  # Roll disturbance from crosswind
        tau_dist[1] -= 0.01 * v_rel[0]  # Pitch disturbance from crosswind
        tau_dist = np.clip(tau_dist, -0.2, 0.2)

        return F_drag, tau_dist
