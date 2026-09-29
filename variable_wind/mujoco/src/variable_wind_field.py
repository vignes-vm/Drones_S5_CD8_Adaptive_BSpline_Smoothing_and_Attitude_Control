"""
Variable Spatial Wind Field Module.
Implements a piecewise distance-dependent wind profile with dynamic speed and 3D direction changes
along the flight trajectory, incorporating aerodynamic drag and Dryden-like turbulence gusts.
"""

from typing import Dict, List, Tuple
import numpy as np


class SpatialWindZone:
    """Defines a specific spatial distance interval with distinct wind velocity and gust properties."""
    def __init__(
        self,
        name: str,
        d_start: float,
        d_end: float,
        base_velocity: np.ndarray, # [vx, vy, vz] in m/s
        gust_intensity: float = 0.5, # standard deviation of turbulence in m/s
        description: str = ""
    ):
        self.name = name
        self.d_start = d_start
        self.d_end = d_end
        self.base_velocity = np.array(base_velocity, dtype=np.float64)
        self.gust_intensity = gust_intensity
        self.description = description

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.base_velocity))


class VariableSpatialWindField:
    """
    Manages piecewise spatial wind zones across the trajectory distance,
    with smooth blending between zones and continuous stochastic turbulence.
    """
    def __init__(
        self,
        start_point: np.ndarray = np.array([1.0, 1.0, 1.0]),
        transition_width: float = 2.0, # distance over which wind transitions smoothly
        rho_air: float = 1.225,        # air density kg/m^3
        cd_body: float = 1.05,         # aerodynamic drag coefficient
        cross_section_area: float = 0.05 # effective frontal area m^2
    ):
        self.start_point = np.array(start_point, dtype=np.float64)
        self.transition_width = transition_width
        self.rho_air = rho_air
        self.cd_body = cd_body
        self.cross_section_area = cross_section_area

        # Ornstein-Uhlenbeck turbulence state
        self._gust_state = np.zeros(3, dtype=np.float64)
        self._ou_tau = 0.5  # correlation time constant (s)
        
        # Default 5-Zone Piecewise Spatial Wind Profile along Trajectory Distance
        self.zones: List[SpatialWindZone] = [
            SpatialWindZone(
                name="Zone 1: Gentle Headwind (Departure)",
                d_start=0.0,
                d_end=7.0,
                base_velocity=np.array([-2.5, 1.0, 0.2]),
                gust_intensity=0.3,
                description="Moderate opposing headwind during initial climb and departure"
            ),
            SpatialWindZone(
                name="Zone 2: Severe Lateral Crosswind",
                d_start=7.0,
                d_end=17.0,
                base_velocity=np.array([1.5, 9.5, -0.6]),
                gust_intensity=1.2,
                description="Strong lateral crosswind from +Y direction with downward shear"
            ),
            SpatialWindZone(
                name="Zone 3: Thermal Updraft & Tailwind",
                d_start=17.0,
                d_end=27.0,
                base_velocity=np.array([7.0, -3.5, 2.8]),
                gust_intensity=0.8,
                description="Accelerating tailwind combined with strong vertical updraft"
            ),
            SpatialWindZone(
                name="Zone 4: High-Shear Storm Zone",
                d_start=27.0,
                d_end=37.0,
                base_velocity=np.array([-11.0, -8.0, -1.8]),
                gust_intensity=1.6,
                description="Severe oblique storm gust opposing motion with strong downdrafts"
            ),
            SpatialWindZone(
                name="Zone 5: Steady Terminal Approach",
                d_start=37.0,
                d_end=100.0,
                base_velocity=np.array([-3.5, 2.5, 0.4]),
                gust_intensity=0.4,
                description="Stabilizing approach wind near target waypoint"
            )
        ]

    def set_custom_zones(self, zones: List[SpatialWindZone]) -> None:
        """Allows injecting custom distance-based wind zones."""
        self.zones = sorted(zones, key=lambda z: z.d_start)

    def get_zone_at_distance(self, distance: float) -> SpatialWindZone:
        """Finds active zone for a given flight distance."""
        for z in self.zones:
            if z.d_start <= distance < z.d_end:
                return z
        return self.zones[-1]

    def get_nominal_wind_vector(self, distance: float) -> np.ndarray:
        """
        Computes nominal wind vector at a given distance using smooth sigmoid blending
        across zone boundaries to prevent discrete unphysical force impulses.
        """
        if not self.zones:
            return np.zeros(3, dtype=np.float64)

        # If before first zone
        if distance <= self.zones[0].d_start:
            return self.zones[0].base_velocity.copy()

        # If beyond last zone
        if distance >= self.zones[-1].d_end:
            return self.zones[-1].base_velocity.copy()

        # Find active zone and potential blending with adjacent zones
        active_idx = 0
        for i, z in enumerate(self.zones):
            if z.d_start <= distance < z.d_end:
                active_idx = i
                break

        curr_zone = self.zones[active_idx]
        v_wind = curr_zone.base_velocity.copy()

        # Smooth transition to next zone near boundary
        if active_idx < len(self.zones) - 1:
            next_zone = self.zones[active_idx + 1]
            boundary = curr_zone.d_end
            dist_to_boundary = boundary - distance
            half_w = self.transition_width * 0.5
            
            if dist_to_boundary < half_w:
                # Sigmoidal blending factor in [0, 1]
                blend = 0.5 * (1.0 + np.sin(np.pi * (half_w - dist_to_boundary) / (2.0 * half_w)))
                v_wind = (1.0 - blend) * curr_zone.base_velocity + blend * next_zone.base_velocity

        return v_wind

    def get_total_wind(self, pos: np.ndarray, distance: float, dt: float = 0.01) -> Tuple[np.ndarray, SpatialWindZone]:
        """
        Returns (v_wind_total, active_zone) where v_wind_total includes nominal spatial wind
        plus continuous stochastic turbulent gust.
        """
        active_zone = self.get_zone_at_distance(distance)
        v_nominal = self.get_nominal_wind_vector(distance)

        # Update Ornstein-Uhlenbeck gust process: dG = - (G / tau) dt + sigma * sqrt(dt) * W
        sigma = active_zone.gust_intensity
        dw = np.random.randn(3)
        decay = np.exp(-dt / self._ou_tau)
        self._gust_state = decay * self._gust_state + sigma * np.sqrt(1.0 - decay**2) * dw

        v_total = v_nominal + self._gust_state
        return v_total, active_zone

    def compute_aerodynamic_wrench(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_rot_matrix: np.ndarray,
        distance: float,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SpatialWindZone]:
        """
        Computes aerodynamic drag force (in world frame) and aerodynamic moment (in body frame):
        F_drag = -0.5 * rho * Cd * A * ||v_rel|| * v_rel
        where v_rel = v_drone - v_wind.
        
        Returns:
            (F_drag_world, Tau_aero_body, v_wind_total, active_zone)
        """
        v_wind, active_zone = self.get_total_wind(drone_pos, distance, dt=dt)
        v_rel_world = drone_vel - v_wind
        speed_rel = np.linalg.norm(v_rel_world)

        if speed_rel < 1e-6:
            f_drag_world = np.zeros(3, dtype=np.float64)
        else:
            # Quadratic drag law
            drag_mag = 0.5 * self.rho_air * self.cd_body * self.cross_section_area * (speed_rel ** 2)
            f_drag_world = -drag_mag * (v_rel_world / speed_rel)

        # Center-of-pressure offset induced pitching/rolling aerodynamic moment
        # r_cop in body frame: [0, 0, 0.02] (slightly above center of mass)
        r_cop_body = np.array([0.005, 0.005, 0.02], dtype=np.float64)
        f_drag_body = drone_rot_matrix.T @ f_drag_world
        tau_cop = np.cross(r_cop_body, f_drag_body)

        # Aerodynamic angular damping
        tau_aero_body = tau_cop

        return f_drag_world, tau_aero_body, v_wind, active_zone
