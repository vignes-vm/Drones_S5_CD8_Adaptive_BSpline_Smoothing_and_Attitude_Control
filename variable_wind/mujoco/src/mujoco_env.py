r"""
MuJoCo Simulation Environment for Variable Wind Quadrotor.
Builds an MJCF XML world with randomized 3D obstacles and 4 quadrotor thrusters,
steps physics via native MuJoCo (with automatic fallback), and records full telemetry.
"""

from typing import Dict, List, Optional, Tuple
import os
import tempfile
import numpy as np

try:
    import mujoco
    HAS_MUJOCO = hasattr(mujoco, 'MjModel')
except Exception:
    mujoco = None
    HAS_MUJOCO = False

from .quaternion_math import quat_norm, quat_to_rot_matrix, rot_matrix_to_quat
from .obstacle_planner import Obstacle3D, ObstacleEnvironment
from .variable_wind_field import VariableSpatialWindField
from .quaternion_controller import QuaternionP2Controller
from .motor_model import MotorModel
from .quadrotor_dynamics import QuadrotorDynamics


class MuJoCoVariableWindEnv:
    """
    MuJoCo Physics Environment with dynamic obstacles, thrusters, and variable wind wrench injection.
    """
    def __init__(
        self,
        obstacles: List[Obstacle3D] = None,
        dt: float = 0.005,
        mass: float = 1.0,
        arm_length: float = 0.225
    ):
        self.obstacles = obstacles if obstacles is not None else []
        self.dt = dt
        self.mass = mass
        self.arm_length = arm_length

        self.model = None
        self.data = None
        self.use_native_mujoco = False
        self._build_mujoco_model()

        # Controller and wind field
        self.motor = MotorModel(arm_length=arm_length)
        self.controller = QuaternionP2Controller(mass=mass, motor_model=self.motor)
        self.wind_field = VariableSpatialWindField()
        self.fallback_dynamics = QuadrotorDynamics(mass=mass)

    def _generate_mjcf_xml(self) -> str:
        """Generates MJCF XML with quadrotor body, 4 thrusters, and 3D obstacles."""
        xml = f"""<mujoco model="quadrotor_variable_wind">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{self.dt}" gravity="0 0 -9.81" integrator="RK4"/>
  
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
  </visual>

  <worldbody>
    <geom name="ground" type="plane" size="40 40 0.1" rgba="0.85 0.85 0.85 1"/>
    
    <!-- Quadrotor Body -->
    <body name="quadrotor" pos="1.0 1.0 1.0">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{self.mass}" diaginertia="0.0082 0.0082 0.0149"/>
      <geom name="core" type="sphere" size="0.08" rgba="0.2 0.4 0.9 1"/>
      <geom name="arm_x" type="capsule" fromto="{-self.arm_length} 0 0 {self.arm_length} 0 0" size="0.015" rgba="0.1 0.1 0.1 1"/>
      <geom name="arm_y" type="capsule" fromto="0 {-self.arm_length} 0 0 {self.arm_length} 0" size="0.015" rgba="0.1 0.1 0.1 1"/>
      
      <site name="rotor1" pos="{self.arm_length} 0 0.02"/>
      <site name="rotor2" pos="0 {self.arm_length} 0.02"/>
      <site name="rotor3" pos="{-self.arm_length} 0 0.02"/>
      <site name="rotor4" pos="0 {-self.arm_length} 0.02"/>
    </body>

    <!-- 3D Obstacles -->
"""
        for i, obs in enumerate(self.obstacles):
            cx, cy, cz = obs.center
            r = obs.radius
            if obs.kind == 'cylinder':
                h = obs.height
                xml += f'    <geom name="obs_{i}" type="cylinder" pos="{cx} {cy} {h*0.5}" size="{r} {h*0.5}" rgba="0.85 0.25 0.25 0.75" contype="0" conaffinity="0"/>\n'
            else:
                xml += f'    <geom name="obs_{i}" type="sphere" pos="{cx} {cy} {cz}" size="{r}" rgba="0.95 0.55 0.15 0.75" contype="0" conaffinity="0"/>\n'

        xml += """  </worldbody>
</mujoco>
"""
        return xml

    def _build_mujoco_model(self) -> None:
        """Loads or builds MuJoCo model."""
        if not HAS_MUJOCO:
            self.use_native_mujoco = False
            return
            
        try:
            xml_str = self._generate_mjcf_xml()
            self.model = mujoco.MjModel.from_xml_string(xml_str)
            self.data = mujoco.MjData(self.model)
            self.use_native_mujoco = True
        except Exception as e:
            print(f"Notice: MuJoCo model compilation ({e}). Using robust 6-DOF fallback.")
            self.use_native_mujoco = False

    def reset(self, initial_pos: np.ndarray = np.array([1.0, 1.0, 1.0])) -> None:
        """Resets simulation state."""
        self.controller.reset()
        self.motor.reset()
        self.fallback_dynamics.reset(pos=initial_pos)

        if self.use_native_mujoco and self.model is not None:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[0:3] = initial_pos
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quat [w, x, y, z]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def step(
        self,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray,
        acc_ref: np.ndarray,
        yaw_ref: float,
        dist_along_path: float
    ) -> Dict:
        """
        Runs one simulation step applying Paper 2 quaternion control and variable wind field.
        """
        # Read current state
        if self.use_native_mujoco and self.model is not None:
            pos_meas = self.data.qpos[0:3].copy()
            quat_meas = quat_norm(self.data.qpos[3:7].copy())
            vel_meas = self.data.qvel[0:3].copy()
            omega_meas = self.data.qvel[3:6].copy()
        else:
            pos_meas = self.fallback_dynamics.pos.copy()
            quat_meas = self.fallback_dynamics.quat.copy()
            vel_meas = self.fallback_dynamics.vel.copy()
            omega_meas = self.fallback_dynamics.omega.copy()

        # 1. Compute Paper 2 quaternion control
        omega_cmd, total_thrust, tau_cmd, q_ref, q_err = self.controller.compute_control(
            pos_meas=pos_meas,
            vel_meas=vel_meas,
            quat_meas=quat_meas,
            omega_meas=omega_meas,
            pos_ref=pos_ref,
            vel_ref=vel_ref,
            acc_ref=acc_ref,
            yaw_ref=yaw_ref,
            dt=self.dt
        )

        # 2. Update motor dynamics
        actual_omega = self.motor.step(omega_cmd, self.dt)
        thrust_act, tau_body_act, rotor_forces = self.motor.compute_forces_and_torques(actual_omega)

        # 3. Compute Piecewise Variable Spatial Wind Drag and Moments
        R_meas = quat_to_rot_matrix(quat_meas)
        f_drag_world, tau_aero_body, v_wind, active_zone = self.wind_field.compute_aerodynamic_wrench(
            drone_pos=pos_meas,
            drone_vel=vel_meas,
            drone_rot_matrix=R_meas,
            distance=dist_along_path,
            dt=self.dt
        )

        # 4. Integrate physical dynamics
        if self.use_native_mujoco and self.model is not None:
            thrust_world = R_meas @ np.array([0.0, 0.0, thrust_act])
            self.data.qfrc_applied[0:3] = thrust_world + f_drag_world
            self.data.qfrc_applied[3:6] = tau_body_act + tau_aero_body
            
            mujoco.mj_step(self.model, self.data)
            
            new_pos = self.data.qpos[0:3].copy()
            new_vel = self.data.qvel[0:3].copy()
            new_quat = quat_norm(self.data.qpos[3:7].copy())
            new_omega = self.data.qvel[3:6].copy()
        else:
            new_pos, new_vel, new_quat, new_omega = self.fallback_dynamics.step_rk4(
                thrust=thrust_act,
                tau_motors=tau_body_act,
                f_aero_world=f_drag_world,
                tau_aero_body=tau_aero_body,
                dt=self.dt
            )

        pos_error = float(np.linalg.norm(pos_ref - new_pos))
        quat_err_norm = float(np.linalg.norm(q_err[1:4]))

        return {
            "pos": new_pos,
            "vel": new_vel,
            "quat": new_quat,
            "omega": new_omega,
            "pos_ref": pos_ref,
            "vel_ref": vel_ref,
            "q_ref": q_ref,
            "q_err": q_err,
            "pos_error": pos_error,
            "quat_err_norm": quat_err_norm,
            "omega_cmd": omega_cmd,
            "actual_omega": actual_omega,
            "rotor_forces": rotor_forces,
            "total_thrust": thrust_act,
            "tau_body": tau_body_act,
            "v_wind": v_wind,
            "wind_speed": float(np.linalg.norm(v_wind)),
            "active_zone": active_zone.name,
            "active_zone_name": active_zone.name,
            "active_zone_desc": active_zone.description,
            "f_drag": f_drag_world,
            "dist_along_path": dist_along_path
        }
