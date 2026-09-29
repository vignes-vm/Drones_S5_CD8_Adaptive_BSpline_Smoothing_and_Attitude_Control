"""
Gazebo Simulation Environment & SDF Generator for Variable Wind Quadrotor.
Generates Gazebo SDF world files with randomized obstacles, quadrotor kinematics/dynamics,
and runs a Gazebo-compatible 6-DOF physics engine with variable spatial wind.
"""

from typing import Dict, List, Optional, Tuple
import os
import numpy as np

from .quaternion_math import quat_norm, quat_to_rot_matrix, rot_matrix_to_quat
from .obstacle_planner import Obstacle3D
from .variable_wind_field import VariableSpatialWindField
from .quaternion_controller import QuaternionP2Controller
from .motor_model import MotorModel
from .quadrotor_dynamics import QuadrotorDynamics


class GazeboVariableWindEnv:
    """
    Gazebo environment generating SDF worlds and running Gazebo-standard 6-DOF physics.
    """
    def __init__(
        self,
        obstacles: List[Obstacle3D] = None,
        dt: float = 0.005,
        mass: float = 1.0,
        arm_length: float = 0.225,
        world_dir: str = None
    ):
        self.obstacles = obstacles if obstacles is not None else []
        self.dt = dt
        self.mass = mass
        self.arm_length = arm_length
        self.world_dir = world_dir

        self.motor = MotorModel(arm_length=arm_length)
        self.controller = QuaternionP2Controller(mass=mass, motor_model=self.motor)
        self.wind_field = VariableSpatialWindField()
        self.dynamics = QuadrotorDynamics(mass=mass)

        if world_dir:
            self.export_sdf_world(world_dir)

    def export_sdf_world(self, output_dir: str) -> str:
        """Exports an SDF 1.6 World file with randomized 3D obstacles and quadrotor."""
        os.makedirs(output_dir, exist_ok=True)
        sdf_path = os.path.join(output_dir, "variable_wind_arena.world")
        
        sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="variable_wind_arena">
    <physics type="ode">
      <max_step_size>{self.dt}</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>{int(1.0 / self.dt)}</real_time_update_rate>
      <gravity>0 0 -9.81</gravity>
    </physics>

    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground Plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- 3D Obstacles -->
"""
        for i, obs in enumerate(self.obstacles):
            cx, cy, cz = obs.center
            r = obs.radius
            if obs.kind == 'cylinder':
                h = obs.height
                sdf_content += f"""
    <model name="obstacle_cyl_{i}">
      <static>true</static>
      <pose>{cx} {cy} {h*0.5} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
        </collision>
        <visual name="vis">
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient><diffuse>0.8 0.2 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>"""
            else:
                sdf_content += f"""
    <model name="obstacle_sph_{i}">
      <static>true</static>
      <pose>{cx} {cy} {cz} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><sphere><radius>{r}</radius></sphere></geometry>
        </collision>
        <visual name="vis">
          <geometry><sphere><radius>{r}</radius></sphere></geometry>
          <material><ambient>0.9 0.5 0.1 1</ambient><diffuse>0.9 0.5 0.1 1</diffuse></material>
        </visual>
      </link>
    </model>"""

        sdf_content += "\n  </world>\n</sdf>\n"

        with open(sdf_path, "w", encoding="utf-8") as f:
            f.write(sdf_content)

        return sdf_path

    def reset(self, initial_pos: np.ndarray = np.array([1.0, 1.0, 1.0])) -> None:
        """Resets physics and controller states."""
        self.controller.reset()
        self.motor.reset()
        self.dynamics.reset(pos=initial_pos)

    def step(
        self,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray,
        acc_ref: np.ndarray,
        yaw_ref: float,
        dist_along_path: float
    ) -> Dict:
        """
        Runs one simulation step with Paper 2 control and distance-dependent wind field.
        """
        pos_meas = self.dynamics.pos.copy()
        quat_meas = self.dynamics.quat.copy()
        vel_meas = self.dynamics.vel.copy()
        omega_meas = self.dynamics.omega.copy()

        # 1. Paper 2 Quaternion Control
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

        # 2. Motor dynamics
        actual_omega = self.motor.step(omega_cmd, self.dt)
        thrust_act, tau_body_act, rotor_forces = self.motor.compute_forces_and_torques(actual_omega)

        # 3. Variable Wind Aerodynamic Wrench
        R_meas = quat_to_rot_matrix(quat_meas)
        f_drag_world, tau_aero_body, v_wind, active_zone = self.wind_field.compute_aerodynamic_wrench(
            drone_pos=pos_meas,
            drone_vel=vel_meas,
            drone_rot_matrix=R_meas,
            distance=dist_along_path,
            dt=self.dt
        )

        # 4. Integrate 6-DOF Dynamics
        new_pos, new_vel, new_quat, new_omega = self.dynamics.step_rk4(
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
            "active_zone_name": active_zone.name,
            "active_zone_desc": active_zone.description,
            "f_drag": f_drag_world,
            "dist_along_path": dist_along_path
        }
