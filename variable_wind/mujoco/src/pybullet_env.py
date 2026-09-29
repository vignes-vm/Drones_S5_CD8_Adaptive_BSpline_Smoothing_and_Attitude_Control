r"""
PyBullet Simulation Environment for Variable Wind Quadrotor.
Spawns 3D arena with randomized obstacles, visualizes 3D B-spline path,
simulates quadrotor physics, and applies distance-dependent wind drag/moments.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    p = None
    pybullet_data = None
    HAS_PYBULLET = False

from .quaternion_math import (
    quat_norm,
    quat_to_rot_matrix,
    rot_matrix_to_quat,
    euler_to_quat
)
from .obstacle_planner import Obstacle3D
from .variable_wind_field import VariableSpatialWindField
from .quaternion_controller import QuaternionP2Controller
from .motor_model import MotorModel
from .quadrotor_dynamics import QuadrotorDynamics


class PyBulletVariableWindEnv:
    """
    PyBullet environment with dynamic 3D obstacles, trajectory lines, and variable wind wrench.
    """
    def __init__(
        self,
        obstacles: List[Obstacle3D] = None,
        use_gui: bool = False,
        dt: float = 0.005,
        mass: float = 1.0,
        arm_length: float = 0.225
    ):
        self.obstacles = obstacles if obstacles is not None else []
        self.use_gui = use_gui and HAS_PYBULLET
        self.dt = dt
        self.mass = mass
        self.arm_length = arm_length

        self.physics_client = None
        self.drone_id = None
        self.obstacle_ids = []
        self.use_native_pybullet = False

        self.motor = MotorModel(arm_length=arm_length)
        self.controller = QuaternionP2Controller(mass=mass, motor_model=self.motor)
        self.wind_field = VariableSpatialWindField()
        self.fallback_dynamics = QuadrotorDynamics(mass=mass)

        self._connect_and_build()

    def _connect_and_build(self) -> None:
        """Initializes PyBullet physics engine and arena."""
        if not HAS_PYBULLET:
            self.use_native_pybullet = False
            return

        try:
            mode = p.GUI if self.use_gui else p.DIRECT
            self.physics_client = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.dt)

            if self.use_gui:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
                p.resetDebugVisualizerCamera(
                    cameraDistance=25.0,
                    cameraYaw=45.0,
                    cameraPitch=-30.0,
                    cameraTargetPosition=[12.0, 12.0, 5.0]
                )

            # Ground plane
            p.loadURDF("plane.urdf")

            # Build randomized 3D obstacles in PyBullet
            self._spawn_obstacles()

            # Build quadrotor dynamic body
            self._spawn_quadrotor()
            self.use_native_pybullet = True
        except Exception as e:
            print(f"Notice: PyBullet initialization notice ({e}). Using robust 6-DOF fallback.")
            self.use_native_pybullet = False

    def _spawn_obstacles(self) -> None:
        """Spawns cylinder and sphere obstacles in PyBullet."""
        if not self.use_native_pybullet:
            return
        for obs in self.obstacles:
            cx, cy, cz = obs.center
            r = obs.radius
            if obs.kind == 'cylinder':
                h = obs.height
                col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
                vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=[0.85, 0.25, 0.25, 0.8])
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[cx, cy, h * 0.5]
                )
            else:
                col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
                vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[0.95, 0.55, 0.15, 0.8])
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[cx, cy, cz]
                )
            self.obstacle_ids.append(body_id)

    def _spawn_quadrotor(self) -> None:
        """Spawns 6-DOF quadrotor multi-body."""
        if not self.use_native_pybullet:
            return
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.arm_length, self.arm_length, 0.04])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.arm_length, self.arm_length, 0.04], rgbaColor=[0.1, 0.35, 0.85, 1.0])
        
        self.drone_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[1.0, 1.0, 1.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0] # pybullet is [x, y, z, w]
        )
        
        # Linear and angular damping
        p.changeDynamics(self.drone_id, -1, linearDamping=0.01, angularDamping=0.01)

    def draw_reference_trajectory(self, trajectory_pts: np.ndarray, color: List[float] = [0, 1, 0]) -> None:
        """Draws reference B-spline path in PyBullet GUI window."""
        if not (self.use_gui and self.use_native_pybullet):
            return
        for i in range(len(trajectory_pts) - 1):
            p.addUserDebugLine(
                trajectory_pts[i].tolist(),
                trajectory_pts[i+1].tolist(),
                lineColorRGB=color,
                lineWidth=2.5,
                lifeTime=0
            )

    def reset(self, initial_pos: np.ndarray = np.array([1.0, 1.0, 1.0])) -> None:
        """Resets quadrotor state."""
        self.controller.reset()
        self.motor.reset()
        self.fallback_dynamics.reset(pos=initial_pos)

        if self.use_native_pybullet and self.drone_id is not None:
            p.resetBasePositionAndOrientation(
                self.drone_id,
                initial_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0] # [x, y, z, w]
            )
            p.resetBaseVelocity(self.drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def step(
        self,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray,
        acc_ref: np.ndarray,
        yaw_ref: float,
        dist_along_path: float
    ) -> Dict:
        """
        Runs one step of PyBullet physics with Paper 2 control and variable wind.
        """
        if self.use_native_pybullet and self.drone_id is not None:
            pos_tuple, quat_tuple = p.getBasePositionAndOrientation(self.drone_id)
            vel_tuple, omega_tuple = p.getBaseVelocity(self.drone_id)

            pos_meas = np.array(pos_tuple, dtype=np.float64)
            # Convert PyBullet quat [x, y, z, w] to scalar-first [w, x, y, z]
            quat_meas = np.array([quat_tuple[3], quat_tuple[0], quat_tuple[1], quat_tuple[2]], dtype=np.float64)
            quat_meas = quat_norm(quat_meas)
            vel_meas = np.array(vel_tuple, dtype=np.float64)
            omega_meas = np.array(omega_tuple, dtype=np.float64)
        else:
            pos_meas = self.fallback_dynamics.pos.copy()
            quat_meas = self.fallback_dynamics.quat.copy()
            vel_meas = self.fallback_dynamics.vel.copy()
            omega_meas = self.fallback_dynamics.omega.copy()

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

        # 3. Variable Spatial Wind Aerodynamic Wrench
        R_meas = quat_to_rot_matrix(quat_meas)
        f_drag_world, tau_aero_body, v_wind, active_zone = self.wind_field.compute_aerodynamic_wrench(
            drone_pos=pos_meas,
            drone_vel=vel_meas,
            drone_rot_matrix=R_meas,
            distance=dist_along_path,
            dt=self.dt
        )

        # 4. Integrate Physics
        if self.use_native_pybullet and self.drone_id is not None:
            thrust_world = R_meas @ np.array([0.0, 0.0, thrust_act])
            total_force_world = thrust_world + f_drag_world
            tau_world = R_meas @ (tau_body_act + tau_aero_body)

            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=-1,
                forceObj=total_force_world.tolist(),
                posObj=pos_meas.tolist(),
                flags=p.WORLD_FRAME
            )
            p.applyExternalTorque(
                objectUniqueId=self.drone_id,
                linkIndex=-1,
                torqueObj=tau_world.tolist(),
                flags=p.WORLD_FRAME
            )

            p.stepSimulation()

            new_pos_t, new_quat_t = p.getBasePositionAndOrientation(self.drone_id)
            new_vel_t, new_omega_t = p.getBaseVelocity(self.drone_id)
            new_pos = np.array(new_pos_t, dtype=np.float64)
            new_vel = np.array(new_vel_t, dtype=np.float64)
            new_quat = np.array([new_quat_t[3], new_quat_t[0], new_quat_t[1], new_quat_t[2]], dtype=np.float64)
            new_omega = np.array(new_omega_t, dtype=np.float64)
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
            "active_zone_name": active_zone.name,
            "active_zone_desc": active_zone.description,
            "f_drag": f_drag_world,
            "dist_along_path": dist_along_path
        }

    def close(self) -> None:
        """Disconnects PyBullet physics."""
        if HAS_PYBULLET and self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None
