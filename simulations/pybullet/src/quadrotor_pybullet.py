"""
PyBullet Quadrotor Physical Body Wrapper.
Supports native PyBullet C++ multi-body engine and pure-Python PyBullet physics engine fallback.
Constructs a 6-DOF quadrotor matching Paper 2 parameters:
  mass = 1.0 kg
  Ixx = 6.5e-4 kg.m^2, Iyy = 6.5e-4 kg.m^2, Izz = 1.2e-3 kg.m^2
  arm length d = 0.23 m
"""
import numpy as np
from .quaternion_math import QuaternionMath
from .motor_model import MotorModel

try:
    import pybullet as p
    PYBULLET_NATIVE = True
except ImportError:
    PYBULLET_NATIVE = False
    p = None


class PyBulletQuadrotor:

    def __init__(self, physics_environment=None, physics_client=None, start_position=(0, 0, 1),
                 start_orientation_euler=(0, 0, 0), enable_motor_model=True, motor_model=None):

        self.env = physics_environment
        self.client = physics_client if physics_client is not None else (self.env.client if self.env else None)
        self.native = (p is not None and self.client is not None and PYBULLET_NATIVE)

        self.mass = 1.0
        self.Ixx = 6.5e-4
        self.Iyy = 6.5e-4
        self.Izz = 1.2e-3
        self.d   = 0.23
        self.g   = 9.81
        self.inertia = np.diag([self.Ixx, self.Iyy, self.Izz])

        self.enable_motor_model = enable_motor_model
        self.motor_model = motor_model if motor_model is not None else MotorModel(d=self.d)

        self.start_pos = np.array(start_position, dtype=float)
        self.start_euler = np.array(start_orientation_euler, dtype=float)
        self.start_quat_ham = QuaternionMath.from_euler(*self.start_euler)

        if self.native:
            self.body_id = self._create_native_body()
        elif self.env and self.env.fallback:
            self.body_id = self.env.fallback.create_body(
                mass=self.mass, position=self.start_pos, orientation=self.start_quat_ham, inertia=self.inertia
            )
        else:
            # Standalone fallback engine
            from .pybullet_env import PyBulletFallbackEngine
            self.fallback = PyBulletFallbackEngine()
            self.body_id = self.fallback.create_body(
                mass=self.mass, position=self.start_pos, orientation=self.start_quat_ham, inertia=self.inertia
            )

        self.reset(start_position, start_orientation_euler)

        self.motor_speeds = np.zeros(4)
        self.applied_thrust = 0.0
        self.applied_torques = np.zeros(3)
        self.is_saturated = False

    def _create_native_body(self):
        base_col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=self.client)
        base_vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0.2, 0.2, 0.8, 1.0], physicsClientId=self.client)

        link_masses = [0.02, 0.02, 0.02, 0.02]
        link_col_shapes = [p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=0.02, physicsClientId=self.client) for _ in range(4)]
        link_colors = [[1.0, 0.1, 0.1, 1.0], [0.1, 0.8, 0.1, 1.0], [0.1, 0.1, 1.0, 1.0], [0.8, 0.8, 0.1, 1.0]]
        link_vis_shapes = [p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=0.02, rgbaColor=link_colors[i], physicsClientId=self.client) for i in range(4)]
        link_positions = [[self.d, 0, 0], [0, -self.d, 0], [-self.d, 0, 0], [0, self.d, 0]]
        link_orientations = [[0, 0, 0, 1] for _ in range(4)]
        link_inertial_frame_positions = [[0, 0, 0] for _ in range(4)]
        link_inertial_frame_orientations = [[0, 0, 0, 1] for _ in range(4)]
        link_parent_indices = [0, 0, 0, 0]
        link_joint_types = [p.JOINT_FIXED for _ in range(4)]
        link_joint_axes = [[0, 0, 1] for _ in range(4)]

        q_pb = QuaternionMath.hamilton_to_pybullet(self.start_quat_ham)

        body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=base_col_shape,
            baseVisualShapeIndex=base_vis_shape,
            basePosition=self.start_pos.tolist(),
            baseOrientation=q_pb.tolist(),
            baseInertialFramePosition=[0, 0, 0],
            baseInertialFrameOrientation=[0, 0, 0, 1],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col_shapes,
            linkVisualShapeIndices=link_vis_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            physicsClientId=self.client
        )
        p.changeDynamics(body_id, -1, linearDamping=0.01, angularDamping=0.01, physicsClientId=self.client)
        return body_id

    def reset(self, position=None, euler=None):
        pos = np.array(position if position is not None else self.start_pos, dtype=float)
        eul = np.array(euler if euler is not None else self.start_euler, dtype=float)

        q_ham = QuaternionMath.from_euler(*eul)

        if self.native:
            q_pb = QuaternionMath.hamilton_to_pybullet(q_ham)
            p.resetBasePositionAndOrientation(self.body_id, pos.tolist(), q_pb.tolist(), physicsClientId=self.client)
            p.resetBaseVelocity(self.body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        elif self.env and self.env.fallback:
            self.env.fallback.reset_body(self.body_id, pos, q_ham)
        elif hasattr(self, 'fallback'):
            self.fallback.reset_body(self.body_id, pos, q_ham)

        self.motor_speeds = np.zeros(4)
        self.applied_thrust = 0.0
        self.applied_torques = np.zeros(3)
        self.is_saturated = False

    def get_state(self):
        if self.native:
            pos, q_pb = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client)
            vel_world, omega_world = p.getBaseVelocity(self.body_id, physicsClientId=self.client)
            q_ham = QuaternionMath.pybullet_to_hamilton(q_pb)
            R_world_from_body = QuaternionMath.to_rotation_matrix(q_ham)
            omega_body = R_world_from_body.T @ np.array(omega_world)
            return np.array(pos), np.array(vel_world), q_ham, omega_body
        elif self.env and self.env.fallback:
            pos, q_ham, vel_world, omega_world = self.env.fallback.get_body_state(self.body_id)
            R_world_from_body = QuaternionMath.to_rotation_matrix(q_ham)
            omega_body = R_world_from_body.T @ omega_world
            return pos, vel_world, q_ham, omega_body
        elif hasattr(self, 'fallback'):
            pos, q_ham, vel_world, omega_world = self.fallback.get_body_state(self.body_id)
            R_world_from_body = QuaternionMath.to_rotation_matrix(q_ham)
            omega_body = R_world_from_body.T @ omega_world
            return pos, vel_world, q_ham, omega_body

    def apply_control_inputs(self, total_thrust_cmd, torques_cmd, tau_dist=None, force_dist=None):
        if self.enable_motor_model:
            motor_w, act_thrust, act_torques, saturated = self.motor_model.allocate_and_limit(
                total_thrust_cmd, torques_cmd
            )
            self.motor_speeds = motor_w
            self.applied_thrust = act_thrust
            self.applied_torques = act_torques
            self.is_saturated = saturated
        else:
            self.applied_thrust = max(0.0, float(total_thrust_cmd))
            self.applied_torques = np.clip(np.asarray(torques_cmd, dtype=float), -4.0, 4.0)
            self.is_saturated = False

        if tau_dist is not None:
            self.applied_torques += np.asarray(tau_dist, dtype=float)

        if self.native:
            p.applyExternalForce(
                objectUniqueId=self.body_id, linkIndex=-1, forceObj=[0.0, 0.0, float(self.applied_thrust)],
                posObj=[0.0, 0.0, 0.0], flags=p.LINK_FRAME, physicsClientId=self.client
            )
            p.applyExternalTorque(
                objectUniqueId=self.body_id, linkIndex=-1, torqueObj=self.applied_torques.tolist(),
                flags=p.LINK_FRAME, physicsClientId=self.client
            )
            if force_dist is not None:
                p.applyExternalForce(
                    objectUniqueId=self.body_id, linkIndex=-1, forceObj=force_dist.tolist(),
                    posObj=[0.0, 0.0, 0.0], flags=p.WORLD_FRAME, physicsClientId=self.client
                )
        elif self.env and self.env.fallback:
            self.env.fallback.apply_force(self.body_id, [0.0, 0.0, float(self.applied_thrust)], is_body_frame=True)
            self.env.fallback.apply_torque(self.body_id, self.applied_torques, is_body_frame=True)
            if force_dist is not None:
                self.env.fallback.apply_force(self.body_id, force_dist, is_body_frame=False)
        elif hasattr(self, 'fallback'):
            self.fallback.apply_force(self.body_id, [0.0, 0.0, float(self.applied_thrust)], is_body_frame=True)
            self.fallback.apply_torque(self.body_id, self.applied_torques, is_body_frame=True)
            if force_dist is not None:
                self.fallback.apply_force(self.body_id, force_dist, is_body_frame=False)
