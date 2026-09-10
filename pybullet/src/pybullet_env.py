"""
PyBullet 3D Obstacle Environment Generator & Physics Simulator Engine.
Supports native PyBullet engine when installed, and pure-Python PyBullet physics simulator fallback.
Constructs 3D static cylinders and cuboids matching Environment3D, ground plane,
and draws real-time 3D debug lines for global path, smoothed spline, and drone trajectory.
"""
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_NATIVE = True
except ImportError:
    PYBULLET_NATIVE = False
    p = None
    pybullet_data = None


class PyBulletFallbackEngine:
    """Pure-Python PyBullet physics simulator engine fallback."""

    def __init__(self, dt=0.002, gravity=-9.81):
        self.dt = float(dt)
        self.gravity = np.array([0.0, 0.0, float(gravity)], dtype=float)
        self.bodies = {}
        self.next_body_id = 0

    def create_body(self, mass=1.0, position=(0, 0, 0), orientation=(0, 0, 0, 1), inertia=None):
        body_id = self.next_body_id
        self.next_body_id += 1

        if inertia is None:
            inertia = np.diag([6.5e-4, 6.5e-4, 1.2e-3])

        self.bodies[body_id] = {
            'mass': float(mass),
            'pos': np.array(position, dtype=float),
            'quat': np.array(orientation, dtype=float),
            'vel': np.zeros(3, dtype=float),
            'omega': np.zeros(3, dtype=float),
            'inertia': np.asarray(inertia, dtype=float),
            'inv_inertia': np.linalg.inv(np.asarray(inertia, dtype=float)),
            'forces_world': np.zeros(3, dtype=float),
            'torques_body': np.zeros(3, dtype=float)
        }
        return body_id

    def apply_force(self, body_id, force, is_body_frame=False):
        if body_id not in self.bodies:
            return
        b = self.bodies[body_id]
        force = np.asarray(force, dtype=float)

        if is_body_frame:
            from .quaternion_math import QuaternionMath
            R = QuaternionMath.to_rotation_matrix(b['quat'])
            force_world = R @ force
        else:
            force_world = force

        b['forces_world'] += force_world

    def apply_torque(self, body_id, torque, is_body_frame=True):
        if body_id not in self.bodies:
            return
        b = self.bodies[body_id]
        b['torques_body'] += np.asarray(torque, dtype=float)

    def step(self):
        dt = self.dt
        from .quaternion_math import QuaternionMath

        for b_id, b in self.bodies.items():
            if b['mass'] <= 0.0:
                continue  # Static object

            # Linear dynamics
            acc_world = (b['forces_world'] / b['mass']) + self.gravity
            b['vel'] += acc_world * dt
            b['pos'] += b['vel'] * dt

            # Rotational dynamics (Paper 2 Eq. 18): I*domega/dt = tau - omega x (I*omega)
            gyro = np.cross(b['omega'], b['inertia'] @ b['omega'])
            alpha = b['inv_inertia'] @ (b['torques_body'] - gyro)
            b['omega'] += alpha * dt

            # Quaternion kinematics: dq/dt = 0.5 * q x [0, omega]
            w_quat = np.array([0.0, b['omega'][0], b['omega'][1], b['omega'][2]])
            dq = 0.5 * QuaternionMath.multiply(b['quat'], w_quat)
            b['quat'] = QuaternionMath.normalize(b['quat'] + dq * dt)

            # Clear accumulated forces & torques
            b['forces_world'].fill(0.0)
            b['torques_body'].fill(0.0)

    def get_body_state(self, body_id):
        if body_id not in self.bodies:
            return np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3)
        b = self.bodies[body_id]
        from .quaternion_math import QuaternionMath
        R = QuaternionMath.to_rotation_matrix(b['quat'])
        omega_world = R @ b['omega']
        return b['pos'].copy(), b['quat'].copy(), b['vel'].copy(), omega_world.copy()

    def reset_body(self, body_id, pos, quat):
        if body_id in self.bodies:
            b = self.bodies[body_id]
            b['pos'] = np.array(pos, dtype=float)
            b['quat'] = np.array(quat, dtype=float)
            b['vel'].fill(0.0)
            b['omega'].fill(0.0)


class PyBulletEnvironment:

    def __init__(self, gui=False, environment_3d=None, dt=0.002):
        self.gui = gui
        self.dt = dt
        self.env = environment_3d

        self.native = PYBULLET_NATIVE
        self.client = None
        self.fallback = None

        if self.native:
            try:
                mode = p.GUI if gui else p.DIRECT
                self.client = p.connect(mode)
                p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
                p.setGravity(0, 0, -9.81, physicsClientId=self.client)
                p.setTimeStep(self.dt, physicsClientId=self.client)
                if self.gui:
                    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=2500, cameraYaw=45, cameraPitch=-30,
                        cameraTargetPosition=[1000, 1000, 250], physicsClientId=self.client
                    )
                self.plane_id = p.loadURDF("plane.urdf", [1000, 1000, 0], physicsClientId=self.client)
            except Exception:
                self.native = False

        if not self.native:
            self.fallback = PyBulletFallbackEngine(dt=dt)

        self.obstacle_ids = []
        if self.env is not None:
            self._create_obstacles()

    def _create_obstacles(self):
        for obs in self.env.obstacles:
            if self.native:
                if obs.shape == 'cylinder':
                    r, h = obs.dimensions
                    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=self.client)
                    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=[0.2, 0.7, 0.3, 0.8], physicsClientId=self.client)
                    obs_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=obs.position.tolist(), physicsClientId=self.client)
                    self.obstacle_ids.append(obs_id)
                elif obs.shape == 'cuboid':
                    l, w, h = obs.dimensions
                    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2], physicsClientId=self.client)
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2], rgbaColor=[0.2, 0.4, 0.8, 0.8], physicsClientId=self.client)
                    obs_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=obs.position.tolist(), physicsClientId=self.client)
                    self.obstacle_ids.append(obs_id)
            elif self.fallback:
                obs_id = self.fallback.create_body(mass=0.0, position=obs.position)
                self.obstacle_ids.append(obs_id)

    def draw_path_lines(self, waypoints=None, spline_pts=None):
        if self.native and self.gui:
            if waypoints is not None:
                wps = np.asarray(waypoints)
                for i in range(len(wps) - 1):
                    p.addUserDebugLine(wps[i].tolist(), wps[i+1].tolist(), lineColorRGB=[0.9, 0.9, 0.2], lineWidth=2, physicsClientId=self.client)
            if spline_pts is not None:
                pts = np.asarray(spline_pts)
                for i in range(len(pts) - 1):
                    p.addUserDebugLine(pts[i].tolist(), pts[i+1].tolist(), lineColorRGB=[1.0, 0.1, 0.1], lineWidth=4, physicsClientId=self.client)

    def draw_trajectory_segment(self, p1, p2, color=[0.1, 0.9, 0.1], width=3):
        if self.native and self.gui:
            p.addUserDebugLine(np.asarray(p1).tolist(), np.asarray(p2).tolist(), lineColorRGB=color, lineWidth=width, physicsClientId=self.client)

    def step(self):
        if self.native:
            p.stepSimulation(physicsClientId=self.client)
        elif self.fallback:
            self.fallback.step()

    def close(self):
        if self.native and self.client is not None and p.isConnected(self.client):
            p.disconnect(self.client)
