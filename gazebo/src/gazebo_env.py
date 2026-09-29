"""
Gazebo Environment Manager & World Generator.
Procedurally converts Environment3D (cylinders & cuboids) into Gazebo SDF worlds,
manages native Gazebo process launching (when installed), and provides a
6-DOF rigid-body simulation engine (RK4, obstacle and ground contact detection).

NOTE: All flight dynamics are computed by GazeboFallbackEngine in Python. The native
Gazebo process (if launched) only loads the generated world for visualisation; state
and control are not exchanged with it.
"""
import os
import shutil
import subprocess
import numpy as np
from .quaternion_math import QuaternionMath


MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def generate_gazebo_world_sdf(environment_3d, output_path, world_name="drone_airspace",
                              start_position=None):
    """
    Generates a full Gazebo SDF 1.8 world file populated with obstacles from Environment3D.
    The quadrotor is spawned at start_position (defaults to environment_3d.start).
    Requires GZ_SIM_RESOURCE_PATH to include gazebo/models so model://quadrotor resolves
    (GazeboEnvironment.launch sets this automatically).
    """
    if start_position is None:
        start_position = environment_3d.start
    obstacles_xml = []
    for idx, obs in enumerate(environment_3d.obstacles):
        pos = obs.position
        if obs.shape == 'cylinder':
            radius, height = obs.dimensions
            z_center = height / 2.0
            obs_xml = f"""
    <!-- Cylinder Tower {idx} -->
    <model name="cylinder_obs_{idx}">
      <static>true</static>
      <pose>{pos[0]:.2f} {pos[1]:.2f} {z_center:.2f} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><cylinder><radius>{radius:.2f}</radius><length>{height:.2f}</length></cylinder></geometry>
        </collision>
        <visual name="vis">
          <geometry><cylinder><radius>{radius:.2f}</radius><length>{height:.2f}</length></cylinder></geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.85 0.25 0.25 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""
        else:
            length, width, height = obs.dimensions
            z_center = height / 2.0
            obs_xml = f"""
    <!-- Cuboid Building {idx} -->
    <model name="cuboid_obs_{idx}">
      <static>true</static>
      <pose>{pos[0]:.2f} {pos[1]:.2f} {z_center:.2f} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{length:.2f} {width:.2f} {height:.2f}</size></box></geometry>
        </collision>
        <visual name="vis">
          <geometry><box><size>{length:.2f} {width:.2f} {height:.2f}</size></box></geometry>
          <material>
            <ambient>0.2 0.6 0.3 1</ambient>
            <diffuse>0.25 0.7 0.35 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""
        obstacles_xml.append(obs_xml)

    joined_obstacles = "\n".join(obstacles_xml)

    sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="{world_name}">
    <physics name="dart_physics" type="dart">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 500 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="col">
          <geometry><plane><normal>0 0 1</normal><size>3000 3000</size></plane></geometry>
        </collision>
        <visual name="vis">
          <geometry><plane><normal>0 0 1</normal><size>3000 3000</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material>
        </visual>
      </link>
    </model>

{joined_obstacles}

    <include>
      <uri>model://quadrotor</uri>
      <name>quadrotor</name>
      <pose>{start_position[0]:.2f} {start_position[1]:.2f} {start_position[2]:.2f} 0 0 0</pose>
    </include>
  </world>
</sdf>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(sdf_content)
    return output_path


class GazeboFallbackEngine:
    """
    6-DOF rigid-body physics engine (RK4) with obstacle and ground contact detection.
    This is the engine that actually produces the flight dynamics; native Gazebo is not
    coupled to it.
    """

    def __init__(self, environment_3d=None, dt=0.002):
        self.env_3d = environment_3d
        self.dt = dt
        self.time = 0.0

        # Contact reporting (detection only; dynamics are not altered by obstacle hits)
        self.collided = False
        self.collision_count = 0
        self.first_collision = None   # (time, obstacle_index, position)
        self.ground_contact_steps = 0

        # State: position, velocity, orientation [w,x,y,z], angular velocity [p,q,r]
        self.position = np.array([0.0, 0.0, 1.0], dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega = np.zeros(3, dtype=float)

        # Vehicle physical constants
        self.mass = 1.0
        self.Ixx = 6.5e-4
        self.Iyy = 6.5e-4
        self.Izz = 1.2e-3
        self.inertia = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.g = 9.81

    def reset(self, position=None, orientation=None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        if orientation is not None:
            self.orientation = QuaternionMath.normalize(orientation)
        self.velocity = np.zeros(3, dtype=float)
        self.omega = np.zeros(3, dtype=float)
        self.time = 0.0
        self.collided = False
        self.collision_count = 0
        self.first_collision = None
        self.ground_contact_steps = 0

    def _find_obstacle_hit(self, point):
        """Returns the index of the obstacle containing point (no safety buffer), else None."""
        if self.env_3d is None:
            return None
        for idx, obs in enumerate(self.env_3d.obstacles):
            delta = point - obs.position
            if obs.shape == 'cylinder':
                radius, height = obs.dimensions
                hit = np.hypot(delta[0], delta[1]) <= radius and abs(delta[2]) <= height / 2.0
            else:
                hit = bool(np.all(np.abs(delta) <= obs.dimensions / 2.0))
            if hit:
                return idx
        return None

    def step(self, total_thrust, torques_body, wind_force_world=None, wind_torque_body=None):
        """
        Steps 6-DOF dynamics using 4th-order Runge-Kutta (RK4).
        total_thrust: Scalar upward thrust in body Z (N)
        torques_body: 3-element vector [tau_x, tau_y, tau_z] in body frame (Nm)
        """
        if wind_force_world is None:
            wind_force_world = np.zeros(3)
        if wind_torque_body is None:
            wind_torque_body = np.zeros(3)

        dt = self.dt

        # Helper to compute derivatives: d(pos)/dt, d(vel)/dt, d(q)/dt, d(omega)/dt
        def state_derivatives(pos, vel, q, omega):
            R = QuaternionMath.to_rotation_matrix(q)
            # Body z-axis points upward in standard quadrotor definition
            z_body = R[:, 2]
            f_thrust_world = total_thrust * z_body
            gravity = np.array([0.0, 0.0, -self.mass * self.g])
            acc = (f_thrust_world + gravity + wind_force_world) / self.mass

            # Rotational dynamics: I * w_dot = tau - w x (I * w)
            tau_total = torques_body + wind_torque_body
            gyro = np.cross(omega, self.inertia @ omega)
            omega_dot = self.inv_inertia @ (tau_total - gyro)

            # Quaternion kinematics: q_dot = 0.5 * q x [0, omega]
            omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
            q_dot = 0.5 * QuaternionMath.multiply(q, omega_quat)

            return vel, acc, q_dot, omega_dot

        # RK4 Integration
        v1, a1, q_dot1, w_dot1 = state_derivatives(self.position, self.velocity, self.orientation, self.omega)

        p2 = self.position + 0.5 * dt * v1
        v2_in = self.velocity + 0.5 * dt * a1
        q2_in = QuaternionMath.normalize(self.orientation + 0.5 * dt * q_dot1)
        w2_in = self.omega + 0.5 * dt * w_dot1
        v2, a2, q_dot2, w_dot2 = state_derivatives(p2, v2_in, q2_in, w2_in)

        p3 = self.position + 0.5 * dt * v2
        v3_in = self.velocity + 0.5 * dt * a2
        q3_in = QuaternionMath.normalize(self.orientation + 0.5 * dt * q_dot2)
        w3_in = self.omega + 0.5 * dt * w_dot2
        v3, a3, q_dot3, w_dot3 = state_derivatives(p3, v3_in, q3_in, w3_in)

        p4 = self.position + dt * v3
        v4_in = self.velocity + dt * a3
        q4_in = QuaternionMath.normalize(self.orientation + dt * q_dot3)
        w4_in = self.omega + dt * w_dot3
        v4, a4, q_dot4, w_dot4 = state_derivatives(p4, v4_in, q4_in, w4_in)

        # Combine
        self.position += (dt / 6.0) * (v1 + 2.0*v2 + 2.0*v3 + v4)
        self.velocity += (dt / 6.0) * (a1 + 2.0*a2 + 2.0*a3 + a4)
        self.orientation = QuaternionMath.normalize(
            self.orientation + (dt / 6.0) * (q_dot1 + 2.0*q_dot2 + 2.0*q_dot3 + q_dot4)
        )
        self.omega += (dt / 6.0) * (w_dot1 + 2.0*w_dot2 + 2.0*w_dot3 + w_dot4)
        self.time += dt

        # Ground contact constraint
        if self.position[2] < 0.0:
            self.position[2] = 0.0
            self.velocity[2] = max(0.0, self.velocity[2])
        if self.position[2] <= 0.0:
            self.ground_contact_steps += 1

        # Obstacle contact detection
        hit = self._find_obstacle_hit(self.position)
        if hit is not None:
            self.collided = True
            self.collision_count += 1
            if self.first_collision is None:
                self.first_collision = (self.time, hit, self.position.copy())

        return self.position.copy(), self.velocity.copy(), self.orientation.copy(), self.omega.copy()


class GazeboEnvironment:
    """
    Manages Gazebo simulation environment setup, world generation, and process lifecycle.
    """

    def __init__(self, world_sdf_path=None, environment_3d=None, gui=False, dt=0.002,
                 world_name="drone_airspace"):
        self.dt = dt
        self.gui = gui
        self.env_3d = environment_3d
        self.world_path = world_sdf_path
        self.world_name = world_name
        self.process = None

        # Native Gazebo CLI: modern `gz sim` (Harmonic+) or `ign gazebo` (Fortress).
        # Classic `gazebo` cannot load SDF 1.8 / gz-sim plugins, so it is not used.
        if shutil.which("gz"):
            self.cli_cmd = ["gz", "sim"]
        elif shutil.which("ign"):
            self.cli_cmd = ["ign", "gazebo"]
        else:
            self.cli_cmd = None
        self.has_gz = self.cli_cmd is not None
        self.cli_name = self.cli_cmd[0] if self.cli_cmd else None

        # Always initialize fallback engine for unified access
        self.fallback = GazeboFallbackEngine(environment_3d=self.env_3d, dt=self.dt)

        # Procedurally build world file if an Environment3D is supplied
        if self.env_3d is not None and (self.world_path is None or not os.path.exists(self.world_path)):
            default_world = os.path.join(os.path.dirname(__file__), "..", "worlds", "generated_airspace.sdf")
            self.world_path = generate_gazebo_world_sdf(self.env_3d, default_world)

    def launch(self):
        """
        Launches native Gazebo (world visualisation only) if installed. Returns True when a
        native process was started; otherwise the Python physics engine is used alone.
        gui=True opens the GUI; gui=False runs headless (server only).
        """
        if self.has_gz and self.world_path and os.path.exists(self.world_path):
            cmd = self.cli_cmd + ["-r"] + ([] if self.gui else ["-s"]) + [self.world_path]
            env = os.environ.copy()
            for var in ("GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH"):
                env[var] = os.pathsep.join(filter(None, [MODELS_DIR, env.get(var, "")]))
            try:
                self.process = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"[Gazebo] Native process launched (world view only): {' '.join(cmd)}")
                print("[Gazebo] Flight dynamics are computed by the Python 6-DOF engine.")
                return True
            except Exception as e:
                print(f"[Gazebo Warning] Could not launch native process: {e}")
                self.process = None
        else:
            print("[Gazebo] Native Gazebo not found; using the Python 6-DOF physics engine.")
        return False

    def set_pose(self, model_name, position, quaternion):
        """
        Teleports `model_name` to `position` (xyz, meters) / `quaternion` ([w, x, y, z], Hamilton)
        inside the running native Gazebo process, via the `/world/<name>/set_pose` service that
        the `gz-sim-user-commands-system` plugin exposes (already loaded in every world this
        module generates). This drives the model's pose directly -- it does not apply forces or
        velocities in native Gazebo's own physics engine -- so it is a best-effort *visual* pose
        stream for realtime playback, not a physics co-simulation: the Python 6-DOF engine
        (GazeboFallbackEngine / GazeboQuadrotor) remains the sole source of truth for dynamics.

        Returns True if the service call round-tripped inside the timeout, else False (native
        Gazebo not running, or this call was slower than the timeout). A False return is safe to
        ignore and keep simulating -- the next call simply publishes a newer pose.
        """
        if not self.has_gz or self.process is None:
            return False
        w, x, y, z = quaternion
        req = (f'name: "{model_name}" '
               f'position: {{x: {position[0]:.4f} y: {position[1]:.4f} z: {position[2]:.4f}}} '
               f'orientation: {{x: {x:.6f} y: {y:.6f} z: {z:.6f} w: {w:.6f}}}')
        cmd = [self.cli_name, "service",
               "-s", f"/world/{self.world_name}/set_pose",
               "--reqtype", "gz.msgs.Pose",
               "--reptype", "gz.msgs.Boolean",
               "--timeout", "200",
               "--req", req]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.5)
            return True
        except Exception:
            return False

    def close(self):
        """Terminates native Gazebo process if active."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
