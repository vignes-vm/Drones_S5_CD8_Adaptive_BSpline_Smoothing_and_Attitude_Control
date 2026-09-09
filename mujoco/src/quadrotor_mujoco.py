"""
MuJoCo Quadrotor Simulation Wrapper
Provides clean interface to MuJoCo physics engine with crash-safe live 3D GUI visualizer (mujoco.viewer).
"""

import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from src.quaternion_math import quat_normalize, quat_to_rot_matrix

class QuadrotorMuJoCoSim:
    def __init__(self, xml_path: str = "models/quadrotor.xml", gui: bool = False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MJCF file not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.gui = gui
        self.viewer = None

        if self.gui:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"[Warning] Could not launch native 3D GUI viewer: {e}")
                self.viewer = None
                self.gui = False

    def is_viewer_alive(self) -> bool:
        """Returns True if viewer is active or GUI mode is disabled."""
        if self.gui and self.viewer is not None:
            try:
                return self.viewer.is_running()
            except Exception:
                return False
        return True

    def reset(self, pos: np.ndarray = None, quat: np.ndarray = None, vel: np.ndarray = None, omega: np.ndarray = None):
        """Resets quadrotor state."""
        mujoco.mj_resetData(self.model, self.data)

        if pos is not None:
            self.data.qpos[0:3] = pos
        else:
            self.data.qpos[0:3] = np.array([0.0, 0.0, 1.0])

        if quat is not None:
            self.data.qpos[3:7] = quat_normalize(quat)
        else:
            self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        if vel is not None:
            self.data.qvel[0:3] = vel
        if omega is not None:
            self.data.qvel[3:6] = omega

        mujoco.mj_forward(self.model, self.data)

        if self.gui and self.viewer is not None:
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
            except Exception:
                pass

    def get_state(self) -> dict:
        """
        Returns full current quadrotor state:
        - position: (3,)
        - velocity: (3,)
        - quat: (4,) [q0, q1, q2, q3]
        - omega: (3,) [wx, wy, wz] in the BODY frame (MuJoCo freejoint convention)
        - time: float
        """
        pos = np.array(self.data.qpos[0:3], dtype=np.float64)
        quat = np.array(self.data.qpos[3:7], dtype=np.float64)
        vel = np.array(self.data.qvel[0:3], dtype=np.float64)
        omega = np.array(self.data.qvel[3:6], dtype=np.float64)
        t = float(self.data.time)

        return {
            'pos': pos,
            'vel': vel,
            'quat': quat_normalize(quat),
            'omega': omega,
            'time': t
        }

    def step(self,
             tau: np.ndarray,
             thrust_force: float = 0.0,
             f_ext_world: np.ndarray = None,
             tau_ext_body: np.ndarray = None,
             real_time: bool = False):
        """
        Steps the physics by one dt under the total applied wrench.

        Frame contract (MuJoCo `freejoint`, verified in
        tests/test_wrench_frames.py::test_freejoint_rotational_dofs_are_body_frame):
          qfrc_applied[0:3] -- force,  WORLD frame
          qfrc_applied[3:6] -- torque, BODY frame

        Args:
            tau:          (3,) control torque, BODY frame [Mx, My, Mz].
            thrust_force: scalar thrust magnitude along the body z-axis.
            f_ext_world:  (3,) external disturbance force, WORLD frame (e.g. wind
                          drag). Pass it here rather than writing qfrc_applied
                          directly -- this method owns that buffer and any value
                          written into it from outside is overwritten below.
            tau_ext_body: (3,) external disturbance torque, BODY frame.
        """
        step_start = time.time()

        # Convert body vertical thrust to a world-frame force.
        q = self.data.qpos[3:7]
        R = quat_to_rot_matrix(q)
        world_thrust = R[:, 2] * thrust_force

        # Compose the COMPLETE wrench here, in one place. This buffer is
        # assigned (not accumulated) every step, so every contribution has to
        # arrive through this call. A previous version had callers write the
        # wind force into qfrc_applied[0:3] before calling step(), where the
        # thrust assignment silently discarded it before mj_step ever ran.
        total_force = world_thrust
        if f_ext_world is not None:
            total_force = total_force + np.asarray(f_ext_world, dtype=np.float64)

        total_torque = np.asarray(tau, dtype=np.float64)
        if tau_ext_body is not None:
            total_torque = total_torque + np.asarray(tau_ext_body, dtype=np.float64)

        self.data.qfrc_applied[0:3] = total_force
        self.data.qfrc_applied[3:6] = total_torque

        mujoco.mj_step(self.model, self.data)

        if self.gui and self.viewer is not None:
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
                    if real_time:
                        elapsed = time.time() - step_start
                        if self.dt > elapsed:
                            time.sleep(self.dt - elapsed)
            except Exception:
                pass

    def keep_alive_until_closed(self):
        """Keeps GUI window open after simulation completes so user can interactively view the scene."""
        if self.gui and self.viewer is not None:
            print("\n[Interactive 3D Viewer Active] Use mouse to orbit camera/zoom, space to pause. Close window to exit.")
            try:
                while self.viewer.is_running():
                    self.viewer.sync()
                    time.sleep(0.02)
            except Exception:
                pass
            finally:
                self.close()

    def close(self):
        """Closes 3D GUI viewer cleanly."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
