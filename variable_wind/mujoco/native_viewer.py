r"""
Native MuJoCo Viewer for Variable Wind Quadrotor Simulation.
Opens the official native MuJoCo interactive visualizer window with:
- 3D Obstacle geoms (cylinders and spheres)
- 6-DOF Quadrotor fuselage and 4 thruster sites
- Real-time physics stepping with distance-dependent variable wind disturbance
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.obstacle_planner import ObstacleEnvironment, AStar3DPlanner, GAAdaptiveBSplineSmoother, TrajectoryGenerator
from src.mujoco_env import MuJoCoVariableWindEnv, HAS_MUJOCO

if not HAS_MUJOCO:
    print("Notice: 'mujoco' python bindings are not available in this environment.")
    print("Please use the Matplotlib 3D Window: py -3.12 variable_wind/mujoco/matplotlib_3d_window.py")
    sys.exit(0)

import mujoco
import mujoco.viewer


def launch_native_mujoco_window(seed: int = 42):
    print("=" * 75)
    print("  LAUNCHING NATIVE MUJOCO INTERACTIVE VIEWER WINDOW")
    print("=" * 75)

    # 1. Paper 1: Obstacles & Planning
    print("[1/3] Generating 3D Obstacles & GA Adaptive B-Spline Path...")
    env_obstacles = ObstacleEnvironment(bounds=(25.0, 25.0, 12.0), seed=seed)
    start_pos = np.array([1.0, 1.0, 1.0])
    goal_pos = np.array([22.0, 22.0, 8.0])
    obstacles = env_obstacles.generate_random_obstacles(num_obstacles=12, start=start_pos, goal=goal_pos)

    astar = AStar3DPlanner(env_obstacles, grid_res=1.0)
    waypoints = astar.plan(start_pos, goal_pos)
    smoother = GAAdaptiveBSplineSmoother(env_obstacles, degree=3)
    pos_spline, vel_spline, acc_spline, arc_lengths = smoother.smooth(waypoints, num_samples=300)
    traj_gen = TrajectoryGenerator(pos_spline, vel_spline, acc_spline, arc_lengths, v_max=2.8, a_max=3.2)
    t_total = traj_gen.t_total

    # 2. Initialize MuJoCo Environment
    print("[2/3] Compiling MuJoCo MJCF Model...")
    dt = 0.005
    sim_env = MuJoCoVariableWindEnv(obstacles=obstacles, dt=dt)
    sim_env.reset(initial_pos=start_pos)

    if not sim_env.use_native_mujoco or sim_env.model is None:
        print("Notice: MuJoCo model compilation fallback active.")
        print("Launching Interactive 3D Window instead: python variable_wind/mujoco/interactive_3d_window.py")
        return

    # 3. Launch Native Passive Viewer
    print("[3/3] Opening MuJoCo Native Passive Viewer...")
    with mujoco.viewer.launch_passive(sim_env.model, sim_env.data) as viewer:
        sim_time = 0.0
        while viewer.is_running():
            step_start = time.time()

            p_ref, v_ref, a_ref, yaw_ref, dist_s = traj_gen.sample(sim_time)
            
            sim_env.step(
                pos_ref=p_ref,
                vel_ref=v_ref,
                acc_ref=a_ref,
                yaw_ref=yaw_ref,
                dist_along_path=dist_s
            )

            viewer.sync()
            sim_time += dt

            if sim_time > t_total + 2.0:
                sim_time = 0.0
                sim_env.reset(initial_pos=start_pos)

            # Sync real-time clock
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    launch_native_mujoco_window(seed=42)
