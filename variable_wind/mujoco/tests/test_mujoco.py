r"""
Unit Test Suite for MuJoCo Variable Wind Simulation.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.obstacle_planner import ObstacleEnvironment, AStar3DPlanner, GAAdaptiveBSplineSmoother, TrajectoryGenerator
from src.mujoco_env import MuJoCoVariableWindEnv


class TestMuJoCoVariableWind(unittest.TestCase):
    def test_mujoco_pipeline(self):
        env_obs = ObstacleEnvironment(seed=42)
        obstacles = env_obs.generate_random_obstacles(num_obstacles=6)
        
        astar = AStar3DPlanner(env_obs, grid_res=1.5)
        waypoints = astar.plan(np.array([1.0, 1.0, 1.0]), np.array([20.0, 20.0, 6.0]))
        
        smoother = GAAdaptiveBSplineSmoother(env_obs, degree=3)
        pos, vel, acc, arc_lengths = smoother.smooth(waypoints, num_samples=50, pop_size=10, generations=5)
        
        traj_gen = TrajectoryGenerator(pos, vel, acc, arc_lengths, v_max=2.5, a_max=3.0)
        
        sim_env = MuJoCoVariableWindEnv(obstacles=obstacles, dt=0.01)
        sim_env.reset(initial_pos=np.array([1.0, 1.0, 1.0]))
        
        p_ref, v_ref, a_ref, yaw_ref, dist_s = traj_gen.sample(0.5)
        telem = sim_env.step(pos_ref=p_ref, vel_ref=v_ref, acc_ref=a_ref, yaw_ref=yaw_ref, dist_along_path=dist_s)
        
        self.assertIn("pos", telem)
        self.assertIn("v_wind", telem)
        self.assertIn("pos_error", telem)
        self.assertLess(telem["pos_error"], 5.0)


if __name__ == "__main__":
    unittest.main()
