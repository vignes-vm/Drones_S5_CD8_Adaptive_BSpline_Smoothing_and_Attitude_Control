import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D

env = Environment3D()
env.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=42)

planner = AStarPlanner3D(env, grid_resolution=50)
waypoints = planner.plan(env.start, env.goal)

path_length = planner.compute_path_length(waypoints)

print(f"A* grid: {planner.nx} x {planner.ny} x {planner.nz} "
      f"cells (resolution={planner.resolution}m)")
print(f"Number of waypoints: {len(waypoints)}")
print(f"Path length (Baseline Polyline): {path_length:.1f} m")
straight_dist = np.linalg.norm(env.goal - env.start)
print(f"Straight-line distance:          {straight_dist:.1f} m")
print(f"Straightness ratio (SR):         {path_length / straight_dist:.3f}")

fig = plt.figure(figsize=(12, 10))
ax  = fig.add_subplot(111, projection='3d')
env.visualize(ax, paths=[np.array(waypoints)])
ax.set_title(f'3D A* Baseline Polyline ({len(waypoints)} waypoints, '
             f'L={path_length:.0f}m)')
plt.tight_layout()
plt.savefig('results/astar_path.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/astar_path.png")
