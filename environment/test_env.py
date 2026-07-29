import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.obstacles import Environment3D

env = Environment3D()
env.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=42)

print(f"Environment: {env.space[0]}x{env.space[1]}x{env.space[2]} m")
print(f"Start: {env.start}")
print(f"Goal:  {env.goal}")
print(f"Seed:  {env.seed}")
print(f"\nPlaced {len(env.obstacles)} obstacles:\n")

for i, obs in enumerate(env.obstacles):
    if obs.shape == 'cylinder':
        radius, height = obs.dimensions
        print(f"  [{i}] CYLINDER  pos={obs.position.round(1)}  "
              f"radius={radius:.1f}m  height={height:.1f}m  buffer={obs.buffer}m")
    else:
        length, width, height = obs.dimensions
        print(f"  [{i}] CUBOID    pos={obs.position.round(1)}  "
              f"dims={length:.1f}x{width:.1f}x{height:.1f}m  buffer={obs.buffer}m")

fig = plt.figure(figsize=(12, 10))
ax  = fig.add_subplot(111, projection='3d')
env.visualize(ax)
ax.set_title('3D Obstacle Environment (Paper 1)')
plt.tight_layout()
plt.savefig('results/environment_test.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/environment_test.png")
