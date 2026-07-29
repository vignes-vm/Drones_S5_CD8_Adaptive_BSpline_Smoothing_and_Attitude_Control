"""
3D obstacle environment matching Paper 1:
"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"
"""
import numpy as np


class Obstacle3D:

    def __init__(self, shape, position, dimensions, buffer=5.0):
        self.shape      = shape
        self.position   = np.array(position, dtype=float)
        self.dimensions = np.array(dimensions, dtype=float)
        self.buffer     = buffer

    def _half_extents(self, with_buffer=True):
        b = self.buffer if with_buffer else 0.0
        if self.shape == 'cylinder':
            radius, height = self.dimensions
            return np.array([radius + b, radius + b, height / 2 + b])
        elif self.shape == 'cuboid':
            length, width, height = self.dimensions
            return np.array([length / 2 + b, width / 2 + b, height / 2 + b])
        else:
            raise ValueError(f"Unknown shape: {self.shape}")

    def check_collision(self, point):
        point = np.array(point, dtype=float)
        delta = point - self.position

        if self.shape == 'cylinder':
            radius, height = self.dimensions
            planar_dist = np.hypot(delta[0], delta[1])
            return (planar_dist <= radius + self.buffer and
                    abs(delta[2]) <= height / 2 + self.buffer)
        elif self.shape == 'cuboid':
            half = self._half_extents(with_buffer=True)
            return bool(np.all(np.abs(delta) <= half))
        else:
            raise ValueError(f"Unknown shape: {self.shape}")

    def get_corners(self):
        half = self._half_extents(with_buffer=False)
        signs = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        ])
        return self.position + signs * half


class Environment3D:

    def __init__(self):
        self.space   = (2000.0, 2000.0, 500.0)   # x, y, z extents
        self.start   = np.array([0.0, 0.0, 20.0])
        self.goal    = np.array([2000.0, 1800.0, 200.0])
        self.obstacles = []
        self.seed    = None

    def _bounding_radius(self, shape, dimensions, buffer):
        if shape == 'cylinder':
            radius, height = dimensions
            return max(radius, height / 2) + buffer
        else:
            length, width, height = dimensions
            return 0.5 * np.linalg.norm([length, width, height]) + buffer

    def _valid_placement(self, position, bounding_radius):
        if (np.linalg.norm(position - self.start) < 100.0 + bounding_radius or
                np.linalg.norm(position - self.goal) < 100.0 + bounding_radius):
            return False

        for obs in self.obstacles:
            other_radius = self._bounding_radius(
                obs.shape, obs.dimensions, obs.buffer
            )
            if np.linalg.norm(position - obs.position) < (
                    bounding_radius + other_radius):
                return False
        return True

    def add_random_obstacles(self, n_cylinders=5, n_cuboids=5, seed=None,
                              buffer=5.0, max_attempts=500):
        self.seed = seed
        rng = np.random.default_rng(seed)

        x_max, y_max, z_max = self.space

        # Cylinders
        placed = 0
        attempts = 0
        while placed < n_cylinders and attempts < max_attempts:
            attempts += 1
            radius = rng.uniform(20.0, 60.0)
            height = rng.uniform(100.0, 400.0)
            z_center = min(height / 2, z_max - height / 2) \
                if height <= z_max else z_max / 2

            x = rng.uniform(radius + buffer, x_max - radius - buffer)
            y = rng.uniform(radius + buffer, y_max - radius - buffer)
            position = np.array([x, y, z_center])

            bounding_r = self._bounding_radius('cylinder', [radius, height], buffer)
            if self._valid_placement(position, bounding_r):
                self.obstacles.append(
                    Obstacle3D('cylinder', position, [radius, height], buffer)
                )
                placed += 1

        # Cuboids
        placed = 0
        attempts = 0
        while placed < n_cuboids and attempts < max_attempts:
            attempts += 1
            length = rng.uniform(50.0, 200.0)
            width  = rng.uniform(50.0, 200.0)
            height = rng.uniform(100.0, 300.0)
            z_center = min(height / 2, z_max - height / 2) \
                if height <= z_max else z_max / 2

            x = rng.uniform(length / 2 + buffer, x_max - length / 2 - buffer)
            y = rng.uniform(width / 2 + buffer, y_max - width / 2 - buffer)
            position = np.array([x, y, z_center])

            bounding_r = self._bounding_radius(
                'cuboid', [length, width, height], buffer
            )
            if self._valid_placement(position, bounding_r):
                self.obstacles.append(
                    Obstacle3D('cuboid', position, [length, width, height], buffer)
                )
                placed += 1

    def check_path_collision(self, points):
        points = np.asarray(points, dtype=float)
        for point in points:
            for obs in self.obstacles:
                if obs.check_collision(point):
                    return True
        return False

    def visualize(self, ax, waypoints=None, paths=None):
        for obs in self.obstacles:
            if obs.shape == 'cylinder':
                radius, height = obs.dimensions
                theta = np.linspace(0, 2 * np.pi, 24)
                z = np.linspace(
                    obs.position[2] - height / 2,
                    obs.position[2] + height / 2, 2
                )
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = obs.position[0] + radius * np.cos(theta_grid)
                y_grid = obs.position[1] + radius * np.sin(theta_grid)
                ax.plot_wireframe(x_grid, y_grid, z_grid,
                                   color='green', linewidth=0.8, alpha=0.7)
            elif obs.shape == 'cuboid':
                corners = obs.get_corners()
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]
                for i, j in edges:
                    p1, p2 = corners[i], corners[j]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            color='blue', linewidth=0.8, alpha=0.7)

        ax.scatter(*self.start, c='green', s=100, marker='o', label='Start', zorder=6)
        ax.scatter(*self.goal,  c='red',   s=100, marker='^', label='Goal',  zorder=6)

        if waypoints is not None:
            waypoints = np.asarray(waypoints)
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                       c='black', s=40, label='Waypoints', zorder=5)

        if paths is not None:
            for path in paths:
                path = np.asarray(path)
                ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=2)

        ax.set_xlim(0, self.space[0])
        ax.set_ylim(0, self.space[1])
        ax.set_zlim(0, self.space[2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(fontsize=9)
