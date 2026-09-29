"""
Paper 1: 3D Obstacle Environment Generator, 3D A* Path Planner,
and GA Adaptive B-Spline Path Smoother with Time-Parameterization.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import heapq
import numpy as np
from scipy.interpolate import BSpline, splprep, splev


@dataclass
class Obstacle3D:
    """Represents a 3D obstacle (cylinder or sphere)."""
    kind: str  # 'cylinder' or 'sphere'
    center: np.ndarray  # [x, y, z]
    radius: float
    height: float = 10.0  # used for cylinder
    margin: float = 1.2   # safety clearance margin for wind disturbances

    def is_collision(self, point: np.ndarray) -> bool:
        """Checks if a point is within obstacle radius + margin."""
        eff_radius = self.radius + self.margin
        if self.kind == 'cylinder':
            dx = point[0] - self.center[0]
            dy = point[1] - self.center[1]
            dist_xy = np.sqrt(dx**2 + dy**2)
            in_height = 0.0 <= point[2] <= (self.center[2] + self.height)
            return bool(dist_xy < eff_radius and in_height)
        else:  # sphere
            dist_3d = np.linalg.norm(point - self.center)
            return bool(dist_3d < eff_radius)


class ObstacleEnvironment:
    """Manages randomized 3D obstacle generation and collision checking."""
    def __init__(
        self,
        bounds: Tuple[float, float, float] = (25.0, 25.0, 12.0),
        seed: Optional[int] = 42
    ):
        self.bounds = bounds
        self.seed = seed
        self.obstacles: List[Obstacle3D] = []
        if seed is not None:
            np.random.seed(seed)

    def generate_random_obstacles(
        self,
        num_obstacles: int = 14,
        start: np.ndarray = np.array([1.0, 1.0, 1.0]),
        goal: np.ndarray = np.array([22.0, 22.0, 8.0]),
        min_radius: float = 0.8,
        max_radius: float = 2.0
    ) -> List[Obstacle3D]:
        """Generates random non-overlapping obstacles avoiding start and goal."""
        self.obstacles = []
        start_np = np.array(start)
        goal_np = np.array(goal)

        attempts = 0
        max_attempts = 500
        while len(self.obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            kind = 'cylinder' if np.random.rand() > 0.4 else 'sphere'
            r = np.random.uniform(min_radius, max_radius)
            x = np.random.uniform(3.0, self.bounds[0] - 3.0)
            y = np.random.uniform(3.0, self.bounds[1] - 3.0)
            z = np.random.uniform(1.0, self.bounds[2] - 3.0) if kind == 'sphere' else 0.0
            center = np.array([x, y, z])

            # Safety distance from start & goal
            if np.linalg.norm(center[:2] - start_np[:2]) < (r + 5.0):
                continue
            if np.linalg.norm(center[:2] - goal_np[:2]) < (r + 3.5):
                continue

            obs = Obstacle3D(
                kind=kind,
                center=center,
                radius=r,
                height=float(np.random.uniform(5.0, self.bounds[2])),
                margin=1.2
            )
            self.obstacles.append(obs)

        return self.obstacles

    def check_point_collision(self, pt: np.ndarray) -> bool:
        """Returns True if point collides with any obstacle or exceeds bounds."""
        if pt[0] < 0 or pt[0] > self.bounds[0] or \
           pt[1] < 0 or pt[1] > self.bounds[1] or \
           pt[2] < 0.2 or pt[2] > self.bounds[2]:
            return True
        return any(obs.is_collision(pt) for obs in self.obstacles)

    def check_points_collision(self, pts: np.ndarray) -> np.ndarray:
        """Fast vectorized collision check for an (N, 3) batch of points."""
        coll = (pts[:, 0] < 0) | (pts[:, 0] > self.bounds[0]) | \
               (pts[:, 1] < 0) | (pts[:, 1] > self.bounds[1]) | \
               (pts[:, 2] < 0.2) | (pts[:, 2] > self.bounds[2])
        for obs in self.obstacles:
            eff_r = obs.radius + obs.margin
            if obs.kind == 'cylinder':
                dx = pts[:, 0] - obs.center[0]
                dy = pts[:, 1] - obs.center[1]
                dist_sq = dx*dx + dy*dy
                in_h = (pts[:, 2] >= 0.0) & (pts[:, 2] <= (obs.center[2] + obs.height))
                coll |= ((dist_sq < eff_r * eff_r) & in_h)
            else:
                diff = pts - obs.center
                dist_sq = np.sum(diff*diff, axis=1)
                coll |= (dist_sq < eff_r * eff_r)
        return coll

    def check_segment_collision(self, p1: np.ndarray, p2: np.ndarray, num_checks: int = 15) -> bool:
        """Line-segment collision check with fast vectorized batch sampling."""
        alphas = np.linspace(0.0, 1.0, num_checks)[:, np.newaxis]
        pts = (1.0 - alphas) * p1 + alphas * p2
        coll = self.check_points_collision(pts)
        return bool(np.any(coll))


class AStar3DPlanner:
    """3D A* Grid Planner for finding global collision-free waypoints."""
    def __init__(self, env: ObstacleEnvironment, grid_res: float = 1.0):
        self.env = env
        self.res = grid_res

    def _to_grid(self, pt: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.round(pt / self.res).astype(int))

    def _to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        return np.array(idx, dtype=np.float64) * self.res

    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Finds 3D collision-free path from start to goal."""
        start_idx = self._to_grid(start)
        goal_idx = self._to_grid(goal)

        # 26-connectivity offsets
        moves = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
            return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

        open_set = []
        heapq.heappush(open_set, (heuristic(start_idx, goal_idx), 0.0, start_idx))
        came_from = {}
        g_score = {start_idx: 0.0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal_idx or heuristic(current, goal_idx) <= 1.0:
                # Reconstruct path
                path_nodes = [current]
                while current in came_from:
                    current = came_from[current]
                    path_nodes.append(current)
                path_nodes.reverse()
                world_path = [self._to_world(node) for node in path_nodes]
                world_path[0] = start.copy()
                world_path[-1] = goal.copy()
                return self._prune_path(world_path)

            for dx, dy, dz in moves:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                neighbor_pt = self._to_world(neighbor)

                if self.env.check_point_collision(neighbor_pt):
                    continue

                step_cost = np.sqrt(dx**2 + dy**2 + dz**2)
                tentative_g = current_g + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        # Fallback direct path if trapped
        print("Warning: A* found no complete path. Falling back to interpolated line.")
        return [start, 0.5 * (start + goal) + np.array([0, 0, 3.0]), goal]

    def _prune_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Line-of-sight shortcutting to eliminate unnecessary intermediate grid nodes."""
        if len(path) <= 2:
            return path
        pruned = [path[0]]
        curr_idx = 0
        while curr_idx < len(path) - 1:
            next_idx = len(path) - 1
            while next_idx > curr_idx + 1:
                if not self.env.check_segment_collision(path[curr_idx], path[next_idx]):
                    break
                next_idx -= 1
            pruned.append(path[next_idx])
            curr_idx = next_idx
        return pruned


class GAAdaptiveBSplineSmoother:
    """
    Paper 1: Genetic Algorithm Adaptive B-Spline Smoother.
    Optimizes control points & knot placement to minimize curvature, path length,
    and obstacle encroachment, producing C^2 continuous trajectory.
    """
    def __init__(self, env: ObstacleEnvironment, degree: int = 3):
        self.env = env
        self.degree = degree

    def smooth(
        self,
        waypoints: List[np.ndarray],
        num_samples: int = 300,
        pop_size: int = 25,
        generations: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs adaptive GA optimization on waypoints.
        Returns:
            trajectory_positions: (N, 3)
            trajectory_velocities: (N, 3)
            trajectory_accelerations: (N, 3)
            arc_lengths: (N,)
        """
        pts = np.array(waypoints)
        if len(pts) < 4:
            # Subdivide to have enough control points for cubic B-spline
            subdivided = []
            for i in range(len(pts) - 1):
                subdivided.append(pts[i])
                subdivided.append(0.5 * (pts[i] + pts[i+1]))
            subdivided.append(pts[-1])
            pts = np.array(subdivided)

        # Baseline spline parameterization
        tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=1.0, k=min(self.degree, len(pts)-1))
        
        # GA population: perturb interior control points
        # tck[0] = knots, tck[1] = [cx, cy, cz], tck[2] = degree
        knots = tck[0]
        base_ctrl = np.array(tck[1])  # shape (3, M)
        num_ctrl = base_ctrl.shape[1]
        degree = tck[2]
        
        best_ctrl = base_ctrl.copy()
        best_fitness = -1e9

        u_eval = np.linspace(0, 1, 30)

        def evaluate_fitness(ctrl_pts: np.ndarray) -> float:
            tck_cand = (knots, [ctrl_pts[0], ctrl_pts[1], ctrl_pts[2]], degree)
            cand_pts = np.array(splev(u_eval, tck_cand)).T
            
            # 1. Path length penalty
            diffs = np.diff(cand_pts, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))

            # 2. Curvature penalty
            der1 = np.array(splev(u_eval, tck_cand, der=1)).T
            der2 = np.array(splev(u_eval, tck_cand, der=2)).T
            cross_p = np.cross(der1, der2)
            denom = (np.linalg.norm(der1, axis=1) ** 3) + 1e-6
            curvatures = np.linalg.norm(cross_p, axis=1) / denom
            curv_cost = np.sum(curvatures ** 2)

            # 3. Vectorized Obstacle collision penalty
            coll_mask = self.env.check_points_collision(cand_pts)
            collision_cost = float(np.sum(coll_mask) * 500.0)

            fitness = -(0.05 * length + 0.1 * curv_cost + collision_cost)
            return float(fitness)

        # GA evolution loop
        for gen in range(generations):
            candidates = [base_ctrl.copy()]
            for _ in range(pop_size - 1):
                mut = best_ctrl.copy()
                if num_ctrl > 2:
                    noise = np.random.randn(3, num_ctrl - 2) * 0.25
                    mut[:, 1:-1] += noise
                candidates.append(mut)

            for cand in candidates:
                fit = evaluate_fitness(cand)
                if fit > best_fitness:
                    best_fitness = fit
                    best_ctrl = cand.copy()

        # Final evaluation of best spline
        best_tck = (knots, [best_ctrl[0], best_ctrl[1], best_ctrl[2]], degree)
        u_fine = np.linspace(0, 1, num_samples)
        pos = np.array(splev(u_fine, best_tck, der=0)).T
        vel = np.array(splev(u_fine, best_tck, der=1)).T
        acc = np.array(splev(u_fine, best_tck, der=2)).T

        # Calculate arc lengths along trajectory
        dists = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        arc_lengths = np.zeros(num_samples)
        arc_lengths[1:] = np.cumsum(dists)

        return pos, vel, acc, arc_lengths


class TrajectoryGenerator:
    """
    Time-parameterizes the smoothed 3D B-spline curve with max velocity and acceleration constraints,
    producing an analytical reference trajectory p_ref(t), v_ref(t), a_ref(t), yaw_ref(t).
    """
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        arc_lengths: np.ndarray,
        v_max: float = 2.5,
        a_max: float = 3.0
    ):
        self.positions = positions
        self.raw_vel = velocities
        self.raw_acc = accelerations
        self.arc_lengths = arc_lengths
        self.total_length = arc_lengths[-1]
        self.v_max = v_max
        self.a_max = a_max

        # Time allocation using trapezoidal profile
        t_acc = v_max / a_max
        d_acc = 0.5 * a_max * (t_acc ** 2)
        if 2 * d_acc > self.total_length:
            # Triangular profile
            d_acc = self.total_length * 0.5
            self.v_peak = np.sqrt(a_max * self.total_length)
            self.t_total = 2.0 * np.sqrt(self.total_length / a_max)
        else:
            self.v_peak = v_max
            d_flat = self.total_length - 2 * d_acc
            t_flat = d_flat / v_max
            self.t_total = 2 * t_acc + t_flat

        self.num_pts = len(positions)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Samples reference state at time t:
        Returns:
            p_ref: [x, y, z]
            v_ref: [vx, vy, vz]
            a_ref: [ax, ay, az]
            yaw_ref: scalar (radians)
            dist_along_path: scalar (m)
        """
        t_clamped = np.clip(t, 0.0, self.t_total)
        s_norm = t_clamped / max(self.t_total, 1e-6)
        
        idx_f = s_norm * (self.num_pts - 1)
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, self.num_pts - 1)
        w1 = idx_f - i0
        w0 = 1.0 - w1

        pos = w0 * self.positions[i0] + w1 * self.positions[i1]
        dist_s = w0 * self.arc_lengths[i0] + w1 * self.arc_lengths[i1]

        # Velocity scaling
        v_dir = self.raw_vel[i0]
        v_dir_norm = np.linalg.norm(v_dir)
        if v_dir_norm > 1e-6:
            v_dir = v_dir / v_dir_norm
            speed = self.v_peak * np.sin(np.pi * np.clip(s_norm, 0, 1))
            vel = v_dir * max(speed, 0.1)
        else:
            vel = np.zeros(3)

        acc = self.raw_acc[i0] * 0.2

        # Desired yaw: aligned with flight heading
        if np.linalg.norm(vel[:2]) > 0.1:
            yaw = float(np.arctan2(vel[1], vel[0]))
        else:
            yaw = 0.0

        return pos, vel, acc, yaw, float(dist_s)
