"""
3D A* path planner — generates the jagged "Baseline Polyline" from
Paper 1 Table 2, prior to any B-Spline smoothing.
"""
import heapq
import itertools
import numpy as np


class AStarPlanner3D:

    def __init__(self, environment, grid_resolution=50,
                 cruise_altitude=100.0, altitude_weight=2.0):
        self.env        = environment
        self.resolution = grid_resolution
        # Altitude-holding: penalise flying below cruise_altitude so the path
        # climbs to cruise and holds it, deviating (horizontally) only to route
        # around obstacles. Without this, pure-distance A* crawls along ground
        # layer 0. Tune cruise_altitude (m) and altitude_weight (penalty per
        # metre below cruise, relative to the ~grid_resolution step cost).
        self.cruise_altitude = cruise_altitude
        self.altitude_weight = altitude_weight

        x_max, y_max, z_max = environment.space
        self.nx = max(1, int(np.ceil(x_max / grid_resolution)))
        self.ny = max(1, int(np.ceil(y_max / grid_resolution)))
        self.nz = max(1, int(np.ceil(z_max / grid_resolution)))

        # 26-connected neighbor offsets (all combos of -1,0,1 except origin)
        self._neighbor_offsets = [
            offset for offset in itertools.product((-1, 0, 1), repeat=3)
            if offset != (0, 0, 0)
        ]

    def _world_to_grid(self, point):
        idx = np.round(np.array(point) / self.resolution).astype(int)
        idx[0] = np.clip(idx[0], 0, self.nx - 1)
        idx[1] = np.clip(idx[1], 0, self.ny - 1)
        idx[2] = np.clip(idx[2], 0, self.nz - 1)
        return tuple(idx)

    def _grid_to_world(self, cell):
        return np.array(cell, dtype=float) * self.resolution

    def _in_bounds(self, cell):
        return (0 <= cell[0] < self.nx and
                0 <= cell[1] < self.ny and
                0 <= cell[2] < self.nz)

    def _is_free(self, cell):
        world_pt = self._grid_to_world(cell)
        return not self.env.check_path_collision([world_pt])

    def plan(self, start, goal):
        start = np.array(start, dtype=float)
        goal  = np.array(goal, dtype=float)

        start_cell = self._world_to_grid(start)
        goal_cell  = self._world_to_grid(goal)

        if not self._is_free(start_cell) or not self._is_free(goal_cell):
            print("WARNING: A* start or goal cell collides with an "
                  "obstacle — returning straight line.")
            return [start, goal]

        counter    = itertools.count()
        open_heap  = [(0.0, next(counter), start_cell)]
        came_from  = {}
        g_score    = {start_cell: 0.0}
        closed_set = set()

        def heuristic(cell):
            return np.linalg.norm(
                self._grid_to_world(cell) - self._grid_to_world(goal_cell)
            )

        found = False
        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal_cell:
                found = True
                break

            current_world = self._grid_to_world(current)
            for dx, dy, dz in self._neighbor_offsets:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                if not self._in_bounds(neighbor) or neighbor in closed_set:
                    continue
                if not self._is_free(neighbor):
                    continue

                neighbor_world = self._grid_to_world(neighbor)
                step_cost = np.linalg.norm(neighbor_world - current_world)
                # cruise-altitude penalty (one-sided: only below cruise)
                step_cost += self.altitude_weight * max(
                    0.0, self.cruise_altitude - neighbor_world[2]
                )
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, np.inf):
                    g_score[neighbor]   = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_heap, (f_score, next(counter), neighbor))

        if not found:
            print("WARNING: A* found no path to goal — returning "
                  "straight line.")
            return [start, goal]

        # Reconstruct path
        cell_path = [goal_cell]
        node = goal_cell
        while node != start_cell:
            node = came_from[node]
            cell_path.append(node)
        cell_path.reverse()

        waypoints = [self._grid_to_world(c) for c in cell_path]
        waypoints[0]  = start
        waypoints[-1] = goal
        return waypoints

    def smooth_waypoints(self, waypoints, min_distance=100):
        waypoints = [np.array(p, dtype=float) for p in waypoints]
        if len(waypoints) <= 2:
            return waypoints

        reduced = [waypoints[0]]
        for pt in waypoints[1:-1]:
            if np.linalg.norm(pt - reduced[-1]) >= min_distance:
                reduced.append(pt)

        goal = waypoints[-1]
        if np.linalg.norm(goal - reduced[-1]) < 1e-9:
            pass
        else:
            reduced.append(goal)

        return reduced

    def compute_path_length(self, waypoints):
        waypoints = np.array(waypoints, dtype=float)
        if len(waypoints) < 2:
            return 0.0
        diffs = np.diff(waypoints, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
