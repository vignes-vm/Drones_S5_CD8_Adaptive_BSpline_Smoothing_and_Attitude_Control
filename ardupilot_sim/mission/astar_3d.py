"""
Grid-based 3D A* — same approach as the MATLAB version.

Plans start->goal on a uniform 3D grid, 26-connected, Euclidean step cost +
heuristic. A one-sided cruise-altitude penalty keeps the path at a sensible
cruise height instead of hugging the ground (pure-distance A* on ground-based
towers otherwise crawls at z=0). Collision uses the shared obstacle_env.
"""
import heapq
import itertools
import numpy as np

from obstacle_env import START, GOAL, SPACE, get_obstacles, in_collision


class AStar3D:
    def __init__(self, resolution=50.0, cruise_altitude=70.0, altitude_weight=1.0):
        self.res = resolution
        self.cruise_altitude = cruise_altitude
        self.altitude_weight = altitude_weight
        self.obstacles = get_obstacles()
        self.nx = int(np.ceil(SPACE[0] / resolution)) + 1
        self.ny = int(np.ceil(SPACE[1] / resolution)) + 1
        self.nz = int(np.ceil(SPACE[2] / resolution)) + 1
        self.offsets = [o for o in itertools.product((-1, 0, 1), repeat=3)
                        if o != (0, 0, 0)]

    def _to_grid(self, p):
        idx = np.round(np.array(p) / self.res).astype(int)
        return (int(np.clip(idx[0], 0, self.nx - 1)),
                int(np.clip(idx[1], 0, self.ny - 1)),
                int(np.clip(idx[2], 0, self.nz - 1)))

    def _to_world(self, c):
        return np.array(c, dtype=float) * self.res

    def _free(self, c):
        return not in_collision(self._to_world(c), self.obstacles)

    def plan(self, start=None, goal=None):
        start = START if start is None else np.asarray(start, float)
        goal = GOAL if goal is None else np.asarray(goal, float)
        s, g = self._to_grid(start), self._to_grid(goal)

        def h(c):
            return np.linalg.norm(self._to_world(c) - self._to_world(g))

        counter = itertools.count()
        open_heap = [(0.0, next(counter), s)]
        came, gscore, closed = {}, {s: 0.0}, set()
        found = False
        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == g:
                found = True
                break
            cur_w = self._to_world(cur)
            for dx, dy, dz in self.offsets:
                nb = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                if not (0 <= nb[0] < self.nx and 0 <= nb[1] < self.ny
                        and 0 <= nb[2] < self.nz):
                    continue
                if nb in closed or not self._free(nb):
                    continue
                nb_w = self._to_world(nb)
                step = np.linalg.norm(nb_w - cur_w)
                # two-sided: hold cruise altitude (penalise BOTH above and below),
                # so the path weaves around towers at cruise instead of overflying;
                # the mandatory climb to the goal altitude still happens, but late.
                step += self.altitude_weight * abs(nb_w[2] - self.cruise_altitude)
                tg = gscore[cur] + step
                if tg < gscore.get(nb, np.inf):
                    gscore[nb] = tg
                    came[nb] = cur
                    heapq.heappush(open_heap, (tg + h(nb), next(counter), nb))

        if not found:
            print("A* WARNING: no path found — returning straight line.")
            return np.array([start, goal])

        path = [g]
        node = g
        while node != s:
            node = came[node]
            path.append(node)
        path.reverse()
        world = [self._to_world(c) for c in path]
        world[0], world[-1] = start, goal          # snap exact endpoints
        return np.array(world)

    @staticmethod
    def path_length(path):
        path = np.asarray(path)
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


if __name__ == "__main__":
    planner = AStar3D()
    p = planner.plan()
    print(f"A* waypoints: {len(p)}  length: {planner.path_length(p):.1f} m")
    print("z range:", p[:, 2].min(), "->", p[:, 2].max())
