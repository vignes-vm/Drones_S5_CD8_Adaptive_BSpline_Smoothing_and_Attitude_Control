"""
Shared obstacle environment — Part 4 "dense obstacle" scenario.

Reimplements the MATLAB Part 4 environment so that BOTH the A* planner
(astar_3d.py) and the mission generator (generate_mission.py) use identical
obstacle data. Obstacles are modelled as vertical cylinders (towers) rising
from the ground, which is how the corridor/scattered scenario is drawn in MATLAB.

  start = [0, 0, 20]         goal = [2000, 1800, 200]     (metres, local ENU-ish)
  11 obstacles = 7 corridor (weaving the direct path) + 4 scattered

>>> NOTE: the obstacle positions below are generated deterministically to match
>>> the Part 4 *layout* (7 corridor + 4 scattered). Replace CORRIDOR_SPEC /
>>> SCATTERED with your exact MATLAB Part 4 obstacle coordinates when available.
"""
import numpy as np

START = np.array([0.0, 0.0, 20.0])
GOAL = np.array([2000.0, 1800.0, 200.0])
# world bounds (x, y, z) in metres
SPACE = np.array([2100.0, 2000.0, 320.0])

# safety buffer added around every obstacle radius (metres)
SAFETY_BUFFER = 40.0


def _corridor_obstacles():
    """7 towers placed along the start->goal line, alternating lateral offset
    so the straight path is blocked and must weave left/right."""
    d = GOAL[:2] - START[:2]
    length = np.linalg.norm(d)
    fwd = d / length
    perp = np.array([-fwd[1], fwd[0]])  # left-hand perpendicular in XY

    obstacles = []
    # (fraction along path, lateral offset m, radius m, height m)
    specs = [
        (0.14,  160, 45, 240),
        (0.27, -180, 40, 260),
        (0.40,  150, 50, 300),
        (0.53, -160, 42, 280),
        (0.66,  170, 46, 320),
        (0.79, -150, 40, 250),
        (0.90,  120, 48, 300),
    ]
    for frac, off, r, h in specs:
        centre = START[:2] + fwd * (frac * length) + perp * off
        obstacles.append({"type": "cylinder",
                          "x": float(centre[0]), "y": float(centre[1]),
                          "radius": float(r), "height": float(h)})
    return obstacles


# 4 scattered towers away from the corridor
SCATTERED = [
    {"type": "cylinder", "x": 300.0,  "y": 1300.0, "radius": 60.0, "height": 220.0},
    {"type": "cylinder", "x": 1600.0, "y": 500.0,  "radius": 55.0, "height": 260.0},
    {"type": "cylinder", "x": 1200.0, "y": 1600.0, "radius": 50.0, "height": 300.0},
    {"type": "cylinder", "x": 800.0,  "y": 300.0,  "radius": 58.0, "height": 200.0},
]


def get_obstacles():
    """Return the full list of 11 obstacle dicts (7 corridor + 4 scattered)."""
    return _corridor_obstacles() + [dict(o) for o in SCATTERED]


def in_collision(point, obstacles=None, buffer=SAFETY_BUFFER):
    """True if `point` (x,y,z) lies inside any tower (+safety buffer).
    Towers span z in [0, height]."""
    if obstacles is None:
        obstacles = get_obstacles()
    x, y, z = point
    for o in obstacles:
        if z < 0 or z > o["height"] + buffer:
            continue
        if np.hypot(x - o["x"], y - o["y"]) <= o["radius"] + buffer:
            return True
    return False


def path_in_collision(points, obstacles=None, buffer=SAFETY_BUFFER):
    """True if ANY sampled point along the path collides."""
    if obstacles is None:
        obstacles = get_obstacles()
    return any(in_collision(p, obstacles, buffer) for p in points)


if __name__ == "__main__":
    obs = get_obstacles()
    print(f"start={START}  goal={GOAL}  obstacles={len(obs)} "
          f"(7 corridor + 4 scattered)")
    for i, o in enumerate(obs):
        print(f"  [{i:2d}] ({o['x']:7.1f}, {o['y']:7.1f}) r={o['radius']:.0f} h={o['height']:.0f}")
    # self-check: start & goal must be collision-free
    assert not in_collision(START), "start in collision!"
    assert not in_collision(GOAL), "goal in collision!"
    print("start & goal are collision-free: OK")
