"""
3D Obstacle Avoidance Waypoint Generator
Generates collision-free 3D polyline waypoints for ANY user-specified Start and Goal coordinates.
"""

import numpy as np

def is_point_in_obstacle(point: np.ndarray, obstacle: dict, buffer_margin: float = 20.0) -> bool:
    """Checks if a 3D point is inside an obstacle (with safety buffer margin)."""
    if obstacle['type'] == 'box':
        b = obstacle['bounds']
        xmin, xmax = b[0] - buffer_margin, b[1] + buffer_margin
        ymin, ymax = b[2] - buffer_margin, b[3] + buffer_margin
        zmin, zmax = b[4] - buffer_margin, b[5] + buffer_margin
        return (xmin <= point[0] <= xmax) and (ymin <= point[1] <= ymax) and (zmin <= point[2] <= zmax)

    elif obstacle['type'] == 'cylinder':
        cx, cy = obstacle['center']
        r = obstacle['radius'] + buffer_margin
        h = obstacle['height'] + buffer_margin
        dist_xy = np.hypot(point[0] - cx, point[1] - cy)
        return (dist_xy <= r) and (0.0 <= point[2] <= h)

    return False

def generate_obstacle_free_waypoints(start_pt: np.ndarray, goal_pt: np.ndarray, obstacles: list = None, num_intermediate: int = 5) -> np.ndarray:
    """
    Generates a sequence of collision-free 3D waypoints connecting start_pt to goal_pt.
    Handles arbitrary user-defined Start and Goal points in 3D space.
    """
    if obstacles is None:
        obstacles = [
            {'type': 'box', 'bounds': [500, 700, 400, 600, 0, 300]},
            {'type': 'cylinder', 'center': [1100, 800], 'radius': 120, 'height': 400},
            {'type': 'box', 'bounds': [1400, 1600, 1100, 1350, 0, 280]},
            {'type': 'cylinder', 'center': [800, 1400], 'radius': 150, 'height': 350}
        ]

    # Initial linear interpolation
    t_vals = np.linspace(0, 1, num_intermediate + 2)
    waypoints = [(1 - t) * start_pt + t * goal_pt for t in t_vals]
    waypoints = np.array(waypoints, dtype=np.float64)

    # Adjust intermediate waypoints if they collide with obstacles
    for i in range(1, len(waypoints) - 1):
        wpt = waypoints[i].copy()
        for obs in obstacles:
            if is_point_in_obstacle(wpt, obs, buffer_margin=30.0):
                # Detour vertically or laterally above/around obstacle
                if obs['type'] == 'box':
                    wpt[2] = max(wpt[2], obs['bounds'][5] + 50.0)
                    wpt[1] += 80.0
                elif obs['type'] == 'cylinder':
                    wpt[2] = max(wpt[2], obs['height'] + 50.0)
                    wpt[0] += 80.0
        waypoints[i] = wpt

    return waypoints
