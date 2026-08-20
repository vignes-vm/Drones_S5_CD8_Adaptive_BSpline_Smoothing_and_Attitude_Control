"""
B-Spline Path Planning & Adaptive Optimization Engine
Implements Paper 2:
"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning" (Nikolaiev & Novotarskyi, 2025)

Includes:
1. Cubic B-spline evaluation & derivative computations (velocity C', acceleration C'')
2. Curvature evaluation kappa(u) = ||C'(u) x C''(u)|| / ||C'(u)||^3
3. Four Smoothing Approaches:
   - Baseline Polyline (No smoothing)
   - Uniform B-Spline
   - Classical B-Spline (Chord-length parameterization)
   - Enhanced Adaptive B-Spline (GA knot optimization + curvature-aware control point refinement)
4. Key performance metrics calculation (Path Length, Max Curvature, Straightness Ratio, Computation Time)
"""

import time
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

def compute_chord_length_parameterization(waypoints: np.ndarray) -> np.ndarray:
    """Computes normalized chord-length parameters u_j in [0, 1] for given waypoints."""
    diffs = np.diff(waypoints, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
    total_len = cum_dists[-1]
    if total_len < 1e-12:
        return np.linspace(0, 1, len(waypoints))
    return cum_dists / total_len

def create_valid_clamped_knots(u_params: np.ndarray, degree: int = 3) -> np.ndarray:
    """Creates a strictly valid clamped knot vector for scipy BSpline."""
    n = len(u_params)
    u_clean = u_params.copy()
    for i in range(1, len(u_clean)):
        if u_clean[i] <= u_clean[i-1] + 1e-5:
            u_clean[i] = u_clean[i-1] + 1e-5
    u_clean = (u_clean - u_clean[0]) / (u_clean[-1] - u_clean[0])

    knots = np.zeros(n + degree + 1)
    knots[:degree+1] = 0.0
    knots[-(degree+1):] = 1.0

    for j in range(1, n - degree):
        knots[j + degree] = np.mean(u_clean[j : j + degree])

    for j in range(degree + 1, len(knots) - (degree + 1)):
        if knots[j] <= knots[j-1] + 1e-4:
            knots[j] = knots[j-1] + 1e-4

    # Enforce exact clamping bounds
    knots[:degree+1] = 0.0
    knots[-(degree+1):] = 1.0

    return knots

class BSplineEvaluator:
    def __init__(self, waypoints: np.ndarray, knots: np.ndarray = None, params: np.ndarray = None, degree: int = 3):
        self.control_points = waypoints.copy()
        self.degree = degree
        n = len(waypoints)

        if params is None:
            params = np.linspace(0, 1, n)

        # Use make_interp_spline to construct guaranteed C2 differentiable spline
        self.spline = make_interp_spline(params, waypoints, k=degree)
        self.spline_d1 = self.spline.derivative(1)
        self.spline_d2 = self.spline.derivative(2)

    def eval_pos(self, u: float) -> np.ndarray:
        return self.spline(np.clip(u, 0.0, 1.0))

    def eval_vel(self, u: float) -> np.ndarray:
        return self.spline_d1(np.clip(u, 0.0, 1.0))

    def eval_acc(self, u: float) -> np.ndarray:
        return self.spline_d2(np.clip(u, 0.0, 1.0))

    def eval_curvature(self, u: float) -> float:
        """Computes kappa(u) = ||C'(u) x C''(u)|| / ||C'(u)||^3 (Eq 2 in Paper 2)."""
        v = self.eval_vel(u)
        a = self.eval_acc(u)

        speed = np.linalg.norm(v)
        if speed < 1e-6:
            return 0.0

        cross_prod = np.cross(v, a)
        curvature = np.linalg.norm(cross_prod) / (speed ** 3)
        return float(curvature)

    def sample_path(self, num_samples: int = 500) -> tuple:
        """Samples positions, velocities, accelerations, and curvatures along u in [0, 1]."""
        u_vals = np.linspace(0, 1, num_samples)
        positions = self.spline(u_vals)
        velocities = self.spline_d1(u_vals)
        accelerations = self.spline_d2(u_vals)

        curvatures = []
        for u in u_vals:
            curvatures.append(self.eval_curvature(u))
        curvatures = np.array(curvatures)

        return positions, velocities, accelerations, curvatures

def compute_path_metrics(positions: np.ndarray, curvatures: np.ndarray, comp_time: float, start_pt: np.ndarray, goal_pt: np.ndarray) -> dict:
    """
    Computes key performance metrics matching Table 2 in Paper 2:
    - Path Length L (m)
    - Max Curvature kappa_max (m^-1)
    - Computation Time (s)
    - Straightness Ratio SR = L / ||goal - start||
    """
    diffs = np.diff(positions, axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(np.sum(step_lengths))

    max_curvature = float(np.max(curvatures)) if len(curvatures) > 0 else 0.0

    direct_dist = float(np.linalg.norm(goal_pt - start_pt))
    straightness_ratio = path_length / direct_dist if direct_dist > 1e-12 else 1.0

    return {
        'path_length': path_length,
        'max_curvature': max_curvature,
        'comp_time': comp_time,
        'straightness_ratio': straightness_ratio
    }

# -----------------------------------------------------------------------------
# 4 PATH SMOOTHING METHODS
# -----------------------------------------------------------------------------

def fit_baseline_polyline(raw_waypoints: np.ndarray, num_samples: int = 500) -> tuple:
    """Method 1: Baseline Polyline (No smoothing)."""
    t0 = time.time()
    n_pts = len(raw_waypoints)
    t_wpt = np.linspace(0, 1, n_pts)
    u_eval = np.linspace(0, 1, num_samples)

    # Piecewise linear interpolation
    pos_x = np.interp(u_eval, t_wpt, raw_waypoints[:, 0])
    pos_y = np.interp(u_eval, t_wpt, raw_waypoints[:, 1])
    pos_z = np.interp(u_eval, t_wpt, raw_waypoints[:, 2])
    positions = np.vstack([pos_x, pos_y, pos_z]).T

    # Numerical derivatives
    dt = 1.0 / num_samples
    vels = np.gradient(positions, axis=0) / dt
    accs = np.gradient(vels, axis=0) / dt

    curvatures = []
    for i in range(len(positions)):
        v = vels[i]
        a = accs[i]
        spd = np.linalg.norm(v)
        if spd < 1e-6:
            curvatures.append(0.0)
        else:
            curvatures.append(np.linalg.norm(np.cross(v, a)) / (spd**3))
    curvatures = np.array(curvatures)

    comp_time = time.time() - t0
    metrics = compute_path_metrics(positions, curvatures, comp_time, raw_waypoints[0], raw_waypoints[-1])

    return positions, curvatures, metrics

def fit_uniform_bspline(raw_waypoints: np.ndarray, num_samples: int = 500) -> tuple:
    """Method 2: Uniform B-Spline."""
    t0 = time.time()
    n = len(raw_waypoints)
    params = np.linspace(0, 1, n)

    evaluator = BSplineEvaluator(raw_waypoints, params=params)
    positions, vels, accs, curvatures = evaluator.sample_path(num_samples)

    comp_time = time.time() - t0
    metrics = compute_path_metrics(positions, curvatures, comp_time, raw_waypoints[0], raw_waypoints[-1])

    return positions, curvatures, metrics, evaluator

def fit_classical_bspline(raw_waypoints: np.ndarray, num_samples: int = 500) -> tuple:
    """Method 3: Classical B-Spline with Chord-length parameterization."""
    t0 = time.time()
    params = compute_chord_length_parameterization(raw_waypoints)

    evaluator = BSplineEvaluator(raw_waypoints, params=params)
    positions, vels, accs, curvatures = evaluator.sample_path(num_samples)

    comp_time = time.time() - t0
    metrics = compute_path_metrics(positions, curvatures, comp_time, raw_waypoints[0], raw_waypoints[-1])

    return positions, curvatures, metrics, evaluator

def fit_enhanced_adaptive_bspline(raw_waypoints: np.ndarray, num_samples: int = 500) -> tuple:
    """
    Method 4: Proposed Enhanced Adaptive B-Spline
    - Genetic Algorithm (GA) for knot vector / parameter optimization (Eq 4 in Paper 2)
    - Curvature-aware control point refinement
    """
    t0 = time.time()
    n = len(raw_waypoints)

    # Initial baseline parameter vector (Chord-length anchor)
    base_params = compute_chord_length_parameterization(raw_waypoints)

    # 1. Genetic Algorithm for Parameter & Knot Vector Optimization
    pop_size = 30
    generations = 35

    population = []
    for _ in range(pop_size):
        if len(population) == 0:
            population.append(base_params.copy())
        else:
            # Mutate internal parameters
            internal = base_params[1:-1] + np.random.normal(0, 0.03, size=n-2)
            candidate = np.zeros(n)
            candidate[0] = 0.0
            candidate[-1] = 1.0
            candidate[1:-1] = np.sort(np.clip(internal, 0.02, 0.98))
            population.append(candidate)

    def evaluate_fitness(params: np.ndarray, ctrl_pts: np.ndarray) -> float:
        """Evaluates multi-objective fitness function J(D, U) (Eq 4 in Paper 2)."""
        try:
            ev = BSplineEvaluator(ctrl_pts, params=params)
            pos, v, a, curv = ev.sample_path(100)

            # Data fidelity
            w1_data = 5.0 * np.sum(np.square(ev.eval_pos(base_params) - raw_waypoints))
            # Curvature penalty
            w2_curv = 15.0 * np.mean(np.square(curv)) + 50.0 * np.max(curv)
            # Path length
            diffs = np.diff(pos, axis=0)
            w4_len = 1.0 * np.sum(np.linalg.norm(diffs, axis=1))

            return float(w1_data + w2_curv + w4_len)
        except Exception:
            return 1e9

    best_params = base_params.copy()
    best_cost = evaluate_fitness(best_params, raw_waypoints)

    for gen in range(generations):
        costs = [evaluate_fitness(ind, raw_waypoints) for ind in population]
        best_idx = np.argmin(costs)
        if costs[best_idx] < best_cost:
            best_cost = costs[best_idx]
            best_params = population[best_idx].copy()

        # Selection & Mutation
        new_pop = [best_params.copy()]
        for _ in range(pop_size - 1):
            parent = population[np.random.choice(np.argsort(costs)[:5])]
            child_internal = parent[1:-1] + np.random.normal(0, 0.02, size=n-2)
            child = np.zeros(n)
            child[0] = 0.0
            child[-1] = 1.0
            child[1:-1] = np.sort(np.clip(child_internal, 0.02, 0.98))
            new_pop.append(child)
        population = new_pop

    # 2. Curvature-Aware Control Points Refinement
    refined_ctrl_pts = raw_waypoints.copy()

    for iter_ref in range(20):
        evaluator = BSplineEvaluator(refined_ctrl_pts, params=best_params)
        pos, v, a, curv = evaluator.sample_path(200)
        max_c = np.max(curv)
        if max_c > 0.35:
            high_curv_idx = np.argmax(curv)
            u_high = np.linspace(0, 1, 200)[high_curv_idx]
            # Smooth out control points near maximum curvature region
            for c_idx in range(1, n - 1):
                blend = np.exp(-((c_idx / (n - 1) - u_high) ** 2) / 0.08)
                mid_neighbor = 0.5 * (refined_ctrl_pts[c_idx - 1] + refined_ctrl_pts[c_idx + 1])
                refined_ctrl_pts[c_idx] += 0.20 * blend * (mid_neighbor - refined_ctrl_pts[c_idx])

    final_evaluator = BSplineEvaluator(refined_ctrl_pts, params=best_params)
    positions, vels, accs, curvatures = final_evaluator.sample_path(num_samples)

    comp_time = time.time() - t0
    metrics = compute_path_metrics(positions, curvatures, comp_time, raw_waypoints[0], raw_waypoints[-1])

    return positions, curvatures, metrics, final_evaluator
