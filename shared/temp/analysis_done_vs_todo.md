# Project Status: What's Done vs What Needs To Be Done

## 📋 Overview

Your project combines two papers:
- **Paper 1** (bspline1.pdf): *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* — path planning & smoothing
- **Paper 2** (Full_quaternion...pdf): *"Full Quaternion Based Attitude Control for a Quadrotor"* — attitude control

The goal is to **integrate both** so a UAV plans a smooth 3D path and then follows it using quaternion-based attitude control.

---

## ✅ What's Already Done (Python)

### Paper 1 — Path Planning & Smoothing

| Component | File | Status | Notes |
|---|---|---|---|
| 3D A* path search (26-connected grid) | [`astar.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/path_planning/astar.py) | ✅ Done | Collision-aware, with waypoint simplification |
| Cubic B-spline (Cox–de Boor basis) | [`bspline.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/path_planning/bspline.py) | ✅ Done | Chord-length & uniform parameterization, curvature evaluation |
| Stage 1 — Initialization (control point solve) | [`bspline.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/path_planning/bspline.py) | ✅ Done | Least-squares fit via `np.linalg.lstsq` |
| Stage 2 — GA knot optimization | [`knot_ga.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/path_planning/knot_ga.py) | ✅ Done | 4-term fitness (fidelity + curvature penalty + length + knot regularization) |
| Stage 3 — Curvature-aware control point refinement | [`curvature_refine.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/path_planning/curvature_refine.py) | ✅ Done | Finite-difference gradient descent on control points |
| 3-stage orchestrator | [`path_smoother.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/path_planning/path_smoother.py) | ✅ Done | Runs Stage 1→2→3 sequentially |
| 3D obstacle environment (cylinders + cuboids) | [`obstacles.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/environment/obstacles.py) | ✅ Done | Random placement, collision checking, 3D visualization |
| Paper 1 Table 2 recreation (4-method comparison) | [`paper1_recreation.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/paper1_recreation.py) | ✅ Done | 5-seed average across Baseline/Uniform/Classical/Adaptive |

### Paper 2 — Attitude Control

| Component | File | Status | Notes |
|---|---|---|---|
| Quaternion algebra (multiply, conjugate, inverse, Euler conversion, rotation matrix) | [`quaternion.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/attitude_control/quaternion.py) | ✅ Done | Hamilton product, from_euler (Eq.15), to_rotation_matrix, error computation |
| Quadrotor rotational dynamics (Eq.18) | [`quadrotor_model.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/attitude_control/quadrotor_model.py) | ✅ Done | RK4 integration with quaternion renormalization |
| P² attitude controller (Eq.21) | [`p2_controller.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/attitude_control/p2_controller.py) | ✅ Done | τ = −Pq·q_err − Pw·ω, with torque saturation ±4Nm |
| Paper 2 Figures 3, 5, 7 recreation | [`paper2_recreation.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/paper2_recreation.py) | ✅ Done | Step response, sinusoidal tracking, 360° flip |

### Integration (Merged System)

| Component | File | Status | Notes |
|---|---|---|---|
| Tangent-to-quaternion converter | [`trajectory_converter.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/integration/trajectory_converter.py) | ✅ Done | Builds rotation matrix from path tangent, converts to quaternion with sign continuity |
| Gated simulation loop (error-gated u advance) | [`simulation_loop.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/integration/simulation_loop.py) | ✅ Done | Path progress gated by ‖q_err‖ < threshold |
| SLERP interpolation for smooth references | [`trajectory_converter.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/integration/trajectory_converter.py#L92-L121) | ✅ Done | Prevents discontinuous quaternion jumps |
| Full merged demo (A* → Smoothing → Control) | [`final_simulation.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/final_simulation.py) | ✅ Done | 2kHz substep integration for stability |

### Testing & Validation

| Script | Status | Purpose |
|---|---|---|
| [`test_paper2_controller.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/test_paper2_controller.py) | ✅ Done | Standalone controller tests |
| [`validation_tests.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/validation_tests.py) | ✅ Done | 5-scenario robustness tests |

---

## ❌ What's MISSING — Needs To Be Done

These are the gaps between what the papers describe and what the current Python implementation has. These are real, substantive items that a reviewer or professor would notice.

---

### 🔴 CRITICAL GAPS (Core paper algorithms not implemented)

#### 1. **No Translational Dynamics / Position Controller**
> **Paper 2 covers this but it's NOT implemented.**

The current `QuadrotorModel` only has `step_attitude_only()` — it integrates **rotational dynamics only**. There is:
- ❌ No translational equations of motion ($m\ddot{p} = R \cdot F_{thrust} - mg\hat{z}$)
- ❌ No position controller (PD/PID on position → desired thrust + desired angles)
- ❌ No altitude controller (the project title says "Altitude Control" but there is **zero altitude control code**)
- ❌ No thrust computation from individual motor speeds ($F = k_f \sum \omega_i^2$)

The drone never actually *moves* along the path in 3D space. It only rotates in place while the path parameter `u` advances. **This is the biggest gap.**

> **What to implement**: Full 6-DOF dynamics (position + attitude), a position PD controller that computes desired total thrust and desired roll/pitch from position error, feeding into the existing P² attitude controller.

---

#### 2. **No Motor Mixing / Actuator Model**
> **Paper 2 Eq.(5-9) describes the motor-to-torque mapping — NOT implemented.**

- ❌ No individual motor speed computation from desired thrust + torques
- ❌ No motor speed limits
- ❌ No mapping from τ = [τ_x, τ_y, τ_z]ᵀ and total thrust F to individual motor thrusts [F₁, F₂, F₃, F₄]

Currently the P² controller outputs torque directly as if it's applied magically. A real quadrotor requires motor mixing:
$$\begin{bmatrix} F \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix} = \begin{bmatrix} k_f & k_f & k_f & k_f \\ 0 & -dk_f & 0 & dk_f \\ dk_f & 0 & -dk_f & 0 \\ -k_m & k_m & -k_m & k_m \end{bmatrix} \begin{bmatrix} \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \\ \omega_4^2 \end{bmatrix}$$

---

#### 3. **GA Fitness Missing F_dyn (Dynamic Feasibility) Term**
> **Paper 1 Eq.(4) specifies this — acknowledged as omitted in the documentation.**

The GA fitness function in [`knot_ga.py`](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/path_planning/knot_ga.py) uses a 4-term cost, but the paper's Eq.(4) includes a **dynamic feasibility term** $F_{dyn}$ that penalizes paths requiring velocities/accelerations beyond the UAV's physical limits:
$$F_{dyn} = \int \max(0, \|\ddot{C}(u)\| - a_{max})^2 \, du$$

Currently the code doesn't enforce any velocity/acceleration limits on the path.

---

### 🟡 IMPORTANT GAPS (Needed for a complete paper replication)

#### 4. **No Velocity Profiling / Time-Optimal Parameterization**
- ❌ The path parameter `u` advances in **fixed steps** (`u_step=0.005`), not based on a velocity profile
- ❌ No trapezoidal/S-curve velocity profile along the path
- ❌ No relationship between `u` and actual drone speed/acceleration limits
- The drone's "speed" along the path is entirely artificial — controlled by `u_step` and `error_threshold`, not by physics

> **What to implement**: A time-parameterization layer that converts from spline parameter `u` to time `t` using a velocity profile respecting $v_{max}$ and $a_{max}$ constraints.

---

#### 5. **No Collision Checking of the Smoothed Path**
- A* generates a collision-free polyline
- B-spline smoothing can push the path **into obstacles** (control points are shifted by GA and curvature refinement)
- ❌ No post-smoothing collision validation
- ❌ No constraint in the GA fitness to penalize obstacle collisions

> **What to implement**: Add an obstacle-proximity penalty term to the GA fitness, or run a post-smoothing collision check and re-plan if violated.

---

#### 6. **`visualization/plotter.py` is Empty**
- The file exists but is completely empty (0 bytes of content)
- All plotting is scattered across `main.py`, `final_simulation.py`, `paper1_recreation.py`, `paper2_recreation.py`
- ❌ No reusable plotting module

---

#### 7. **No Wind/Disturbance Model**
- Paper 2 mentions robustness to external disturbances
- ❌ Only Gaussian noise on ω measurements is implemented
- ❌ No external force/torque disturbances (wind gusts)
- ❌ No disturbance rejection analysis

---

### 🟢 MINOR / POLISH GAPS

#### 8. **No Formal Unit Tests (pytest/unittest)**
- Test files exist but are plain scripts, not proper test frameworks
- ❌ No `pytest` or `unittest` based test suite
- ❌ No CI/CD or automated verification

#### 9. **No `requirements.txt` or `pyproject.toml`**
- Dependencies: `numpy`, `scipy`, `matplotlib`, `pygad` — but not declared anywhere
- ❌ Missing dependency specification file

#### 10. **Quaternion `from_rotation_matrix()` Has a Bug for Non-Positive Trace**
- The else branch in [`quaternion.py` L88-92](file:///d:/S5_Projects/Drones_S5_CD8_Adaptive_BSpline_Smoothing_and_Altitude_Control/attitude_control/quaternion.py#L88-L92) uses the wrong Shepperd method — it takes square roots but doesn't use the correct sign-determination logic. This can produce incorrect quaternions for certain rotation matrices.

#### 11. **No Gyroscopic Effects from Propellers**
- Paper 2 Eq.(18) includes a $J_r \cdot (\omega \times e_3) \cdot \Omega_r$ gyroscopic term from the combined propeller spin
- ❌ This is omitted in the current dynamics model

---

## 📊 Summary Scorecard

| Category | Items from Papers | Implemented | Missing |
|---|:---:|:---:|:---:|
| **Paper 1 — Path Planning** | 7 | 6 | 1 (F_dyn term) |
| **Paper 2 — Attitude Control** | 6 | 3 | 3 (translational dynamics, motor model, gyroscopic) |
| **Integration** | 4 | 3 | 1 (velocity profiling) |
| **Engineering Quality** | 5 | 2 | 3 (tests, deps, viz module) |
| **Total** | **22** | **14** | **8** |

---

## 🎯 Recommended Work Distribution (3 Team Members)

### Person 1 — Translational Dynamics & Position Control (HIGHEST PRIORITY)
1. Add full 6-DOF dynamics to `QuadrotorModel` (position, velocity, translation forces)
2. Implement position PD controller: position error → desired thrust + desired attitude
3. Connect: path position → position controller → attitude controller → dynamics
4. The drone should actually **fly** along the path in 3D, not just rotate in place

### Person 2 — Path Planning Completeness
1. Add `F_dyn` (dynamic feasibility) term to GA fitness in `knot_ga.py`
2. Add collision-checking penalty for smoothed paths in the GA fitness
3. Implement velocity profiling / time-optimal parameterization along the spline
4. Fill `visualization/plotter.py` with reusable plotting functions

### Person 3 — Motor Model, Testing & Polish
1. Implement motor mixing matrix (4 motors → thrust + 3 torques)
2. Add motor speed limits and compute individual motor speeds
3. Add wind/disturbance model for robustness testing
4. Create `requirements.txt`, write proper `pytest` unit tests
5. Fix `from_rotation_matrix()` quaternion bug

---

## 🏁 What "Properly Combined" Looks Like

When the project is truly complete, running the simulation should:

1. **Generate 3D obstacles** → done ✅
2. **A* finds collision-free waypoints** → done ✅
3. **3-stage B-spline smooths the path** (with collision checking + F_dyn) → 80% done
4. **Velocity profile assigns time to each path point** → ❌ missing
5. **Position controller tracks path positions** → ❌ missing
6. **Position error → desired thrust + desired roll/pitch** → ❌ missing
7. **Attitude controller tracks desired quaternion** → done ✅
8. **Motor mixing computes individual motor speeds** → ❌ missing
9. **Full 6-DOF dynamics integrates everything** → ❌ missing
10. **Drone actually flies the smoothed path** in 3D with position + attitude tracking → ❌ missing

The current system does steps 1–3 and 7, but **skips 4–6 and 8–10**. The drone never moves through space — it only rotates on the spot while `u` is artificially advanced.
