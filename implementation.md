# Implementation Log — Adaptive B-Spline Smoothing and Altitude Control

This document is a detailed, technical account of what this project is trying to do, what has
actually been built and verified so far, how it currently performs, and what remains open. It is
written as a working log/status report, separate from `README.md` (which documents the system as
a reference for someone using it). Read this if you want to know **exactly where the project
stands right now** and **why** it's built the way it is.

---

## 1. What this project is

The project recreates and merges two published papers into a single, obstacle-aware UAV
navigation pipeline:

1. **Paper 1** — Nikolaiev & Novotarskyi, *"An Enhanced Adaptive B-spline Smoothing Approach for
   UAV Path Planning"*, IJISA, Vol. 17, No. 4, 2025 (`bspline1.pdf` in the repo root). This is the
   path-planning half: take a raw, jagged waypoint path (from A\*), and smooth it into a
   curvature-bounded cubic B-Spline using a genetic algorithm to place knots and a local search to
   refine control points.
2. **Paper 2** — Fresk & Nikolakopoulos, *"Full Quaternion Based Attitude Control for a
   Quadrotor,"* ECC 2013. This is the control half: a singularity-free quaternion P² (proportional–
   proportional) attitude controller that tracks a reference orientation.

The two are merged: the smoothed spline's local tangent becomes a reference attitude, and the
quaternion controller tracks it, with the path parameter `u` gated so the drone never outruns its
own attitude loop.

Both a **Python** implementation (this session's focus) and a **MATLAB** implementation
(`matlab/uav_matlab/`, parallel, includes 3D animations) exist. They are meant to be kept in sync,
though — as documented below — they currently are **not** fully in sync.

---

## 2. Pipeline architecture

```
Raw waypoints
     │
     ▼
┌─────────────────────┐
│ 3D A* planner        │  environment/obstacles.py, path_planning/astar.py
│ (26-connected grid)  │  2000×2000×500 m space, cylinder/cuboid obstacles + buffer
└─────────┬────────────┘
          │ Douglas–Peucker pruned waypoints
          ▼
┌───────────────────────────────────────────────────────────┐
│ Adaptive B-Spline Smoothing (Paper 1, §3.4 algorithm)       │
│                                                               │
│  Stage 1: Initialization                                    │
│    — least-squares control polygon + baseline knot vector   │
│    (path_planning/bspline.py — BSplinePath)                 │
│                                                               │
│  Stage 2: Global Optimization (GA over the knot vector)     │
│    (path_planning/knot_ga.py — KnotGA)                      │
│                                                               │
│  Stage 3: Local Curvature Refinement  ◄── NEWLY IMPLEMENTED  │
│    — nudges control points directly where curvature is      │
│    still too high after the GA                              │
│    (path_planning/curvature_refine.py — CurvatureRefiner)   │
│                                                               │
│  Orchestrated by: path_planning/path_smoother.py             │
│    — AdaptiveBSplineSmoother.smooth(waypoints)               │
└─────────────────────────┬─────────────────────────────────┘
                           │ smoothed B-Spline C(u)
                           ▼
┌───────────────────────────────────────────────────────────┐
│ Guidance + Attitude Control (Paper 2, merged via integration/)│
│                                                               │
│  Tangent C'(u) → reference quaternion q_ref                 │
│    (integration/trajectory_converter.py)                    │
│                                                               │
│  Quaternion P² controller drives drone toward q_ref          │
│    (attitude_control/p2_controller.py, quaternion.py,       │
│     quadrotor_model.py — RK4 rigid-body integration)         │
│                                                               │
│  u only advances once attitude error < threshold             │
│    (integration/simulation_loop.py — MergedSimulation)       │
└───────────────────────────────────────────────────────────┘
```

---

## 3. What has been implemented — module by module

### 3.1 Environment (`environment/obstacles.py`)
- `Environment3D`: a 2000×2000×500 m space, start `(0,0,20)`, goal `(2000,1800,200)` — matching
  Paper 1's simulation setup exactly.
- `Obstacle3D`: cylinders and cuboids, each with a configurable safety buffer (paper default: 5 m).
- Random obstacle placement (`add_random_obstacles`) with rejection sampling to avoid overlaps and
  keep clearance around start/goal.
- Collision checking (`check_path_collision`) and a matplotlib 3D visualizer.

**Status: complete, matches the paper's environment description.**

### 3.2 Path planning (`path_planning/astar.py`)
- `AStarPlanner3D`: 26-connected voxel-grid A\* search (all `{-1,0,1}³ \ {0}` neighbor offsets),
  Euclidean heuristic, standard open/closed-set implementation with a binary heap.
- Falls back to a straight line with a warning if start/goal collide or no path is found.
- `smooth_waypoints`: a simple minimum-distance waypoint reducer (not full Douglas–Peucker despite
  the README's flowchart language — see §5 below for the discrepancy note).

**Status: complete and functionally correct.**

### 3.3 B-Spline representation (`path_planning/bspline.py`)
- `BSplinePath`: pure NumPy implementation of a degree-`k` (default cubic, `k=3`) B-Spline.
  - Parameterization: chord-length (default) or uniform.
  - Knot vector: **uniform** by default when none is supplied (not Piegl & Tiller averaging — see
    §5).
  - Basis functions: hand-written recursive Cox–de Boor (`_basis`) — the `scipy.interpolate.BSpline`
    import at the top of the file is unused.
  - Control points: solved via least-squares normal equations (`np.linalg.lstsq`), which degrades
    gracefully instead of failing on a near-singular basis matrix; only an **exact** interpolation
    when the number of control points equals the number of waypoints (the default case).
  - `evaluate`, `evaluate_derivative` (central finite differences, `h=1e-5`), `compute_curvature`
    (`‖C'×C''‖/‖C'‖³`), `sample_path`, `path_length`, `max_curvature`.

**Status: complete, verified numerically (see §4).**

### 3.4 GA knot optimization (`path_planning/knot_ga.py`)
- `KnotGA`: wraps `pygad` to search over interior knot positions.
- Search space: fixed absolute range `[0.01, 0.99]` per interior knot (not bounded relative to a
  baseline — see §5 for how this differs from MATLAB and the paper's description).
- Cost function — **4 of the paper's 5 terms** (`Fdyn`, the velocity/acceleration penalty, is
  missing entirely — confirmed by grepping the whole codebase for any velocity/acceleration
  constraint, zero hits):

  | Term | Weight (default) | Meaning |
  |---|---:|---|
  | Data fidelity (MSE) | `w1 = 1.0` | mean-squared distance between the spline and the original waypoints (mean, not sum, to avoid drowning out curvature at scale) |
  | Curvature penalty | `w2 = 2.0` | `Σ max(0, κ(u) − κ_max)²` over 50 samples |
  | Path length | `w3 = 1.0` | arc length over 200 samples |
  | Knot regularization | `w4 = 0.5` | penalizes uneven interior-knot spacing |

- Returns the best (sorted) knot vector found.

**Status: complete and working, but structurally simpler than both the paper and the MATLAB
implementation** (see §5).

### 3.5 Local curvature-aware control-point refinement — **NEW THIS SESSION**
This was the single biggest gap found when the actual paper PDF was read and compared against the
code: Paper 1 §3.4 specifies **three** algorithm stages, not two. The third stage — a dedicated
local search that directly moves **control points** (not knots) wherever curvature still exceeds
`κ_max` after the GA — was entirely unimplemented. `path_planning/curvature_refine.py` and
`path_planning/path_smoother.py` existed in the repo tree as **empty placeholder files**. This
applied to *both* the Python and MATLAB implementations (`bspline_engine.m` / `knot_ga.m` were
checked directly — no refinement stage there either).

**What was built (Python only, this session):**

- **`path_planning/curvature_refine.py` — `CurvatureRefiner`**
  Implements Paper 1 §3.3 ("Curvature-aware Control Points Adjustment"). Algorithm, per call to
  `.refine(spline)`:
  1. Sample curvature at 200 points along the spline; find the single worst violation
     (`κ(u) > κ_max`).
  2. Identify which control points influence that `u` — for a degree-`k` basis, control point `i`
     has nonzero support on `[t_i, t_{i+k+1})`, so at most `k+1 = 4` control points are affected
     for a cubic spline.
  3. For each affected control point, estimate `∇κ` at the worst `u` via central finite
     differences (`h = 1e-3`) on each of the 3 axes.
  4. Step the control point against the gradient (`step_size`, default `5.0`). Keep the move only
     if curvature at that `u` actually dropped; otherwise revert.
  5. If no control point improved things this round, shrink the step size (`shrink=0.5`) down to
     a floor (`min_step=0.05`).
  6. Repeat until no sample exceeds `κ_max`, or `max_iters` (default `30`) is reached.

  This is a genuine **derivative-free local search** as the paper describes ("gradient descent or
  a derivative-free algorithm... to shift `d_i` in directions that lower `‖C'(u)×C''(u)‖`"), and
  it operates on the same curvature definition the paper uses (Eq. 2).

- **`path_planning/path_smoother.py` — `AdaptiveBSplineSmoother`**
  Orchestrates all three stages behind one call: `.smooth(waypoints)` builds the stage-1 baseline
  spline, runs `KnotGA` for stage 2, then runs `CurvatureRefiner` on a fresh fit of the GA's knots
  for stage 3, exposing `stage1_spline` / `stage2_spline` / `stage3_spline` for before/after
  comparison.

- **`main.py`** was rewired to call `AdaptiveBSplineSmoother` instead of hand-calling `KnotGA` +
  `BSplinePath` directly, so the top-level demo now actually runs the paper's complete algorithm.

**Verification performed this session:**
- Ran the full 3-stage pipeline on the repo's 8-waypoint test set at the default `kappa_max=0.5` —
  confirmed stage 3 correctly does nothing (0 iterations), because this environment's natural
  curvature (~0.003–0.04 m⁻¹, per the existing `validation_tests.py` comment) never approaches
  0.5. This is expected behavior, not a bug.
- Re-ran with an artificially tight `kappa_max=0.01` to force violations, and confirmed the
  refiner actually engages and converges:
  - First run (`step_size=10.0`, `n_inner_knots=6`): stage 2 left `max_curvature=0.0174`; stage 3
    brought it to `0.0092` in 2 iterations (path length 2900.0m → 2891.0m).
  - Second run (different GA seed run, same config): stage 2 left `max_curvature=0.0630`; stage 3
    brought it to `0.0088` in 14 iterations (path length 2898.8m → 2881.2m), visibly fighting
    curvature spikes at several different `u` locations before converging.
- Ran `main.py` end-to-end after the rewire — completes without errors, produces the same
  structure of output/plots as before (numbers shift slightly run-to-run because the attitude
  simulation injects Gaussian noise).

**Status: implemented and verified working. Not yet ported to MATLAB.**

### 3.6 Attitude control (`attitude_control/`)
- `quaternion.py` — `Quaternion` class: Hamilton product, conjugate/inverse, normalize, Euler
  conversions (both directions), rotation-matrix conversions (both directions), `error()`.
- `quadrotor_model.py` — `QuadrotorModel`: rigid-body rotational dynamics
  (`I·ω̇ = τ − ω×(Iω)`), RK4 integration (`step_attitude_only`) with quaternion renormalization
  every step, Paper 2's inertia values (`Ixx=Iyy=6.5e-4`, `Izz=1.2e-3`).
- `p2_controller.py` — `P2Controller`: quaternion error computed with shortest-path correction
  (flip sign if `q_err.q0 < 0`), P² law `τ = −Pq·q_err(1:3) − Pw·ω`, saturated to `±4 Nm`.

**Status: complete, matches Paper 2's equations directly.**

### 3.7 Integration (`integration/`)
- `trajectory_converter.py` — `TrajectoryConverter`: builds an orthonormal frame from the spline
  tangent (with a fallback "up" vector to avoid gimbal-adjacent singularities), converts to a
  reference quaternion, keeps quaternion continuity (flips sign to avoid double-cover jumps),
  smooths the tangent over a small window, and provides a static SLERP helper.
- `simulation_loop.py` — `MergedSimulation`: runs the full merged sim. Path parameter `u` only
  advances once `‖q_err(1:3)‖ < error_threshold` (default `0.05`), advancing by `u_step` (default
  `0.005`) and SLERPing the reference quaternion 30% of the way toward the new target each time
  (avoids reference-quaternion jumps).

**Status: complete, this is where the two papers actually get coupled.**

### 3.8 Top-level scripts
| Script | Purpose | Status |
|---|---|---|
| `main.py` | Full pipeline demo — now uses the complete 3-stage smoother | ✅ updated + verified this session |
| `final_simulation.py` | A\* + obstacles + adaptive smoothing + control, Paper 2's exact gains at 2 kHz internal rate | untouched, not re-verified this session |
| `paper1_recreation.py` | Recreates Table 2 of Paper 1 (4-method comparison, 5 seeded trials) | untouched |
| `paper2_recreation.py` | Recreates Paper 2's Figures 3/5/7 | untouched |
| `test_paper2_controller.py` | Controller-only validation | untouched |
| `validation_tests.py` | Multi-scenario robustness check across random obstacle fields | untouched |
| `generate_presentation_plots.py` | Polished figures for `results/presentation/` | untouched |

None of the untouched scripts import `curvature_refine.py`/`path_smoother.py` yet — only `main.py`
was rewired this session. They still use `KnotGA` + `BSplinePath` directly (2-stage pipeline).

---

## 4. Current results

### 4.1 Paper 1 recreation (reference values, from the paper itself)

| Method | Length (m) | Max κ (m⁻¹) | Straightness Ratio |
|---|---:|---:|---:|
| Baseline Polyline | 2984.6 | 0.65 | 1.107 |
| Uniform B-Spline | 2840.8 | 0.56 | 1.053 |
| Classical B-Spline (chord-length) | 2803.2 | 0.49 | 1.039 |
| Adaptive B-Spline (GA) | 2731.4 | 0.38 | 1.013 |

These are the paper's own published numbers; `paper1_recreation.py` / `run_paper1.m` attempt to
reproduce this pattern (shorter + lower curvature as methods get more adaptive) on the repo's own
A\*-derived waypoint set — not to hit these exact numbers, since the underlying environment differs.

### 4.2 This session's actual runs (8-waypoint hand-picked test set, `main.py`)

**Default config (`kappa_max=0.5`, stage 3 inactive):**
```
Uniform B-Spline:        length=3065.4m, max_curvature=0.0070
Adaptive B-Spline (GA):  length=2893.2m, max_curvature=0.0216
Improvement:             172.2m shorter (5.6%), straightness 1.137 → 1.073
```

**Attitude control (same run):**
```
Controller gains:     Pq=0.2, Pw=0.1  (empirically stable pair, not paper's Pq=20/Pw=4)
Max torque applied:   0.075 Nm  (well within the ±4 Nm limit)
Steady-state error:   0.01464
Final error:          0.01311
Final path progress:  99.0% of u in 20s simulated
```

**Stress-testing the new refinement stage (`kappa_max=0.01`, forced violation):**
```
Stage 2 (GA only):        max_curvature=0.0630
Stage 3 (+ refinement):   max_curvature=0.0088   (14 iterations, length 2898.8m → 2881.2m)
```
This confirms the refinement stage does real, measurable work when curvature constraints are
actually binding — at the repo's default `kappa_max=0.5` it's a no-op because nothing violates
that limit in the first place.

### 4.3 What's *not* yet re-verified
`final_simulation.py`, `paper2_recreation.py`, `validation_tests.py`, and
`generate_presentation_plots.py` were not re-run this session — they're unaffected by the changes
(none of them import the new modules) but their last-known-good state is whatever is already
checked into `results/`.

---

## 5. Known gaps and deviations from the papers (honest accounting)

These were all found by reading the actual paper PDF and diffing it against the code — not
guessed:

1. **`Fdyn` (dynamic feasibility / velocity-acceleration penalty) is missing** from the GA cost
   function in *both* Python and MATLAB. The paper's Eq. 4 objective has 5 weighted terms; this
   codebase implements 4 (Python) or effectively 2 (MATLAB: length + curvature penalty only).
2. **MATLAB lacks the stage-3 local curvature refinement** entirely (confirmed by reading
   `bspline_engine.m`/`knot_ga.m` directly). Python now has it; MATLAB doesn't. This is the
   headline item for "what's next."
3. **Python's GA search space differs from both the paper and MATLAB.** The paper and MATLAB
   bound each interior knot to `±0.15` around a knot-averaging baseline (guarantees a
   well-conditioned, sorted knot vector by construction). Python's `KnotGA` instead searches a
   fixed absolute range `[0.01, 0.99]` with no baseline relationship.
4. **Python's default knot vector is uniform, not averaged.** `BSplinePath` without an explicit
   `knot_vector` argument builds uniform interior knots. The Piegl & Tiller averaging formula
   (used as MATLAB's default, and required for the "Classical B-Spline" comparison method) exists
   only as a separately duplicated helper function in `paper1_recreation.py` and
   `validation_tests.py`, not inside `bspline.py` itself.
5. **`AStarPlanner3D.smooth_waypoints` is a minimum-distance filter, not Douglas–Peucker**, despite
   the README's pipeline diagram naming Douglas–Peucker explicitly. (Flagged here for
   completeness; not yet corrected in code or README.)
6. **No `requirements.txt` and no committed virtual environment** — dependencies must be installed
   manually (`pip install numpy scipy matplotlib pygad`). `pygad` in particular was found to be
   genuinely missing from this machine's Python environment and was installed this session to run
   verification.
7. **`n_inner_knots` and `kappa_max` vary per script** with no single source of truth — `main.py`
   uses `n_inner_knots=6, kappa_max=0.5`; `validation_tests.py` uses `n_inner_knots=10,
   kappa_max=0.05` (tuned down because its A\*-derived paths only reach ~0.003–0.04 m⁻¹ curvature,
   so the paper's default `kappa_max` never triggers any penalty there).
8. **Controller gain mismatch by design, not bug**: Paper 2's exact gains (`Pq=20, Pw=4`) are
   discrete-time unstable at this project's timestep/inertia combination unless integrated at
   ≥2 kHz internally. `main.py`'s merged simulation instead uses a reduced, empirically-stable
   pair (`Pq=0.2, Pw=0.1`) at the coarser `dt=0.01`. This is documented and intentional, not an
   oversight.

---

## 6. What's next

Immediate next step (as discussed): **MATLAB simulation work.** Concretely, the options on the
table are:

- **Port the stage-3 curvature refinement to MATLAB** (`part1_bspline/curvature_refine.m` or
  similar), so `run_paper1.m` and `run_part4_full.m`/`run_phase4_dense.m` run the same complete
  3-stage algorithm the Python side now does — bringing the two implementations back to parity.
- **Run the existing MATLAB scripts** (`run_paper1`, `test_step_response`,
  `run_merged_simulation`, `run_part4_full`, `run_phase4_dense`) to check their current output
  against what's already checked into `matlab/uav_matlab/results/`, and decide whether any of
  those figures/animations are stale relative to the Python-side changes made this session.
- Optionally close some of the other gaps in §5 (add the `Fdyn` term, align the Python GA's search
  space with the baseline-relative bounding the paper describes, fix the Douglas–Peucker naming
  mismatch) — deferred until MATLAB parity is settled, since those are cheaper, more contained
  changes.

This document should be updated as that work proceeds, rather than left as a point-in-time
snapshot — treat it as a running log, not a one-off report.
