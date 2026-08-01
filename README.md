# Adaptive B-Spline Smoothing and Altitude Control

An integrated UAV navigation stack that combines GA-optimised adaptive B-Spline path smoothing with a quaternion-based P² attitude controller, recreating and merging two published approaches:

1. **Path Planning** — *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* — a 3D A* planner generates a jagged baseline path through an obstacle field, which is then smoothed into a curvature-constrained B-Spline whose interior knots are placed by a genetic algorithm (GA) to minimize path length and curvature.
2. **Attitude Control** — Fresk & Nikolakopoulos, *"Full Quaternion Based Attitude Control for a Quadrotor"* (ECC 2013) — a singularity-free quaternion P² (proportional-proportional) controller with an outer attitude loop and inner angular-rate loop.

The two are merged so the B-Spline path's tangent direction generates real-time attitude references that the quaternion controller tracks, with path progress gated on attitude-error convergence. Both a Python and a MATLAB implementation are included.

## Repository Layout

```
path_planning/       3D A* planner, B-Spline path (SciPy-backed), GA knot optimizer
attitude_control/    Quaternion algebra, quadrotor rigid-body dynamics, P² controller
environment/         3D obstacle environment (boxes/spheres) used by A*
integration/         Merges path planning output into attitude references + sim loop
visualization/       Shared plotting helpers
matlab/uav_matlab/   Parallel MATLAB implementation (parts 1-4) with animations
results/             Generated figures from the Python scripts
results/presentation/ Polished figures for review/presentation use
```

### Top-level scripts

| Script | What it does |
|---|---|
| `paper1_recreation.py` | Recreates Table 2 of Paper 1 — compares 4 path-smoothing methods on an A*-generated baseline, averaged over 5 seeded trials |
| `paper2_recreation.py` | Recreates Figures 3, 5, 7 of Paper 2 — step, sinusoidal, and flip-maneuver responses of the quaternion controller in isolation |
| `test_paper2_controller.py` | Standalone validation of the P² controller (no path planning) against the paper's own scenarios |
| `main.py` | End-to-end pipeline: GA knot optimization → B-Spline path → merged attitude-tracking simulation, with full plot suite |
| `final_simulation.py` | Complete merged demo: A* + obstacle-aware adaptive B-Spline smoothing feeding attitude references into the P² controller |
| `validation_tests.py` | Robustness check of the merged A* + smoothing pipeline across multiple test scenarios |
| `generate_presentation_plots.py` | Reruns the full pipeline and renders polished figures to `results/presentation/` |

## Requirements

- Python 3.11+
- `numpy`, `scipy`, `matplotlib`, `pygad`

```bash
pip install numpy scipy matplotlib pygad
```

A `.venv` with these dependencies already installed is included in this repo for convenience.

## Running

```bash
python main.py                          # full GA + B-Spline + attitude-control pipeline
python final_simulation.py              # A* + obstacles + adaptive smoothing + control
python paper1_recreation.py             # Paper 1 Table 2 recreation
python paper2_recreation.py             # Paper 2 Figures 3/5/7 recreation
python test_paper2_controller.py        # controller-only validation
python validation_tests.py              # multi-scenario robustness check
python generate_presentation_plots.py   # polished figures -> results/presentation/
```

All scripts use the `Agg` matplotlib backend and save figures to `results/` (or `results/presentation/`) rather than opening interactive windows.

### MATLAB

The `matlab/uav_matlab/` tree mirrors the Python pipeline in four parts:

```
part1_bspline/        bspline_engine.m, knot_ga.m, run_paper1.m
part2_controller/     quaternion_utils.m, test_step_response.m
part3_merged/         run_merged_simulation.m  (path -> attitude control)
part4_environment/    astar_3d.m, obstacle_env.m, run_part4_full.m, run_phase4_dense.m
```

Run any `run_*.m` script from MATLAB with the repo on the path (each script calls `addpath(genpath(...))` itself). `part4_environment` scripts additionally render 3D obstacle-avoidance animations, saved as `.mp4` under `matlab/uav_matlab/results/`.

## Key implementation notes

- **Controller gains**: Paper 2's exact gains (`Pq=20`, `Pw=4`) are discrete-time unstable at `dt=0.01` given this project's low rotational inertia (`Ixx=Iyy=6.5e-4`). `paper2_recreation.py`, `final_simulation.py`, and `test_paper2_controller.py` integrate the control law and physics internally at `dt=0.0005` (2 kHz) while logging at the paper's own `dt=0.01` resolution. `main.py`'s merged simulation instead uses a reduced, empirically stable gain pair (`Pq=0.2`, `Pw=0.1`) at `dt=0.01`.
- **Path progress gating**: in the merged simulation, the path parameter `u` only advances once the current attitude tracking error has converged below a threshold, so the drone doesn't outrun its own attitude controller.
- **Curvature constraint**: the GA knot optimizer (`path_planning/knot_ga.py`) searches interior knot placement to keep the B-Spline's maximum curvature under a configurable `kappa_max`, evaluated against path length and straightness ratio.

## Results

Generated figures (path comparisons, curvature, attitude tracking, torques, quaternion error, angular velocity, path progress) are written to `results/`, with a curated set in `results/presentation/`. MATLAB animations of obstacle-avoidance runs are in `matlab/uav_matlab/results/`.
