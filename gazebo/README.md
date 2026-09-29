# Gazebo Physics Implementation for Adaptive B-Spline UAV Guidance & Quaternion Control

This directory contains the **Gazebo 3D Physics Simulation Stack** integrating both research papers:
1. **Paper 1**: *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* (3D A* Search + 3-Stage GA Adaptive B-Spline Path Smoothing in Cluttered 3D Airspace).
2. **Paper 2**: *"Full Quaternion Based Attitude Control for a Quadrotor"* (Singularity-Free Quaternion $P^2$ Attitude Control Law + '+' Motor Allocation Matrix + 6-DOF Rigid-Body Dynamics).

> **How "Gazebo" is used here.** Flight dynamics (6-DOF rigid body, RK4 at 500 Hz, motor mixing, wind)
> are computed by a Python engine (`GazeboFallbackEngine` in `src/gazebo_env.py`). The package
> generates Gazebo SDF worlds (obstacles + `models/quadrotor`) and, if `gz sim` / `ign gazebo` is
> installed, `GazeboEnvironment.launch()` opens that world (`--gui` for the window, headless otherwise;
> `GZ_SIM_RESOURCE_PATH` is set automatically). State and control are **not** exchanged with the native
> Gazebo process, so it is a world viewer, not a co-simulation. The engine reports obstacle and ground
> contacts (`collided`, `first_collision`, `ground_contact_steps`).
>
> The mission planner (`src/path_utils.py`) enforces a 15 m altitude floor and retries with denser
> A* waypoints until the smoothed spline is obstacle-free, because the shared smoother is not collision-aware.
> Reference timing is arc-length based and decelerates to stop at the goal.

---

## 📁 Directory Structure

```
gazebo/
├── README.md                           # Documentation & execution guide
├── run_simulation.py                   # Master integrated mission runner (A* -> GA B-Spline -> Gazebo 6-DOF flight)
├── view_in_gazebo_gui.py               # Live real-time interactive 3D flight viewer (matplotlib, no native Gazebo needed)
├── simulate_paper1.py                  # Paper 1 recreation (4-method smoothing benchmark in Gazebo airspace)
├── simulate_paper2.py                  # Paper 2 recreation (Step response, Sinusoidal tracking, 360° Flip)
├── models/
│   └── quadrotor/
│       ├── model.config                # Gazebo model configuration
│       └── model.sdf                   # Standard SDF 1.8 quadrotor model with Paper 2 inertia & rotors
├── worlds/
│   └── airspace_world.sdf              # SDF 1.8 airspace world with physics, lighting, and 3D obstacles
├── src/
│   ├── __init__.py                     # Package exports
│   ├── quaternion_math.py              # Hamilton quaternion algebra, Shepperd algorithm, and metrics
│   ├── motor_model.py                  # '+' quadrotor 4-rotor allocation and saturation matrix
│   ├── wind_model.py                   # Aerodynamic drag and sinusoidal wind gust disturbance model
│   ├── gazebo_env.py                   # SDF world generator, Gazebo launcher, Python 6-DOF engine w/ contact detection
│   ├── path_utils.py                   # Collision-checked smoothing + arc-length speed profile
│   ├── quadrotor_gazebo.py             # 6-DOF Gazebo quadrotor body wrapper & actuator interface
│   └── trajectory_tracker.py           # Outer position PD + Inner Quaternion P² guidance tracker
├── output_plots/                       # Telemetry plots & generated figures
└── tests/
    └── test_gazebo_sim.py              # Automated Gazebo unit test suite
```

---

## ⚙️ Core Architecture & Physics Parameters

- **Quadrotor Mass**: $m = 1.0\text{ kg}$
- **Inertia Tensor**: $I = \text{diag}(6.5 \times 10^{-4}, 6.5 \times 10^{-4}, 1.2 \times 10^{-3})\text{ kg}\cdot\text{m}^2$
- **Arm Length**: $d = 0.23\text{ m}$
- **Physics Timestep**: $\Delta t = 0.002\text{ s}$ ($500\text{ Hz}$)
- **Attitude Controller Gains**: $P_q = 20.0, P_w = 4.0$ (Paper 2)
- **Torque Bounds**: $\pm 4.0\text{ N}\cdot\text{m}$ for the Paper 2 benchmarks; the mission tracker (`run_simulation.py`) uses $\pm 1.0\text{ N}\cdot\text{m}$
- **Position Controller Gains**: $K_{p,pos} = [1.5, 1.5, 3.0], K_{d,pos} = [1.8, 1.8, 2.5]$

---

## 🚀 How to Run

### 1. Run Automated Unit Test Suite
Runs tests verifying quaternion math, Shepperd algorithm, motor allocation matrix, SDF world generation, hover equilibrium, and trajectory tracking:

```bash
python3 -m pytest gazebo/tests -v
```

---

### 2. Main Integrated Navigation & Flight Simulation
Runs 3D A* planning, 3-Stage GA Adaptive B-Spline path smoothing, exports the obstacle airspace to an SDF world, executes closed-loop 6-DOF flight tracking with motor mixing and dynamic gating, and saves 4 telemetry plots:

```bash
# Standard Simulation Run
python3 gazebo/run_simulation.py

# Options: --speed 25 --seed 42 --duration <s> --motor-lag

# Open the generated world in the Gazebo GUI (when `gz` / `ign` is installed)
python3 gazebo/run_simulation.py --gui

# Watch the flight live in the Gazebo GUI, paced to wall-clock time
python3 gazebo/run_simulation.py --realtime

# Enable Wind & Gust Disturbances
python3 gazebo/run_simulation.py --wind
```

#### Live playback in the Gazebo GUI (`--realtime`)

By default the Python 6-DOF engine computes the whole flight in a batch (much faster than real
time) and `--gui` only shows a **static** world — the quadrotor model doesn't move, since state
isn't exchanged with native Gazebo (see the note at the top of this file).

`--realtime` (implies `--gui`) changes that: while the flight loop runs, it periodically calls
`GazeboEnvironment.set_pose()`, which teleports the `quadrotor` model in the running Gazebo
process to the just-computed position/orientation via the `/world/<name>/set_pose` service
(provided by the `gz-sim-user-commands-system` plugin, already loaded in every generated world).
Each publish is paced by sleeping until wall-clock time catches up to simulation time, so you see
the drone fly the mission in the GUI over roughly the same duration the flight actually takes.

Requires a native Gazebo install (`gz sim`, Harmonic or newer — install via
`brew tap osrf/simulation && brew install gz-harmonic` on macOS). Notes and caveats:

- **This is a best-effort visual pose stream, not a physics co-simulation.** The model is
  teleported directly; native Gazebo's own physics engine never applies forces/torques to it, and
  the Python engine above remains the sole source of truth for the actual flight dynamics and the
  telemetry plots saved afterward.
- The requested update rate is set with `--fps` (default 15 Hz). The *achieved* rate is capped by
  the round-trip latency of each `gz service` call on your machine — if that's slower than the
  requested interval, playback falls proportionally behind real time (frames are never dropped or
  reordered, just delayed).
- If `gz`/`ign` isn't found, `--realtime` prints a warning and the flight still runs (at full
  compute speed, no GUI playback) rather than failing.

**Generated Artifacts** in `gazebo/output_plots/`:
- `gazebo_3d_flight.png`: 3D flight trajectory through obstacle field.
- `gazebo_position_tracking.png`: $X, Y, Z$ position tracking vs. reference.
- `gazebo_attitude_and_torques.png`: Roll, pitch, yaw and control torques ($M_x, M_y, M_z$).
- `gazebo_path_progress.png`: Path parameter $u$ progress and position tracking error over time.

---

### 3. Live Real-Time 3D Flight Viewer (no native Gazebo required)
Opens an interactive matplotlib 3D window and flies the same A* -> Adaptive B-Spline -> quaternion-controlled
mission live, paced to wall-clock time -- the Gazebo-folder equivalent of `mujoco/view_in_mujoco_gui.py`.
On reaching the goal it holds briefly, then loops back to the start:

```bash
python3 gazebo/view_in_gazebo_gui.py

# Options: --seed 42 --speed 25 --fps 30 --hold 3 --wind --motor-lag
```

- Left-click + drag to orbit the camera, scroll to zoom, close the window to exit.
- This renders with matplotlib's own 3D window (no `gz sim` install needed) -- the flight dynamics
  are the same `GazeboFallbackEngine` used everywhere else in this folder, just shown live instead
  of only saved as PNGs afterward.

---

### 4. Paper 1 Path Smoothing Benchmark
Recreates Table 2 of Paper 1 in Gazebo airspace across seeded obstacle environments, comparing **Baseline Polyline ($A^*$)**, **Uniform B-Spline**, **Classical Chord B-Spline**, and **Adaptive B-Spline (GA)**:

```bash
python3 gazebo/simulate_paper1.py
```

**Generated Artifacts**:
- `gazebo/output_plots/gazebo_paper1_comparison.png`

---

### 5. Paper 2 Attitude Control Maneuvers Benchmark
Recreates Figures 3 (Step Response), 5 (Sinusoidal Tracking), and 7 (360° Flip Maneuver) using the Quaternion $P^2$ attitude controller:

```bash
python3 gazebo/simulate_paper2.py
```

**Generated Artifacts**:
- `gazebo/output_plots/gazebo_paper2_step_response.png`
- `gazebo/output_plots/gazebo_paper2_sine_tracking.png`
- `gazebo/output_plots/gazebo_paper2_flip.png`
