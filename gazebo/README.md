# Gazebo Physics Implementation for Adaptive B-Spline UAV Guidance & Quaternion Control

This directory contains the **Gazebo 3D Physics Simulation Stack** integrating both research papers:
1. **Paper 1**: *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* (3D A* Search + 3-Stage GA Adaptive B-Spline Path Smoothing in Cluttered 3D Airspace).
2. **Paper 2**: *"Full Quaternion Based Attitude Control for a Quadrotor"* (Singularity-Free Quaternion $P^2$ Attitude Control Law + '+' Motor Allocation Matrix + 6-DOF Rigid-Body Dynamics).

---

## 📁 Directory Structure

```
gazebo/
├── README.md                           # Documentation & execution guide
├── run_simulation.py                   # Master integrated mission runner (A* -> GA B-Spline -> Gazebo 6-DOF flight)
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
│   ├── gazebo_env.py                   # World generator from Environment3D & Gazebo launcher/fallback
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
- **Torque Bounds**: $\pm 4.0\text{ N}\cdot\text{m}$
- **Position Controller Gains**: $K_{p,pos} = [1.5, 1.5, 3.0], K_{d,pos} = [1.8, 1.8, 2.5]$

---

## 🚀 How to Run

### 1. Run Automated Unit Test Suite
Runs tests verifying quaternion math, Shepperd algorithm, motor allocation matrix, SDF world generation, hover equilibrium, and trajectory tracking:

```bash
python3 -m unittest gazebo/tests/test_gazebo_sim.py
```

---

### 2. Main Integrated Navigation & Flight Simulation
Runs 3D A* planning, 3-Stage GA Adaptive B-Spline path smoothing, exports the obstacle airspace to an SDF world, executes closed-loop 6-DOF flight tracking with motor mixing and dynamic gating, and saves 4 telemetry plots:

```bash
# Standard Simulation Run
python3 gazebo/run_simulation.py

# Visual 3D GUI Mode (when native Gazebo is installed)
python3 gazebo/run_simulation.py --gui

# Enable Wind & Gust Disturbances
python3 gazebo/run_simulation.py --wind
```

**Generated Artifacts** in `gazebo/output_plots/`:
- `gazebo_3d_flight.png`: 3D flight trajectory through obstacle field.
- `gazebo_position_tracking.png`: $X, Y, Z$ position tracking vs. reference.
- `gazebo_attitude_and_torques.png`: Roll, pitch, yaw and control torques ($M_x, M_y, M_z$).
- `gazebo_path_progress.png`: Path parameter $u$ progress and position tracking error over time.

---

### 3. Paper 1 Path Smoothing Benchmark
Recreates Table 2 of Paper 1 in Gazebo airspace across seeded obstacle environments, comparing **Baseline Polyline ($A^*$)**, **Uniform B-Spline**, **Classical Chord B-Spline**, and **Adaptive B-Spline (GA)**:

```bash
python3 gazebo/simulate_paper1.py
```

**Generated Artifacts**:
- `gazebo/output_plots/gazebo_paper1_comparison.png`

---

### 4. Paper 2 Attitude Control Maneuvers Benchmark
Recreates Figures 3 (Step Response), 5 (Sinusoidal Tracking), and 7 (360° Flip Maneuver) using the Quaternion $P^2$ attitude controller:

```bash
python3 gazebo/simulate_paper2.py
```

**Generated Artifacts**:
- `gazebo/output_plots/gazebo_paper2_step_response.png`
- `gazebo/output_plots/gazebo_paper2_sine_tracking.png`
- `gazebo/output_plots/gazebo_paper2_flip.png`
