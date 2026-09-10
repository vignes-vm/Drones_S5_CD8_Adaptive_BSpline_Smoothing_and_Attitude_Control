# PyBullet Physics Implementation for Adaptive B-Spline UAV Guidance & Quaternion Control

This directory contains the **PyBullet 3D Physics Simulation Stack** integrating both research papers:
1. **Paper 1**: *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* (3D A* Search + Douglas–Peucker Pruning + 3-Stage GA Adaptive B-Spline Path Smoothing).
2. **Paper 2**: *"Full Quaternion Based Attitude Control for a Quadrotor"* (Singularity-Free Quaternion $P^2$ Attitude Control Law + Motor Mixing Matrix + 6-DOF Rigid-Body Rotational & Translational Dynamics).

---

## 📁 Directory Structure

```
pybullet/
├── __init__.py                         # Package exports
├── README.md                           # Documentation & execution guide
├── run_simulation.py                   # Main integrated demo script (A* -> GA B-Spline -> PyBullet Flight)
├── simulate_paper1.py                  # Paper 1 recreation (4-method smoothing benchmark)
├── simulate_paper2.py                  # Paper 2 recreation (Step, Sinusoidal, 360° Flip attitude maneuvers)
├── output_plots/                       # Generated plot results & telemetry metrics
│   ├── pybullet_3d_flight.png
│   ├── pybullet_position_tracking.png
│   ├── pybullet_actuators_and_error.png
│   ├── pybullet_paper2_step_response.png
│   ├── pybullet_paper2_sine_tracking.png
│   └── pybullet_paper2_flip.png
├── src/
│   ├── __init__.py                     # Module exports
│   ├── quaternion_math.py              # Hamilton & PyBullet quaternion algebra + Shepperd algorithm
│   ├── motor_model.py                  # Motor allocation matrix (4 rotors -> thrust & torques)
│   ├── quadrotor_pybullet.py           # 6-DOF PyBullet Quadrotor physical multi-body wrapper
│   ├── pybullet_env.py                 # PyBullet 3D obstacle environment (cylinders, cuboids, ground)
│   ├── trajectory_tracker.py          # Outer-loop 3D position PD + Inner-loop Paper 2 Quaternion P^2
│   └── wind_model.py                  # Environmental wind & gust aerodynamic drag model
└── tests/
    └── test_pybullet_sim.py            # Automated PyBullet unit test suite
```

---

## ⚙️ Core Architecture & Physics Parameters

- **Quadrotor Mass**: $m = 1.0\text{ kg}$
- **Inertia Tensor**: $I = \text{diag}(6.5 \times 10^{-4}, 6.5 \times 10^{-4}, 1.2 \times 10^{-3})\text{ kg}\cdot\text{m}^2$
- **Arm Length**: $d = 0.23\text{ m}$
- **Physics Timestep**: $\Delta t = 0.002\text{ s}$ ($500\text{ Hz}$)
- **Attitude Controller Gains**: $P_q = 20.0, P_w = 4.0$ (Paper 2)
- **Torque Bounds**: $\pm 4.0\text{ N}\cdot\text{m}$
- **Position Controller Gains**: $K_{p,pos} = [2.5, 2.5, 4.0], K_{d,pos} = [2.0, 2.0, 3.0]$

---

## 🚀 What to Run

### 1. Main Integrated Navigation & Flight Simulation
Runs 3D A* planning, GA Adaptive B-Spline path smoothing, loads the physical 6-DOF Quadrotor into PyBullet, tracks the trajectory using position PD + quaternion $P^2$ attitude control with motor mixing, and saves telemetry plots:

```bash
# Headless Direct Physics Mode (Default)
py -3.12 pybullet/run_simulation.py

# Visual 3D Interactive GUI Mode
py -3.12 pybullet/run_simulation.py --gui

# Enable Environmental Wind & Gust Disturbances
py -3.12 pybullet/run_simulation.py --gui --wind
```

---

### 2. Paper 1 Path Smoothing Benchmark
Recreates Table 2 of Paper 1 in PyBullet space across 5 seeded obstacle environments, comparing **Baseline Polyline ($A^*$)**, **Uniform B-Spline**, **Classical Chord B-Spline**, and **Adaptive B-Spline (GA)**:

```bash
py -3.12 pybullet/simulate_paper1.py
```

---

### 3. Paper 2 Attitude Control Maneuvers Benchmark
Recreates Figures 3 (Step Response), 5 (Sinusoidal Tracking), and 7 (360° Flip Maneuver) using the Quaternion $P^2$ attitude controller in PyBullet:

```bash
py -3.12 pybullet/simulate_paper2.py
```

---

### 4. Run Automated PyBullet Unit Test Suite
Runs tests verifying quaternion math, Shepperd algorithm, motor allocation matrix, trajectory tracker, and wind model:

```bash
py -3.12 -m unittest pybullet/tests/test_pybullet_sim.py
```
