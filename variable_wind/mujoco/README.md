# MuJoCo Quadrotor Simulation with Variable Wind

Combines **Paper 1** (3D A* Path Planning + GA Adaptive B-Spline Smoothing) and **Paper 2** (Full Quaternion $P^2$ Attitude Control & Motor Dynamics) in **MuJoCo** under a **Piecewise Distance-Dependent Spatial Variable Wind Field**.

---

## 🚀 How to Run

### 1. Native MuJoCo Interactive Viewer
```powershell
py -3.12 variable_wind/mujoco/native_viewer.py
```

### 2. Matplotlib-based Interactive 3D Window (Play / Pause / Scrubber / HUD)
```powershell
py -3.12 variable_wind/mujoco/matplotlib_3d_window.py
```

### 3. Run Batch Simulation & Generate Telemetry Plots
```powershell
py -3.12 variable_wind/mujoco/run_simulation.py
```

### 4. Run Unit Tests
```powershell
py -3.12 -m unittest variable_wind/mujoco/tests/test_mujoco.py
```
