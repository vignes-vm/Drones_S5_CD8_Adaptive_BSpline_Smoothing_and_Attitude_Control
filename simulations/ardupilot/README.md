# ArduPilot SITL Integration

This package connects the **Adaptive B-Spline Path Smoother** (Paper 1) and **Quaternion Guidance & Attitude Control** (Paper 2) to **ArduPilot SITL (ArduCopter)** over MAVLink using `pymavlink`.

---

## 1. Prerequisites

Ensure `pymavlink` and project dependencies are installed:
```bash
pip install pymavlink numpy matplotlib pygad
```

---

## 2. Launching ArduPilot SITL

### Option A: Using `sim_vehicle.py` (WSL / Linux / Cygwin)
```bash
# Navigate to ArduPilot source tree
cd ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter --console --map
```
This starts SITL listening on UDP `127.0.0.1:14550` (or `14551` via MAVProxy).

### Option B: Pre-built Cygwin / Windows SITL Executable
```bash
# Run ArduCopter SITL executable directly
ArduCopter.elf -S --model quad --speedup 1 --slave 0
```

---

## 3. Running the Companion Bridge

### Dry-Run Test Mode (No SITL Required)
Test the path planner, GA smoother, trajectory converter, and mock guidance loop directly:
```bash
python ardupilot/sitl_bridge.py --dry-run
```

### Live SITL Mode (Attitude Target Streaming)
Stream B-Spline tangent quaternions \(q_{\text{ref}}\) to ArduPilot via `SET_ATTITUDE_TARGET` (MAVLink msg #82):
```bash
python ardupilot/sitl_bridge.py --connection udpin:127.0.0.1:14550 --mode attitude
```

### Live SITL Mode (Kinodynamic Position Target Streaming)
Stream 3D position targets \(p(u)\) to ArduPilot via `SET_POSITION_TARGET_LOCAL_NED` (MAVLink msg #84):
```bash
python ardupilot/sitl_bridge.py --connection udpin:127.0.0.1:14550 --mode position
```

---

## 4. Pipeline Workflow

1. **Path Planning:** 3D A* generates collision-free waypoints through the obstacle field.
2. **Spline Smoothing:** `AdaptiveBSplineSmoother` runs 3-stage optimization:
   - Stage 1: Least-squares initialization with chord-length parameterization.
   - Stage 2: GA interior knot placement to minimize path length, curvature, and knot variance.
   - Stage 3: Local control point refinement wherever curvature \(\kappa(u) > \kappa_{\text{max}}\).
3. **Trajectory Conversion:** Tangent vector \(C'(u)\) is mapped to reference unit quaternion \(q_{\text{ref}}(u)\).
4. **Attitude Error-Gated Execution:**
   - Companion reads MAVLink telemetry (`ATTITUDE` / `LOCAL_POSITION_NED`).
   - Computes quaternion tracking error \(q_{\text{err}} = q_{\text{ref}} \otimes q_{\text{meas}}^{-1}\).
   - Path parameter \(u\) advances **only when** \(\|q_{\text{err}, 1:3}\| < \varepsilon\).
5. **Output Generation:** Saves 3D flight paths and tracking error plots to `ardupilot/results/`.
