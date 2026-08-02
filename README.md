# Adaptive B-Spline Smoothing and Altitude Control

An integrated UAV navigation stack that fuses **GA-optimised adaptive B-Spline path smoothing** with a **singularity-free quaternion attitude controller**, built by recreating and merging two published methods into a single obstacle-aware pipeline: a 3D A\* planner finds a collision-free route, a genetic algorithm reshapes it into a curvature-bounded B-Spline, and a quaternion P² controller tracks that spline's tangent in real time.

Both a **Python** implementation and a **MATLAB** implementation (with 3D animations) are included and kept in sync.

| Paper | Contribution used here |
|---|---|
| *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* | 3D A\* baseline path → GA-driven adaptive knot placement → curvature-constrained cubic B-Spline |
| Fresk & Nikolakopoulos, *"Full Quaternion Based Attitude Control for a Quadrotor"* (ECC 2013) | Singularity-free quaternion P² (proportional–proportional) attitude/rate control law |

---

## 1. System Overview

```mermaid
flowchart LR
    subgraph Planning["Path Planning"]
        A["3D Obstacle\nEnvironment"] --> B["A* Search\n(26-connected grid)"]
        B --> C["Douglas-Peucker\nPruning"]
        C --> D["Sparse\nWaypoints"]
    end

    subgraph Smoothing["Adaptive Smoothing"]
        D --> E["Chord-Length\nParameterization"]
        E --> F["GA Knot\nOptimization"]
        F --> G["Cubic B-Spline\nInterpolation"]
    end

    subgraph Control["Attitude Control"]
        G --> H["Tangent Vector\nat u"]
        H --> I["Tangent to\nQuaternion Reference"]
        I --> J["Quaternion P^2\nController"]
        J --> K["Rigid-Body\nDynamics (RK4)"]
        K -->|"attitude error"| L{"Error <\nthreshold?"}
        L -->|"yes: advance u"| H
        L -->|"no: hold u"| J
    end

    G --> H
    K --> M["Attitude & Torque\nLogs / Animation"]
```

The path parameter `u` (position along the spline) only advances once the attitude tracking error has converged — the drone doesn't outrun its own attitude loop.

---

## 2. Path Planning — 3D A\*

The environment (`obstacle_env`) is a `2000 × 2000 × 500 m` volume populated with cylinders (towers) and cuboids (buildings), each inflated by a safety **buffer**. A point `p` collides with an obstacle if it falls inside the buffered cylinder/cuboid volume.

**A\* cost model**, on a 26-connected voxel grid of resolution `r`:

```math
f(n) = g(n) + h(n)
```

```math
g(n) = g(\text{parent}) + \lVert \Delta \rVert \cdot r, \qquad \Delta \in \{-1,0,1\}^3 \setminus \{0\}
```

```math
h(n) = \lVert (n - n_{goal}) \cdot r \rVert_2 \quad \text{(Euclidean, admissible on a 3D grid)}
```

The raw voxel path is then **pruned with Douglas–Peucker** (tolerance `0.6r`) to produce a sparse set of waypoints suitable for spline fitting — collapsing long straight runs while keeping corners.

```mermaid
flowchart TD
    S([Start]) --> Open["Push start into open set,\nf = h(start)"]
    Open --> Pick["Pop node with lowest f-score"]
    Pick --> Goal{Is goal?}
    Goal -- yes --> Reconstruct["Backtrack cameFrom\nchain -> raw path"]
    Goal -- no --> Expand["Expand 26 neighbors"]
    Expand --> Collide{In obstacle\n+ buffer?}
    Collide -- yes --> Pick
    Collide -- no --> Relax["tentative_g = g(current) + step_cost\nRelax if better than known g"]
    Relax --> Pick
    Reconstruct --> DP["Douglas-Peucker\nsimplification, eps=0.6r"]
    DP --> W([Sparse waypoints])
```

---

## 3. Adaptive B-Spline Smoothing

### 3.1 Parameterization

Waypoints `{P_0, ..., P_n}` are assigned parameter values `u_i` by **chord-length parameterization**:

```math
u_0 = 0, \qquad u_i = u_{i-1} + \frac{\lVert P_i - P_{i-1} \rVert}{\sum_{k=1}^{n}\lVert P_k - P_{k-1} \rVert}, \qquad u_n = 1
```

### 3.2 Knot vector (de Boor / Piegl & Tiller averaging)

For a degree-`p` spline (cubic, `p = 3`) through `n+1` points, the interior knots are placed by **averaging** the parameter values (Piegl & Tiller, Eq. 9.8) — this guarantees the **Schoenberg–Whitney condition**, keeping the collocation matrix well-conditioned:

```math
U = \{\underbrace{0,\dots,0}_{p+1},\ u_{p+1}^{*},\dots,u_{n}^{*},\ \underbrace{1,\dots,1}_{p+1}\},
\qquad u_{j+p}^{*} = \frac{1}{p}\sum_{i=j}^{j+p-1} u_i
```

### 3.3 Basis functions — Cox–de Boor recursion

```math
N_{i,0}(u) = \begin{cases} 1 & u_i \le u < u_{i+1} \\ 0 & \text{otherwise} \end{cases}
```

```math
N_{i,p}(u) = \frac{u - u_i}{u_{i+p}-u_i}\,N_{i,p-1}(u) \;+\; \frac{u_{i+p+1}-u}{u_{i+p+1}-u_{i+1}}\,N_{i+1,p-1}(u)
```

### 3.4 Control-point solve (global interpolation)

Control points `Q_j` are recovered by solving the **linear interpolation system** built from the basis matrix `N_{i,p}(u_k)` so the curve passes exactly through every waypoint:

```math
C(u_k) = \sum_{j=0}^{n} N_{j,p}(u_k)\, Q_j = P_k, \qquad k = 0,\dots,n
```

```math
\mathbf{N}\,\mathbf{Q} = \mathbf{P} \quad\Longrightarrow\quad \mathbf{Q} = \mathbf{N}^{-1}\mathbf{P}
```

### 3.5 Curvature

Curvature is evaluated numerically from the first and second derivatives of `C(u)` (central differences, step `h`):

```math
\kappa(u) = \frac{\lVert C'(u) \times C''(u) \rVert}{\lVert C'(u) \rVert^{3}}
```

### 3.6 Adaptive knot placement via Genetic Algorithm

The classical knot-averaging vector is used as a baseline; the GA searches only the **interior knots**, bounded within `±0.15` of their baseline position (keeps them sorted ⇒ Schoenberg–Whitney holds by construction), minimizing:

```math
\text{cost}(U) = L(U) + 1000 \cdot \max\!\big(0,\ \kappa_{max}(U) - \kappa_{limit}\big)^2
```

where `L(U)` is arc length (numerically integrated over 500 samples) and `κ_max(U)` is the peak curvature of the resulting spline. The quadratic penalty makes any knot arrangement that violates the curvature limit `κ_limit` (default `0.5 m⁻¹`) strictly worse than a compliant but slightly longer path.

```mermaid
flowchart LR
    B0["Baseline knots\n(averaging)"] --> Pop["Initial GA population\n(perturbed interior knots)"]
    Pop --> Fit["Fit B-Spline per\ncandidate knot vector"]
    Fit --> Eval["cost = length + curvature penalty"]
    Eval --> Sel{Converged /\nmax generations?}
    Sel -- no --> GA["Selection, crossover 0.8,\nmutation"] --> Pop
    Sel -- yes --> Best["Best interior knots"]
    Best --> Final["Final adaptive\nB-Spline path"]
```

### Recreated results (Table 2 of Paper 1)

| Method | Length (m) | Max κ (m⁻¹) | Straightness Ratio |
|---|---:|---:|---:|
| Baseline Polyline | 2984.6 | 0.65 | 1.107 |
| Uniform B-Spline | 2840.8 | 0.56 | 1.053 |
| Classical B-Spline (chord-length) | 2803.2 | 0.49 | 1.039 |
| **Adaptive B-Spline (GA)** | **2731.4** | **0.38** | **1.013** |

*(reference values from the paper; `run_paper1.m` / `paper1_recreation.py` reproduce these on the repo's own A\*-derived waypoint set.)*

---

## 4. Attitude Control — Quaternion P² Controller

### 4.1 Quaternion kinematics

With `q = [q_0, q_1, q_2, q_3]^T` the unit attitude quaternion and `ω` the body angular rate:

```math
\dot{q} = \tfrac{1}{2}\, q \otimes \begin{bmatrix}0\\ \omega\end{bmatrix}
```

Quaternion (Hamilton) product `p ⊗ q`:

```math
p \otimes q = \begin{bmatrix}
p_0 q_0 - p_1 q_1 - p_2 q_2 - p_3 q_3\\
p_0 q_1 + p_1 q_0 + p_2 q_3 - p_3 q_2\\
p_0 q_2 - p_1 q_3 + p_2 q_0 + p_3 q_1\\
p_0 q_3 + p_1 q_2 - p_2 q_1 + p_3 q_0
\end{bmatrix}
```

### 4.2 Rigid-body rotational dynamics

```math
I\dot{\omega} = \tau - \omega \times (I\omega)
```

where `I = diag(I_xx, I_yy, I_zz)` is the body inertia tensor and `τ` the control torque. State is integrated with classical **RK4**, then the quaternion is renormalized every step.

### 4.3 Quaternion attitude error

```math
q_{err} = q_{ref} \otimes q_{meas}^{-1}, \qquad q_{meas}^{-1} = \bar{q}_{meas} = [q_0,-q_1,-q_2,-q_3]^T
```

The scalar part is forced positive (`q_err ← −q_err` if `q_{err,0} < 0`) to always take the **shortest rotational path**, avoiding the double-cover singularity of the quaternion group.

### 4.4 P² (proportional–proportional) control law

```math
\tau = -P_q\, \vec{q}_{err} - P_w\, \omega, \qquad \vec{q}_{err} = [q_{err,1}, q_{err,2}, q_{err,3}]^T
```

saturated element-wise to `±τ_max`. This is a cascade of an outer **attitude-error loop** (gain `P_q`) directly onto an inner **angular-rate damping loop** (gain `P_w`) — no separate rate-loop integrator, no Euler-angle singularities.

```mermaid
flowchart LR
    Qref["q_ref\n(desired attitude)"] --> Err["Quaternion error\nq_err = q_ref ⊗ conj(q_meas)"]
    Qmeas["q_meas\n(estimated attitude)"] --> Err
    Err --> Pq["-Pq * q_err(1:3)"]
    Omega["ω (angular rate)"] --> Pw["-Pw * ω"]
    Pq --> Sum(("+")) --> Sat["Saturate\n±tau_max"]
    Pw --> Sum
    Sat --> Tau["τ (torque)"]
    Tau --> RB["Rigid body:\nI·ω̇ = τ - ω × (Iω)"]
    RB --> RK4["RK4 integration\n+ quaternion renorm"]
    RK4 --> Qmeas
```

### 4.5 Stability note (gains vs. timestep)

The paper's exact gains (`P_q = 20`, `P_w = 4`) with this project's inertia (`I_xx = I_yy = 6.5×10⁻⁴`) place the fast closed-loop pole beyond `−6000 rad/s` (≈160 μs time constant). The control law is therefore **recomputed at the same rate as the physics integration** (`dt = 1×10⁻⁴ s`, i.e. 10 kHz) rather than held constant over a coarser control period — holding torque any longer lets `ω` build up before the controller can react, which was the actual source of divergence during development (not the RK4 integrator itself). Results are logged at a coarser, human-readable rate for plotting.

---

## 5. Merged Pipeline — Path-Following Attitude Control

The B-Spline's local tangent `C'(u)` is converted into a reference attitude by building an orthonormal frame (tangent, and two axes derived via a reference "up" vector) and reading off the quaternion of the resulting rotation matrix. The controller then drives the vehicle's actual quaternion toward that reference, and the path parameter `u` is advanced only while tracking error stays under a threshold — coupling **guidance** and **control** into one gated loop (see the flowchart in §1).

```math
u_{k+1} = \begin{cases} \min(u_k + \Delta u,\ u_{hi}) & \lVert \vec{q}_{err}\rVert < \varepsilon \\ u_k & \text{otherwise} \end{cases}
```

---

## 6. Repository Layout

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

### Top-level scripts (Python)

| Script | What it does |
|---|---|
| `paper1_recreation.py` | Recreates Table 2 of Paper 1 — compares 4 path-smoothing methods on an A\*-generated baseline, averaged over 5 seeded trials |
| `paper2_recreation.py` | Recreates Figures 3, 5, 7 of Paper 2 — step, sinusoidal, and flip-maneuver responses of the quaternion controller in isolation |
| `test_paper2_controller.py` | Standalone validation of the P² controller (no path planning) against the paper's own scenarios |
| `main.py` | End-to-end pipeline: GA knot optimization → B-Spline path → merged attitude-tracking simulation, with full plot suite |
| `final_simulation.py` | Complete merged demo: A\* + obstacle-aware adaptive B-Spline smoothing feeding attitude references into the P² controller |
| `validation_tests.py` | Robustness check of the merged A\* + smoothing pipeline across multiple test scenarios |
| `generate_presentation_plots.py` | Reruns the full pipeline and renders polished figures to `results/presentation/` |

### MATLAB tree

```
part1_bspline/        bspline_engine.m, knot_ga.m, run_paper1.m
part2_controller/      quaternion_utils.m, test_step_response.m
part3_merged/          run_merged_simulation.m   (path -> attitude control)
part4_environment/     astar_3d.m, obstacle_env.m, run_part4_full.m, run_phase4_dense.m
```

`part4_environment` scripts additionally render 3D obstacle-avoidance animations, saved as `.mp4` under `matlab/uav_matlab/results/`.

---

## 7. Requirements

**Python 3.11+**

```bash
pip install numpy scipy matplotlib pygad
```

A `.venv` with these dependencies already installed is included in this repo for convenience.

**MATLAB** — Global Optimization Toolbox (for `ga`), no other toolboxes required.

---

## 8. Running

### Python

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

Run any `run_*.m` script from MATLAB with the repo on the path (each script calls `addpath(genpath(...))` itself):

```matlab
run_paper1              % part1_bspline  — Table 2 recreation
test_step_response      % part2_controller — Paper 2 Figure 3 recreation
run_merged_simulation   % part3_merged  — path + controller
run_part4_full          % part4_environment — A* + smoothing + animation
run_phase4_dense        % part4_environment — dense obstacle field variant
```

---

## 9. Key Implementation Notes

- **Controller gains**: Paper 2's exact gains (`Pq=20`, `Pw=4`) are discrete-time unstable at `dt=0.01` given this project's low rotational inertia (`Ixx=Iyy=6.5e-4`). `paper2_recreation.py`, `final_simulation.py`, and `test_paper2_controller.py` integrate the control law and physics internally at `dt=0.0005` (2 kHz) while logging at the paper's own `dt=0.01` resolution. `main.py`'s merged simulation instead uses a reduced, empirically stable gain pair (`Pq=0.2`, `Pw=0.1`) at `dt=0.01`.
- **Path progress gating**: in the merged simulation, the path parameter `u` only advances once the current attitude tracking error has converged below a threshold, so the drone doesn't outrun its own attitude controller.
- **Curvature constraint**: the GA knot optimizer (`path_planning/knot_ga.py`) searches interior knot placement to keep the B-Spline's maximum curvature under a configurable `kappa_max`, evaluated against path length and straightness ratio.
- **Numerical conditioning**: knot averaging (§3.2) is what keeps the basis matrix `N` well-conditioned; `bspline_engine` warns if `cond(N) > 1e8`, which typically means the knot vector and parameter values have fallen out of sync (Schoenberg–Whitney violated).

---

## 10. Results

Generated figures (path comparisons, curvature, attitude tracking, torques, quaternion error, angular velocity, path progress) are written to `results/`, with a curated set in `results/presentation/`. MATLAB animations of obstacle-avoidance runs are in `matlab/uav_matlab/results/`.

---

## 11. References

1. *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* — 3D A\* baseline + GA-driven adaptive B-Spline knot placement.
2. E. Fresk and G. Nikolakopoulos, *"Full Quaternion Based Attitude Control for a Quadrotor,"* European Control Conference (ECC), 2013.
3. L. Piegl and W. Tiller, *The NURBS Book*, 2nd ed., Springer, 1997 — global curve interpolation (Algorithm A9.1) and knot averaging (Eq. 9.8).
4. C. de Boor, *A Practical Guide to Splines*, Springer, 1978 — Cox–de Boor recursion and well-conditioned knot placement.
