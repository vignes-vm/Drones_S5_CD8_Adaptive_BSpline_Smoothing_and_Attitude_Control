# Project Proposal (FRP)

## Project Title

**Adaptive B-Spline Smoothing and Attitude Control of a Quadrotor UAV**

## Team Members

| Name | Roll Number | Email |
|---|---|---|
| Aparna A | CB.SC.U4AIE24206 | cb.sc.u4aie24206@cb.students.amrita.edu |
| G Prajwal Priyadarshan | CB.SC.U4AIE24214 | cb.sc.u4aie24214@cb.students.amrita.edu |
| Kabilan K | CB.SC.U4AIE24224 | cb.sc.u4aie24224@cb.students.amrita.edu |
| Kesav Satya Sai Nimmagadda | CB.SC.U4AIE24236 | cb.sc.u4aie24236@cb.students.amrita.edu |
| Vignes V M | CB.SC.U4AIE24359 | cb.sc.u4aie24359@cb.students.amrita.edu |

## Faculty-Directed Requirements

Following review, the faculty advisor directed two additions to the scope below, which are now binding requirements rather than stretch goals:

1. **Cross-simulator validation.** The merged pipeline must be simulated and validated in **all four simulation environments** — Python, MATLAB/Simulink, MuJoCo, and Gazebo — rather than treating any one of them as the primary result and the others as optional ports.
2. **Spatially-varying wind field.** The wind disturbance model must vary **wind velocity at each point in space** the quadrotor passes through (a position-dependent field `v_wind(x, y, z, t)`), not a single wind vector applied uniformly across the whole airspace at a given instant. Every simulator's wind/disturbance model must sample the field at the vehicle's current position at each integration step.

## Proposed Project Work

This project builds an integrated UAV navigation stack by recreating and merging two published methods into a single obstacle-aware pipeline. A 3D A\* planner finds a collision-free route through a cluttered environment; a genetic algorithm reshapes that route into a curvature-bounded cubic B-Spline via adaptive knot placement; and a singularity-free quaternion P² (proportional–proportional) attitude controller tracks the spline's local tangent in real time, so the vehicle's commanded attitude follows the path it is flying. The path parameter is gated by the attitude-tracking error, coupling guidance and control into one closed loop rather than treating them as independent stages.

The work is based on two base papers — *"An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"* (3D A\* baseline path → GA-driven adaptive knot placement → curvature-constrained cubic B-Spline) and Fresk & Nikolakopoulos' *"Full Quaternion Based Attitude Control for a Quadrotor"* (ECC 2013) (singularity-free quaternion P² attitude/rate control law) — with emphasis on recreating each paper's core algorithm faithfully before merging them. The system is implemented and validated across Python, MATLAB/Simulink, MuJoCo, and Gazebo simulation environments.

## Objectives

- Build a 3D obstacle environment and run 3D A\* search on a 26-connected voxel grid to find a collision-free baseline path.
- Fit an initial cubic B-Spline to the pruned waypoints and optimize its interior knot vector with a genetic algorithm to minimize path length and curvature violations.
- Apply a curvature-aware local refinement pass to remove residual curvature violations left after knot optimization.
- Model UAV rotational dynamics and implement quaternion-based attitude representation to avoid Euler-angle gimbal lock.
- Convert the smoothed spline's local tangent into a reference quaternion and design a quaternion P² controller to track it.
- Gate the path parameter's advancement on attitude-tracking error convergence, merging guidance and control into one closed loop.
- Validate both the path-smoothing algorithm and the attitude controller independently against each base paper's reported results, then validate the merged pipeline.
- Simulate and cross-validate the full merged pipeline in **all four** simulation environments — Python, MATLAB/Simulink, MuJoCo, and Gazebo — as directed by the faculty advisor, rather than in a single primary environment.
- Model wind as a **spatially-varying field** `v_wind(x, y, z, t)` so disturbance magnitude and direction change with the quadrotor's position along the flight path, and inject this position-sampled disturbance into the rigid-body dynamics consistently across all four simulators.

## Methodology

1. Build a 3D obstacle environment (cylinders and cuboids with safety buffers) and run 3D A\* search to find a collision-free raw path.
2. Prune the raw path to a sparse set of waypoints and parameterize them by chord length.
3. Fit an initial cubic B-Spline via least-squares control-point solve.
4. Run a genetic algorithm over the interior knot vector to minimize a weighted objective (data fidelity, curvature penalty, path length, knot regularization).
5. Run a local curvature-aware refinement pass that directly nudges control points wherever curvature still exceeds the limit.
6. At each point along the resulting spline, convert the local tangent into a reference quaternion.
7. Feed the reference quaternion into a quaternion P² controller, which computes control torque from quaternion attitude error and angular-velocity feedback.
8. Integrate rigid-body rotational dynamics (RK4) to update the simulated quadrotor's attitude.
9. Advance the path parameter only once attitude-tracking error converges below a threshold; otherwise hold position and keep correcting.
10. At each integration step, sample a spatially-varying wind field at the vehicle's current 3D position (steady component + position-dependent gust/turbulence terms) and apply the resulting aerodynamic drag force and disturbance torque to the rigid-body dynamics, so wind differs from one point on the path to another instead of being uniform across the airspace.
11. Re-implement steps 1–10 independently in each of the four simulation environments (Python, MATLAB/Simulink, MuJoCo, Gazebo), keeping physical parameters (mass, inertia, gains, torque limits) consistent across them so results are comparable.
12. Log attitude, torque, quaternion error, path progress, and wind-rejection performance in each environment, and render comparison plots/animations across all four.

## Expected Outcomes

- A working 3D A\* + GA-adaptive B-Spline path smoother that reduces path length and peak curvature relative to baseline, uniform, and classical B-Spline smoothing.
- A working singularity-free quaternion P² attitude controller with stable tracking of step, sinusoidal, and full-flip attitude references.
- A merged path-following pipeline where the attitude controller tracks the smoothed spline's tangent with bounded torque, gated by attitude-error convergence.
- Cross-validated results across **all four** simulation environments — Python, MATLAB/Simulink, MuJoCo, and Gazebo — with consistent physical parameters so runs are comparable.
- A spatially-varying wind disturbance model, implemented consistently in each simulator, that shows the controller rejecting wind whose magnitude/direction changes across the flight path rather than a single uniform disturbance vector.

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Core simulation, A\* planning, GA knot optimization, controller implementation, and one of the four required cross-validation environments |
| MATLAB/Simulink | Parallel UAV modelling, controller simulation, 3D animation, and one of the four required cross-validation environments |
| MuJoCo | Physics-based UAV simulation, obstacle-environment rendering, and one of the four required cross-validation environments |
| Gazebo | SDF world generation, independent 6-DOF flight-dynamics validation, and one of the four required cross-validation environments |
| GitHub | Project documentation and version control |

All four simulators must run the merged pipeline under the same spatially-varying wind field so wind-rejection performance is directly comparable across environments, per the faculty-directed requirements above.

## Future Scope

Future extensions include collision-aware spline optimization (folding obstacle avoidance directly into the GA objective rather than the upstream A\* buffer), tighter gain tuning for fast periodic attitude references, richer wind-field models (e.g. Dryden/Von Kármán turbulence spectra sampled per position rather than a deterministic sinusoidal gust), multi-agent path planning, and real-time onboard re-planning.
