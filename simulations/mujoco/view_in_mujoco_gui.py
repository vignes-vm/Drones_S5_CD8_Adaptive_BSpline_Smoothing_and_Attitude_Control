"""
Native MuJoCo GUI Visualizer
Executes Paper 2 Adaptive B-Spline Path Planning & Paper 1 Quaternion Attitude Control live
in the 3D interactive MuJoCo viewer window.
"""

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from src.obstacle_planner import generate_obstacle_free_waypoints, is_point_in_obstacle, DEFAULT_OBSTACLES
from src.bspline_planner import fit_enhanced_adaptive_bspline, fit_classical_bspline
from src.trajectory_tracker import CascadedQuadrotorTracker
from src.time_parameterization import build_speed_limited_profile
from src.quaternion_math import quat_normalize, quat_to_rot_matrix


def simulate_flown_path(model, pos_bspline, scale, v_max, a_max):
    """
    Runs the actual closed-loop controller (no viewer, no real-time throttling)
    over the candidate path and returns the ACTUAL flown positions (unscaled).
    A path can be perfectly clear of every obstacle as a static reference curve
    and still get flown through/near an obstacle, because PID tracking error
    makes the real trajectory deviate from that reference -- so clearance has
    to be checked against what the drone actually flies, not just the plan.
    """
    data = mujoco.MjData(model)
    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81, dt=model.opt.timestep)
    scaled_pos = pos_bspline * scale
    init_pos = scaled_pos[0].copy()
    init_pos[2] = max(init_pos[2], 0.5)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = init_pos
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    params = np.linspace(0.0, 1.0, len(pos_bspline))
    from src.bspline_planner import BSplineEvaluator
    evaluator = BSplineEvaluator(pos_bspline, params=params)
    eval_at_time, T_flight = build_speed_limited_profile(evaluator, scale, v_max, a_max)

    n_steps = int(T_flight / model.opt.timestep)
    flown_unscaled = np.empty((n_steps, 3))
    for i in range(n_steps):
        t = data.time
        p_ref, v_ref, a_ref = eval_at_time(t)
        p_ref[2] = max(p_ref[2], 0.5)
        state = {
            'pos': np.array(data.qpos[0:3], dtype=np.float64),
            'vel': np.array(data.qvel[0:3], dtype=np.float64),
            'quat': quat_normalize(np.array(data.qpos[3:7], dtype=np.float64)),
            'omega': np.array(data.qvel[3:6], dtype=np.float64),
            'time': float(t),
        }
        tau_body, thrust_force = tracker.compute_control(p_ref, v_ref, a_ref, state)
        # freejoint: qfrc_applied[3:6] is BODY frame, [0:3] is WORLD frame.
        data.qfrc_applied[3:6] = tau_body
        R_curr = quat_to_rot_matrix(state['quat'])
        data.qfrc_applied[0:3] = R_curr[:, 2] * thrust_force
        mujoco.mj_step(model, data)
        flown_unscaled[i] = np.array(data.qpos[0:3]) / scale

    return flown_unscaled


def plan_clear_waypoints(start_pt, goal_pt, obstacles=None, clearance=90.0, max_iters=6):
    """
    generate_obstacle_free_waypoints only nudges the discrete waypoints clear of
    obstacles; the cubic spline threaded *through* those waypoints can still cut
    back into an obstacle between them (verified: the unpatched path clips 3 of
    the 4 obstacles here). This iteratively densely-samples the fitted curve,
    and for every obstacle penetration inserts a new waypoint pushed clear (up
    and outward) of the offending obstacle, then re-fits, until the sampled
    curve clears every obstacle by `clearance` or `max_iters` is hit. Uses the
    cheap classical (non-GA) fit for this search loop -- the expensive adaptive
    GA fit is only run once, on the final clear waypoint set, by the caller.
    """
    if obstacles is None:
        obstacles = DEFAULT_OBSTACLES

    wps = generate_obstacle_free_waypoints(start_pt, goal_pt, obstacles=obstacles)
    direction = goal_pt - start_pt

    for _ in range(max_iters):
        positions, _, _, _ = fit_classical_bspline(wps, num_samples=500)
        violations = [
            i for i, p in enumerate(positions)
            if any(is_point_in_obstacle(p, obs, buffer_margin=clearance) for obs in obstacles)
        ]
        if not violations:
            break

        # Group contiguous violating samples into runs; patch each with one new waypoint.
        runs = [[violations[0]]]
        for idx in violations[1:]:
            if idx == runs[-1][-1] + 1:
                runs[-1].append(idx)
            else:
                runs.append([idx])

        new_pts = []
        for run in runs:
            p = positions[run[len(run) // 2]].copy()
            for obs in obstacles:
                if not is_point_in_obstacle(p, obs, buffer_margin=clearance):
                    continue
                if obs['type'] == 'box':
                    cx = 0.5 * (obs['bounds'][0] + obs['bounds'][1])
                    cy = 0.5 * (obs['bounds'][2] + obs['bounds'][3])
                    top = obs['bounds'][5]
                else:
                    cx, cy = obs['center']
                    top = obs['height']
                dir_xy = p[:2] - np.array([cx, cy])
                n = np.linalg.norm(dir_xy)
                dir_xy = dir_xy / n if n > 1e-6 else np.array([1.0, 0.0])
                p[0:2] += dir_xy * clearance
                p[2] = top + clearance
                break
            new_pts.append(p)

        wps = np.vstack([wps, np.array(new_pts)])
        # Re-sort along the start->goal direction so the spline still threads
        # the waypoints in path order after insertion.
        order = np.argsort(wps @ direction)
        wps = wps[order]

    return wps


def _push_point_clear(p, obs, push_dist):
    """Returns p pushed up-and-outward clear of obs by push_dist (unscaled units)."""
    p = p.copy()
    if obs['type'] == 'box':
        cx = 0.5 * (obs['bounds'][0] + obs['bounds'][1])
        cy = 0.5 * (obs['bounds'][2] + obs['bounds'][3])
        top = obs['bounds'][5]
    else:
        cx, cy = obs['center']
        top = obs['height']
    dir_xy = p[:2] - np.array([cx, cy])
    n = np.linalg.norm(dir_xy)
    dir_xy = dir_xy / n if n > 1e-6 else np.array([1.0, 0.0])
    p[0:2] += dir_xy * push_dist
    p[2] = top + push_dist
    return p


def plan_and_verify_clear_path(model, start_pt, goal_pt, scale, v_max, a_max,
                                obstacles=None, max_iters=12, safety_margin=150.0):
    """
    A path clear as a static reference curve can still get physically flown
    close to (or into) an obstacle once real PID tracking error is added --
    confirmed: the drone grazed obs_building2 despite its planned reference
    path having zero violations. This closes the loop: after each candidate
    path is planned, it actually simulates the closed-loop controller flying
    it (headless, no viewer) and checks the REAL flown trajectory for
    obstacle clearance. Any obstacle the flown path gets within
    `safety_margin` of gets an extra waypoint patched in at the point of
    closest approach and the path is re-planned, until the flown trajectory
    clears every obstacle by the margin (or max_iters is hit).
    """
    if obstacles is None:
        obstacles = DEFAULT_OBSTACLES

    wps = plan_clear_waypoints(start_pt, goal_pt, obstacles=obstacles)
    direction = goal_pt - start_pt

    def margin_at(p, obs):
        """Signed clearance of p to obs inflated by safety_margin (negative = violating)."""
        if obs['type'] == 'box':
            b = obs['bounds']
            return min(
                p[0] - (b[0] - safety_margin), (b[1] + safety_margin) - p[0],
                p[1] - (b[2] - safety_margin), (b[3] + safety_margin) - p[1],
                p[2] - (b[4] - safety_margin), (b[5] + safety_margin) - p[2],
            )
        cx, cy = obs['center']
        return (obs['radius'] + safety_margin) - np.hypot(p[0] - cx, p[1] - cy)

    for _ in range(max_iters):
        pos, _, _, _ = fit_classical_bspline(wps, num_samples=400)
        flown = simulate_flown_path(model, pos, scale, v_max, a_max)

        # Patch every currently-violated obstacle in this round (not just the
        # single worst one) so unrelated violations converge in parallel
        # instead of needing one iteration each.
        worst_per_obs = {}  # id(obs) -> (margin, point)
        for p in flown:
            for oi, obs in enumerate(obstacles):
                if is_point_in_obstacle(p, obs, buffer_margin=safety_margin):
                    m = margin_at(p, obs)
                    if oi not in worst_per_obs or m < worst_per_obs[oi][0]:
                        worst_per_obs[oi] = (m, p)

        if not worst_per_obs:
            break

        new_pts = [
            _push_point_clear(pt, obstacles[oi], safety_margin + 60.0)
            for oi, (m, pt) in worst_per_obs.items()
        ]
        wps = np.vstack([wps, np.array(new_pts)])
        wps = wps[np.argsort(wps @ direction)]

    return wps


def launch_flight_gui():
    print("=========================================================================")
    print("   MUJOCO 3D FLIGHT VISUALIZER: ADAPTIVE B-SPLINE & QUATERNION CONTROL   ")
    print("=========================================================================")
    print(" Generating Paper 2 Adaptive B-Spline Trajectory...")

    start_pt = np.array([0.0, 0.0, 50.0])
    goal_pt = np.array([2000.0, 1800.0, 200.0])
    scale = 0.01

    xml_path = "models/airspace_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    raw_wps = plan_clear_waypoints(start_pt, goal_pt)
    pos_bspline, _, _, evaluator = fit_enhanced_adaptive_bspline(raw_wps, num_samples=600)
    print(f" Generated {len(raw_wps)} clear waypoints.")

    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81, dt=model.opt.timestep)

    scaled_pos = pos_bspline * scale
    init_pos = scaled_pos[0].copy()
    init_pos[2] = max(init_pos[2], 0.5)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = init_pos
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    # Speed/accel bounds kept comfortably inside the position controller's
    # tilt-safe acceleration limit (+/-2 m/s^2 while in transit) so it always
    # has margin left to correct tracking error on top of the feedforward.
    eval_at_time, T_flight = build_speed_limited_profile(
        evaluator, scale, v_max=2.0, a_max=1.2
    )
    HOLD_TIME = 4.0  # seconds to hover at the goal before looping back to start
    goal_scaled = evaluator.eval_pos(1.0) * scale

    print(f" Flight duration: {T_flight:.1f}s (speed-limited to <= 2.0 m/s)")

    print("\n[Launching Native MuJoCo 3D Visualizer Window]")
    print(" - Use Left-Click + Drag to rotate 3D camera around quadrotor.")
    print(" - Use Right-Click + Drag to pan camera.")
    print(" - Use Scroll Wheel to zoom in/out.")
    print(" - Close the 3D window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # By default the free camera starts at lookat=(0,0,0) with a small fixed
        # distance -- which is exactly where the blue start marker sits -- so it
        # opens zoomed into the start marker instead of showing the airspace.
        # Frame the whole scene (start -> goal + obstacles) instead.
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = model.stat.extent * 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        step_idx = 0
        while viewer.is_running():
            step_start = time.time()

            t = data.time
            if t > T_flight + HOLD_TIME:
                # Reset flight loop after holding at the goal for a bit
                mujoco.mj_resetData(model, data)
                data.qpos[0:3] = init_pos
                data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
                tracker.reset()
                mujoco.mj_forward(model, data)
                t = 0.0

            if t <= T_flight:
                p_ref, v_ref, a_ref = eval_at_time(t)
            else:
                # Arrived: hold station at the goal (zero feedforward) for HOLD_TIME.
                p_ref, v_ref, a_ref = goal_scaled.copy(), np.zeros(3), np.zeros(3)

            p_ref[2] = max(p_ref[2], 0.5)

            state = {
                'pos': np.array(data.qpos[0:3], dtype=np.float64),
                'vel': np.array(data.qvel[0:3], dtype=np.float64),
                'quat': quat_normalize(np.array(data.qpos[3:7], dtype=np.float64)),
                'omega': np.array(data.qvel[3:6], dtype=np.float64),
                'time': float(t)
            }

            tau_body, thrust_force = tracker.compute_control(p_ref, v_ref, a_ref, state)

            # freejoint: qfrc_applied[3:6] is BODY frame, [0:3] is WORLD frame.
            data.qfrc_applied[3:6] = tau_body
            R_curr = quat_to_rot_matrix(state['quat'])
            data.qfrc_applied[0:3] = R_curr[:, 2] * thrust_force

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if model.opt.timestep > elapsed:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == '__main__':
    launch_flight_gui()
