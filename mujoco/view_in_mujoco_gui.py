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

from src.obstacle_planner import generate_obstacle_free_waypoints
from src.bspline_planner import fit_enhanced_adaptive_bspline
from src.trajectory_tracker import CascadedQuadrotorTracker

def launch_flight_gui():
    print("=========================================================================")
    print("   MUJOCO 3D FLIGHT VISUALIZER: ADAPTIVE B-SPLINE & QUATERNION CONTROL   ")
    print("=========================================================================")
    print(" Generating Paper 2 Adaptive B-Spline Trajectory...")

    start_pt = np.array([0.0, 0.0, 50.0])
    goal_pt = np.array([2000.0, 1800.0, 200.0])

    raw_wps = generate_obstacle_free_waypoints(start_pt, goal_pt)
    pos_bspline, _, _, evaluator = fit_enhanced_adaptive_bspline(raw_wps, num_samples=600)

    xml_path = "models/airspace_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81, dt=model.opt.timestep)

    scale = 0.01
    scaled_pos = pos_bspline * scale
    init_pos = scaled_pos[0].copy()
    init_pos[2] = max(init_pos[2], 0.5)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = init_pos
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    T_flight = 15.0

    print("\n[Launching Native MuJoCo 3D Visualizer Window]")
    print(" - Use Left-Click + Drag to rotate 3D camera around quadrotor.")
    print(" - Use Right-Click + Drag to pan camera.")
    print(" - Use Scroll Wheel to zoom in/out.")
    print(" - Close the 3D window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_idx = 0
        while viewer.is_running():
            step_start = time.time()

            t = data.time
            if t > T_flight:
                # Reset flight loop when target is reached
                mujoco.mj_resetData(model, data)
                data.qpos[0:3] = init_pos
                data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
                tracker.reset()
                mujoco.mj_forward(model, data)
                t = 0.0

            tau_frac = np.clip(t / T_flight, 0.0, 1.0)
            s = 10.0 * (tau_frac**3) - 15.0 * (tau_frac**4) + 6.0 * (tau_frac**5)
            ds = (30.0 * (tau_frac**2) - 60.0 * (tau_frac**3) + 30.0 * (tau_frac**4)) / T_flight
            dds = (60.0 * tau_frac - 180.0 * (tau_frac**2) + 120.0 * (tau_frac**3)) / (T_flight**2)

            p_ref = evaluator.eval_pos(s) * scale
            v_ref = evaluator.eval_vel(s) * ds * scale
            a_ref = (evaluator.eval_acc(s) * (ds**2) + evaluator.eval_vel(s) * dds) * scale

            p_ref[2] = max(p_ref[2], 0.5)

            state = {
                'pos': np.array(data.qpos[0:3], dtype=np.float64),
                'vel': np.array(data.qvel[0:3], dtype=np.float64),
                'quat': quat_normalize(np.array(data.qpos[3:7], dtype=np.float64)),
                'omega': np.array(data.qvel[3:6], dtype=np.float64),
                'time': float(t)
            }

            tau_world, thrust_force = tracker.compute_control(p_ref, v_ref, a_ref, state)

            data.qfrc_applied[3:6] = tau_world
            R_curr = quat_to_rot_matrix(state['quat'])
            data.qfrc_applied[0:3] = R_curr[:, 2] * thrust_force

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if model.opt.timestep > elapsed:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == '__main__':
    launch_flight_gui()
