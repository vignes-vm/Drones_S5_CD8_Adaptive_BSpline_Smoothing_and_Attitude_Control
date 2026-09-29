"""
Live Real-Time 3D Viewer for the Gazebo-Stack UAV Mission.

Runs the same pipeline as run_simulation.py (3D A* -> 3-Stage GA Adaptive B-Spline ->
closed-loop 6-DOF quaternion-controlled flight) but renders it live in an interactive
matplotlib 3D window instead of saving static telemetry plots -- the Gazebo-folder
equivalent of mujoco/view_in_mujoco_gui.py. No native Gazebo install is required: the
Python 6-DOF engine (GazeboFallbackEngine) is the only physics source, exactly as it is
for the non-realtime run_simulation.py path (see gazebo/README.md).

Each rendered frame advances several physics steps (500 Hz) and the frame rate is paced
to wall-clock time, so the flight plays out in the window at roughly the speed it would
actually take to fly. On reaching the goal the drone holds station briefly, then resets
to the start and flies the mission again.
"""
import os
import sys
import time
import argparse
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.path_smoother import AdaptiveBSplineSmoother

from gazebo.src.gazebo_env import GazeboEnvironment
from gazebo.src.quadrotor_gazebo import GazeboQuadrotor
from gazebo.src.trajectory_tracker import GazeboTrajectoryTracker
from gazebo.src.quaternion_math import QuaternionMath
from gazebo.src.wind_model import WindDisturbanceModel
from gazebo.src.path_utils import plan_safe_path, ArcLengthProfile


def parse_args():
    parser = argparse.ArgumentParser(description="Live real-time Gazebo-stack UAV flight viewer")
    parser.add_argument('--seed', type=int, default=42, help="Obstacle random seed")
    parser.add_argument('--speed', type=float, default=25.0, help="Cruise speed (m/s)")
    parser.add_argument('--fps', type=float, default=30.0, help="Target render frame rate (Hz)")
    parser.add_argument('--hold', type=float, default=3.0, help="Seconds to hold at the goal before looping back to start")
    parser.add_argument('--motor-lag', action='store_true', help="Enable first-order motor lag")
    parser.add_argument('--wind', action='store_true', help="Enable wind & gust disturbances")
    parser.add_argument('--wind-speed', type=float, nargs=3, default=(2.0, 1.0, 0.0),
                        metavar=('WX', 'WY', 'WZ'),
                        help="Steady wind velocity in world frame, m/s (default: 2.0 1.0 0.0)")
    parser.add_argument('--gust-amp', type=float, nargs=3, default=(1.5, 1.5, 0.5),
                        metavar=('GX', 'GY', 'GZ'),
                        help="Sinusoidal gust amplitude, m/s (default: 1.5 1.5 0.5)")
    parser.add_argument('--gust-freq', type=float, nargs=3, default=(0.3, 0.2, 0.5),
                        metavar=('FX', 'FY', 'FZ'),
                        help="Sinusoidal gust frequency, Hz (default: 0.3 0.2 0.5)")
    parser.add_argument('--drag-lin', type=float, default=0.01,
                        help="Quadratic linear drag coefficient, kg/m (default: 0.01)")
    parser.add_argument('--drag-rot', type=float, default=0.01,
                        help="Rotational aerodynamic damping coefficient, N*m*s/rad (default: 0.01)")
    return parser.parse_args()


def launch_live_viewer(seed=42, flight_speed=25.0, fps=30.0, hold_time=3.0,
                        motor_lag=False, enable_wind=False, wind_speed=(2.0, 1.0, 0.0),
                        gust_amp=(1.5, 1.5, 0.5), gust_freq=(0.3, 0.2, 0.5),
                        drag_lin=0.01, drag_rot=0.01):
    print("=" * 70)
    print("   LIVE GAZEBO-STACK 3D FLIGHT VIEWER: ADAPTIVE B-SPLINE & QUATERNION CONTROL   ")
    print("=" * 70)

    dt_sim = 0.002  # 500 Hz physics rate

    print("\n[1/3] Constructing 3D obstacle airspace...")
    env_3d = Environment3D()
    env_3d.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)
    print(f"  Start: {env_3d.start} | Goal: {env_3d.goal} | Obstacles: {len(env_3d.obstacles)} (seed={seed})")

    print("\n[2/3] Planning 3D A* path and fitting collision-checked adaptive B-spline (GA)...")
    astar = AStarPlanner3D(env_3d, grid_resolution=50)
    baseline_wp, spline, report = plan_safe_path(
        env_3d, astar,
        lambda: AdaptiveBSplineSmoother(n_inner_knots=6, n_generations=20,
                                        population_size=15, kappa_max=0.5),
        min_altitude=15.0
    )
    spline_pts = spline.sample_path(300)
    print(f"  Smoothed path length: {spline.path_length():.1f} m | obstacle hits: {report['hits']}")

    print("\n[3/3] Initializing 6-DOF physics engine and controller (Python, no native Gazebo)...")
    profile = ArcLengthProfile(spline, cruise_speed=flight_speed)
    pos_start = spline.evaluate(0.0)
    tangent_start = spline.evaluate_derivative(0.001, 1)
    norm_t0 = np.linalg.norm(tangent_start)
    tan_dir0 = tangent_start / norm_t0 if norm_t0 > 1e-4 else np.array([1.0, 0.0, 0.0])
    yaw_start = np.arctan2(tan_dir0[1], tan_dir0[0])

    tangent_end = spline.evaluate_derivative(0.999, 1)
    norm_te = np.linalg.norm(tangent_end)
    tan_dir_end = tangent_end / norm_te if norm_te > 1e-4 else tan_dir0
    yaw_end = np.arctan2(tan_dir_end[1], tan_dir_end[0])
    pos_goal = spline.evaluate(1.0)

    gazebo_env = GazeboEnvironment(environment_3d=env_3d, gui=False, dt=dt_sim,
                                   world_name="airspace_world")

    drone = GazeboQuadrotor(
        gazebo_environment=gazebo_env,
        start_position=pos_start,
        start_orientation_euler=[0.0, 0.0, yaw_start],
        enable_motor_model=True,
        enable_motor_lag=motor_lag
    )
    drone.velocity = profile.speed_at(0.0) * tan_dir0
    gazebo_env.fallback.velocity = drone.velocity.copy()

    tracker = GazeboTrajectoryTracker(
        mass=drone.mass, g=drone.g, Pq=2.5, Pw=0.08, tau_limit=1.0,
        Kp_pos=[1.2, 1.2, 2.5], Kd_pos=[1.5, 1.5, 2.0]
    )

    wind_model = WindDisturbanceModel(
        steady_wind=wind_speed, gust_amplitude=gust_amp, gust_frequency=gust_freq,
        cd_lin=drag_lin, cd_rot=drag_rot
    ) if enable_wind else None
    if enable_wind:
        print(f"  Wind & Gust Disturbances: ENABLED | steady={tuple(wind_speed)} m/s "
              f"| gust_amp={tuple(gust_amp)} m/s | gust_freq={tuple(gust_freq)} Hz "
              f"| cd_lin={drag_lin} | cd_rot={drag_rot}")
    if motor_lag:
        print("  Motor first-order lag: ENABLED")

    # ── Live 3D window setup ──
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')
    env_3d.visualize(ax)
    ax.plot(spline_pts[:, 0], spline_pts[:, 1], spline_pts[:, 2], 'b--', lw=1.3,
            label='B-Spline Reference', zorder=4)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Live Gazebo-Stack Flight (Python 6-DOF Engine)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=-60)

    all_pts = np.vstack([spline_pts, [env_3d.start, env_3d.goal]])
    center = all_pts.mean(axis=0)
    span = float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0))) * 0.6 + 50.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(max(0.0, center[2] - span * 0.5), center[2] + span * 0.5)

    axis_len = span * 0.08
    trail_line, = ax.plot([], [], [], 'r-', lw=2.0, zorder=5, label='Flown Trajectory')
    x_axis_line, = ax.plot([], [], [], color='red', lw=2.5, zorder=7)
    y_axis_line, = ax.plot([], [], [], color='limegreen', lw=2.5, zorder=7)
    z_axis_line, = ax.plot([], [], [], color='blue', lw=2.5, zorder=7)
    drone_marker = ax.scatter([pos_start[0]], [pos_start[1]], [pos_start[2]],
                              c='black', s=70, marker='o', zorder=8)
    status_text = fig.text(0.02, 0.97, '', fontsize=9, family='monospace', va='top')

    # Wind indicator: a fixed-anchor arrow near a corner of the scene, oriented and scaled
    # to the instantaneous wind vector each frame (only shown when --wind is enabled).
    wind_anchor = np.array([center[0] - span * 0.85, center[1] - span * 0.85, center[2] + span * 0.7])
    wind_arrow_line, = ax.plot([], [], [], color='darkorange', lw=3.0, zorder=9)
    wind_label = fig.text(0.02, 0.02, '', fontsize=9, family='monospace', va='bottom', color='darkorange')

    print("\n[Launching Live 3D Viewer Window]")
    print(" - Left-click + drag to orbit the camera.")
    print(" - Scroll to zoom.")
    print(" - Close the window to exit.\n")

    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    substeps = max(1, int(round(1.0 / (fps * dt_sim))))
    frame_dt_target = substeps * dt_sim
    trail_buffer = deque(maxlen=800)

    state = {'t': 0.0, 's_travelled': 0.0, 'hold_elapsed': 0.0, 'goal_reached': False,
             'wind_vel': np.zeros(3)}

    def reset_flight():
        drone.reset(pos_start, [0.0, 0.0, yaw_start])
        drone.velocity = profile.speed_at(0.0) * tan_dir0
        gazebo_env.fallback.velocity = drone.velocity.copy()
        tracker.prev_q_des = np.array([1.0, 0.0, 0.0, 0.0])
        state['t'] = 0.0
        state['s_travelled'] = 0.0
        state['hold_elapsed'] = 0.0
        state['goal_reached'] = False
        trail_buffer.clear()

    def physics_substep():
        if state['goal_reached']:
            state['hold_elapsed'] += dt_sim
            if state['hold_elapsed'] >= hold_time:
                reset_flight()
                return
            pos_target, vel_target, tangent, yaw_target = pos_goal, np.zeros(3), tan_dir_end, yaw_end
        else:
            u_current = profile.u_at(state['s_travelled'])
            v_ref = profile.speed_at(state['s_travelled'])
            pos_target = spline.evaluate(u_current)
            tangent = spline.evaluate_derivative(u_current, 1)
            norm_tan = np.linalg.norm(tangent)
            tangent_dir = tangent / norm_tan if norm_tan > 1e-4 else np.array([1.0, 0.0, 0.0])
            vel_target = v_ref * tangent_dir
            yaw_target = np.arctan2(tangent_dir[1], tangent_dir[0])

        if wind_model:
            w_force, w_torque = wind_model.compute_aerodynamic_wrench(drone.velocity, drone.omega, state['t'])
            state['wind_vel'] = wind_model.get_wind_velocity(state['t'])
        else:
            w_force, w_torque = np.zeros(3), np.zeros(3)

        thrust_cmd, torques_cmd, q_des, axis_err = tracker.compute_full_control(
            pos_target=pos_target, pos_current=drone.position, q_current=drone.q,
            omega_current=drone.omega, vel_target=vel_target, vel_current=drone.velocity,
            tangent=tangent, yaw_target=yaw_target
        )
        drone.step(thrust_cmd, torques_cmd, wind_force_world=w_force, wind_torque_body=w_torque)

        if not state['goal_reached']:
            state['s_travelled'] = min(state['s_travelled'] + v_ref * dt_sim, profile.length)
            if state['s_travelled'] >= profile.length - 1e-6:
                state['goal_reached'] = True
        state['t'] += dt_sim

    try:
        while plt.fignum_exists(fig.number):
            frame_wall_start = time.time()

            for _ in range(substeps):
                physics_substep()

            p = drone.position
            trail_buffer.append(p.copy())
            trail_arr = np.array(trail_buffer)
            trail_line.set_data_3d(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2])

            R = QuaternionMath.to_rotation_matrix(drone.q)
            x_axis_line.set_data_3d([p[0], p[0] + axis_len * R[0, 0]],
                                    [p[1], p[1] + axis_len * R[1, 0]],
                                    [p[2], p[2] + axis_len * R[2, 0]])
            y_axis_line.set_data_3d([p[0], p[0] + axis_len * R[0, 1]],
                                    [p[1], p[1] + axis_len * R[1, 1]],
                                    [p[2], p[2] + axis_len * R[2, 1]])
            z_axis_line.set_data_3d([p[0], p[0] + axis_len * R[0, 2]],
                                    [p[1], p[1] + axis_len * R[1, 2]],
                                    [p[2], p[2] + axis_len * R[2, 2]])
            drone_marker._offsets3d = ([p[0]], [p[1]], [p[2]])

            speed = float(np.linalg.norm(drone.velocity))
            status_text.set_text(
                f"t={state['t']:6.2f}s  progress={100*state['s_travelled']/profile.length:5.1f}%  "
                f"speed={speed:5.1f} m/s  pos=[{p[0]:6.0f},{p[1]:6.0f},{p[2]:5.0f}]m"
                + ("  [HOLDING AT GOAL]" if state['goal_reached'] else "")
            )

            if wind_model:
                v_wind = state['wind_vel']
                wind_speed_now = float(np.linalg.norm(v_wind))
                wind_dir = v_wind / wind_speed_now if wind_speed_now > 1e-3 else np.zeros(3)
                arrow_len = axis_len * 1.5 * min(1.0, wind_speed_now / 5.0 + 0.3)
                tip = wind_anchor + wind_dir * arrow_len
                wind_arrow_line.set_data_3d([wind_anchor[0], tip[0]],
                                            [wind_anchor[1], tip[1]],
                                            [wind_anchor[2], tip[2]])
                wind_label.set_text(
                    f"wind=[{v_wind[0]:5.1f},{v_wind[1]:5.1f},{v_wind[2]:5.1f}] m/s  |wind|={wind_speed_now:4.1f} m/s"
                )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            elapsed = time.time() - frame_wall_start
            if frame_dt_target > elapsed:
                time.sleep(frame_dt_target - elapsed)
    except KeyboardInterrupt:
        pass

    print("\n[Viewer closed]")


if __name__ == '__main__':
    args = parse_args()
    launch_live_viewer(
        seed=args.seed,
        flight_speed=args.speed,
        fps=args.fps,
        hold_time=args.hold,
        motor_lag=args.motor_lag,
        enable_wind=args.wind,
        wind_speed=args.wind_speed,
        gust_amp=args.gust_amp,
        gust_freq=args.gust_freq,
        drag_lin=args.drag_lin,
        drag_rot=args.drag_rot
    )
