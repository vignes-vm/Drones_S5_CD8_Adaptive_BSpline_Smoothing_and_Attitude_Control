"""
Interactive Desktop GUI Application for 3D Quadrotor Flight Simulation & Control.
Opens a native Windows GUI window with live 3D flight animation, real-time telemetry graphs,
and interactive Play/Pause/Slider controls.

Usage:
  py -3.12 pybullet/gui_visualizer.py
"""
import os
import sys
import numpy as np
import matplotlib
# Ensure interactive GUI backend is loaded
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.bspline import BSplinePath
from pybullet.src.pybullet_env import PyBulletEnvironment
from pybullet.src.quadrotor_pybullet import PyBulletQuadrotor
from pybullet.src.quaternion_math import QuaternionMath
from pybullet.src.trajectory_tracker import PyBulletTrajectoryTracker


def launch_gui_simulation(seed=42, duration=20.0):
    print("=" * 65)
    print("LAUNCHING INTERACTIVE DESKTOP GUI SIMULATION WINDOW...")
    print("=" * 65)

    # 1. Environment & Path Planning
    print("[1/3] Generating 3D Obstacle Environment & B-Spline Path...")
    env_3d = Environment3D()
    env_3d.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)

    astar = AStarPlanner3D(env_3d, grid_resolution=50)
    raw_wp = np.array(astar.plan(env_3d.start, env_3d.goal))
    baseline_wp = np.array(astar.smooth_waypoints(raw_wp, min_distance=100))
    spline = BSplinePath(baseline_wp, param_mode='uniform')
    spline_pts = spline.sample_path(200)

    # 2. Physics Simulation Setup
    print("[2/3] Simulating 6-DOF Quadrotor Dynamics...")
    dt_sim = 0.005
    pb_env = PyBulletEnvironment(gui=False, environment_3d=env_3d, dt=dt_sim)
    drone = PyBulletQuadrotor(physics_environment=pb_env, start_position=env_3d.start, enable_motor_model=True)
    tracker = PyBulletTrajectoryTracker(
        mass=drone.mass, g=drone.g, Pq=2.5, Pw=0.08, tau_limit=1.0,
        Kp_pos=[1.2, 1.2, 2.5], Kd_pos=[1.5, 1.5, 2.0]
    )

    flight_speed = 25.0  # 25 m/s flight speed
    duration = (spline.path_length() / flight_speed) + 5.0  # ~115s full trajectory
    n_steps = int(duration / dt_sim)
    dt_u = (flight_speed / max(1.0, spline.path_length())) * dt_sim
    u_current = 0.001

    log = {'t': [], 'pos': [], 'pos_des': [], 'euler': [], 'euler_ref': [], 'motor_w': []}

    for i in range(n_steps):
        t_curr = i * dt_sim
        pos_meas, vel_meas, q_meas, omega_meas = drone.get_state()
        pos_des = spline.evaluate(u_current)
        tangent = spline.evaluate_derivative(u_current, 1)
        tang_norm = np.linalg.norm(tangent)
        v_des = (tangent / tang_norm * flight_speed) if tang_norm > 1e-4 else np.zeros(3)

        thrust, torques, q_ref, _, _ = tracker.step_control(
            pos_des, v_des, pos_meas, vel_meas, q_meas, omega_meas, tangent=tangent
        )
        drone.apply_control_inputs(thrust, torques)
        pb_env.step()

        if u_current < 0.999:
            u_current = min(0.999, u_current + dt_u)

        if i % 10 == 0:  # Sample at 50 Hz for smooth GUI animation
            log['t'].append(t_curr)
            log['pos'].append(pos_meas.copy())
            log['pos_des'].append(pos_des.copy())
            log['euler'].append(QuaternionMath.to_euler(q_meas))
            log['euler_ref'].append(QuaternionMath.to_euler(q_ref))
            log['motor_w'].append(drone.motor_speeds.copy())

    for k in log:
        log[k] = np.array(log[k])

    print(f"[3/3] Launching GUI Window with {len(log['t'])} frame animation...")

    # 3. Create Interactive Matplotlib GUI Window
    fig = plt.figure(figsize=(14, 9), facecolor='#0f172a')
    fig.canvas.manager.set_window_title('PyBullet 3D UAV Flight Simulation GUI')

    # Grid layout: Top-Left (3D View), Top-Right (Position), Bottom-Left (Attitude), Bottom-Right (Motors)
    ax_3d   = fig.add_subplot(221, projection='3d', facecolor='#1e293b')
    ax_pos  = fig.add_subplot(222, facecolor='#1e293b')
    ax_att  = fig.add_subplot(223, facecolor='#1e293b')
    ax_rpm  = fig.add_subplot(224, facecolor='#1e293b')

    plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.15, wspace=0.25, hspace=0.35)

    # ── 3D Viewport Setup ──
    env_3d.visualize(ax_3d)
    ax_3d.plot(baseline_wp[:, 0], baseline_wp[:, 1], baseline_wp[:, 2], 'k:', lw=1.2, label='A* Grid')
    ax_3d.plot(spline_pts[:, 0], spline_pts[:, 1], spline_pts[:, 2], 'r--', lw=1.8, label='GA B-Spline')
    line_trail, = ax_3d.plot([], [], [], 'g-', lw=2.5, label='Drone Path')
    drone_marker, = ax_3d.plot([], [], [], 'o', color='#38bdf8', ms=8, markeredgecolor='white')
    ax_3d.set_title("3D Flight Viewport", color='white', fontsize=11, fontweight='bold')

    # ── Telemetry Plots Setup ──
    # Position
    line_x, = ax_pos.plot(log['t'], log['pos'][:, 0], 'g-', label='X Actual')
    line_y, = ax_pos.plot(log['t'], log['pos'][:, 1], 'b-', label='Y Actual')
    line_z, = ax_pos.plot(log['t'], log['pos'][:, 2], 'r-', label='Z Actual')
    curr_pos_marker = ax_pos.axvline(x=0, color='yellow', linestyle='--', alpha=0.7)
    ax_pos.set_title("Position Tracking X, Y, Z (m)", color='white', fontsize=11, fontweight='bold')
    ax_pos.tick_params(colors='white')
    ax_pos.grid(True, color='#334155', alpha=0.5)
    ax_pos.legend(loc='upper left', fontsize=8)

    # Attitude
    line_r, = ax_att.plot(log['t'], log['euler'][:, 0], 'cyan', label='Roll φ')
    line_p, = ax_att.plot(log['t'], log['euler'][:, 1], 'magenta', label='Pitch θ')
    line_yw, = ax_att.plot(log['t'], log['euler'][:, 2], 'yellow', label='Yaw ψ')
    curr_att_marker = ax_att.axvline(x=0, color='yellow', linestyle='--', alpha=0.7)
    ax_att.set_title("Attitude Angles (rad)", color='white', fontsize=11, fontweight='bold')
    ax_att.tick_params(colors='white')
    ax_att.grid(True, color='#334155', alpha=0.5)
    ax_att.legend(loc='upper left', fontsize=8)

    # Motor RPMs
    for m in range(4):
        ax_rpm.plot(log['t'], log['motor_w'][:, m], label=f'Motor {m+1}')
    curr_rpm_marker = ax_rpm.axvline(x=0, color='yellow', linestyle='--', alpha=0.7)
    ax_rpm.set_title("Rotor Angular Speeds (rad/s)", color='white', fontsize=11, fontweight='bold')
    ax_rpm.tick_params(colors='white')
    ax_rpm.grid(True, color='#334155', alpha=0.5)
    ax_rpm.legend(loc='upper right', fontsize=8)

    # ── Interactive Controls (Slider & Buttons) ──
    ax_slider = plt.axes([0.15, 0.04, 0.55, 0.03], facecolor='#1e293b')
    slider = Slider(ax_slider, 'Time (s)', 0, log['t'][-1], valinit=0, valfmt='%.1f s', color='#3b82f6')
    slider.label.set_color('white')
    slider.valtext.set_color('white')

    ax_btn_play = plt.axes([0.73, 0.035, 0.1, 0.04])
    btn_play = Button(ax_btn_play, 'Play/Pause', color='#3b82f6', hovercolor='#2563eb')
    btn_play.label.set_color('white')

    ax_btn_reset = plt.axes([0.85, 0.035, 0.09, 0.04])
    btn_reset = Button(ax_btn_reset, 'Reset', color='#475569', hovercolor='#334155')
    btn_reset.label.set_color('white')

    state = {'anim_running': True, 'frame': 0}

    def update(frame):
        state['frame'] = frame
        t_val = log['t'][frame]
        
        # Update 3D trail & marker
        line_trail.set_data(log['pos'][:frame, 0], log['pos'][:frame, 1])
        line_trail.set_3d_properties(log['pos'][:frame, 2])

        drone_marker.set_data([log['pos'][frame, 0]], [log['pos'][frame, 1]])
        drone_marker.set_3d_properties([log['pos'][frame, 2]])

        # Update telemetry vertical line markers
        curr_pos_marker.set_xdata([t_val])
        curr_att_marker.set_xdata([t_val])
        curr_rpm_marker.set_xdata([t_val])

        slider.eventson = False
        slider.set_val(t_val)
        slider.eventson = True

        return line_trail, drone_marker

    anim = FuncAnimation(fig, update, frames=len(log['t']), interval=30, blit=False)

    def toggle_play(event):
        if state['anim_running']:
            anim.event_source.stop()
            state['anim_running'] = False
        else:
            anim.event_source.start()
            state['anim_running'] = True

    def reset_anim(event):
        slider.set_val(0)
        update(0)

    def on_slider(val):
        frame = int((val / log['t'][-1]) * (len(log['t']) - 1))
        update(frame)
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)
    btn_reset.on_clicked(reset_anim)
    slider.on_changed(on_slider)

    print("GUI window opened! Interact with Play/Pause, Slider, and 3D Camera controls.")
    plt.show()


if __name__ == '__main__':
    launch_gui_simulation()
