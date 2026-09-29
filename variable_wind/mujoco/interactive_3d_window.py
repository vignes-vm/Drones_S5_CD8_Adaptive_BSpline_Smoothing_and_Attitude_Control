r"""
MuJoCo Interactive 3D Window Visualizer:
Runs the native MuJoCo physics engine simulation (Paper 1 A* + GA B-Spline & Paper 2 Quaternion Control),
records real-time physics telemetry logs, and renders an interactive 3D flight window with:
- 3D flight trajectory and dynamic drone attitude rendering
- Real-time 3D wind vector arrow attached to the drone
- Live HUD displaying: Location (X, Y, Z), Velocities (Vx, Vy, Vz, Speed), RPY Angles (Roll, Pitch, Yaw),
  Wind Vector & Zone, Motor RPMs, and Tracking Errors
- Interactive Controls: Play, Pause, Reset, Timeline Scrubber, Speed Multipliers, Re-plan
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.obstacle_planner import ObstacleEnvironment, AStar3DPlanner, GAAdaptiveBSplineSmoother, TrajectoryGenerator
from src.mujoco_env import MuJoCoVariableWindEnv


def quat_to_rpy_deg(q: np.ndarray) -> Tuple[float, float, float]:
    """Converts quaternion [w, x, y, z] to Roll, Pitch, Yaw in degrees."""
    w, x, y, z = q
    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


class MuJoCoInteractive3DApp:
    def __init__(self, root: tk.Tk, seed: int = 42):
        self.root = root
        self.root.title("MuJoCo 3D Interactive Window - Variable Wind Quadrotor Flight")
        self.root.geometry("1400x920")
        self.root.configure(bg="#1e1e2e")

        self.seed = seed
        self.is_playing = False
        self.current_step = 0
        self.speed_multiplier = 1.0
        self.dt = 0.005

        # 1. Run MuJoCo Engine Simulation and collect logs
        self._run_mujoco_engine_mission()

        # 2. Build UI layout
        self._build_ui()

        # 3. Render Initial Frame
        self._update_3d_frame(self.current_step)

        # 4. Start animation loop
        self._animation_tick()

    def _run_mujoco_engine_mission(self):
        """Runs the native MuJoCo physics engine and logs telemetry."""
        self.env_obstacles = ObstacleEnvironment(bounds=(25.0, 25.0, 12.0), seed=self.seed)
        self.start_pos = np.array([1.0, 1.0, 1.0])
        self.goal_pos = np.array([22.0, 22.0, 8.0])
        self.obstacles = self.env_obstacles.generate_random_obstacles(
            num_obstacles=12, start=self.start_pos, goal=self.goal_pos
        )

        astar = AStar3DPlanner(self.env_obstacles, grid_res=1.0)
        waypoints = astar.plan(self.start_pos, self.goal_pos)
        smoother = GAAdaptiveBSplineSmoother(self.env_obstacles, degree=3)
        self.pos_spline, vel_spline, acc_spline, arc_lengths = smoother.smooth(waypoints, num_samples=300)
        self.traj_gen = TrajectoryGenerator(self.pos_spline, vel_spline, acc_spline, arc_lengths, v_max=2.8, a_max=3.2)
        self.t_total = self.traj_gen.t_total
        self.total_steps = int(np.ceil(self.t_total / self.dt))

        # Instantiate real MuJoCo Environment
        self.mujoco_env = MuJoCoVariableWindEnv(obstacles=self.obstacles, dt=self.dt)
        self.mujoco_env.reset(initial_pos=self.start_pos)

        # Telemetry logs
        self.logs = {
            "time": [],
            "pos": [],
            "vel": [],
            "speed": [],
            "quat": [],
            "rpy": [],
            "omega": [],
            "pos_ref": [],
            "vel_ref": [],
            "pos_error": [],
            "quat_err_norm": [],
            "actual_omega": [],
            "v_wind": [],
            "wind_speed": [],
            "active_zone": [],
            "dist": [],
        }

        # Step MuJoCo Engine for the full duration
        for step_i in range(self.total_steps):
            t = step_i * self.dt
            p_ref, v_ref, a_ref, yaw_ref, dist_s = self.traj_gen.sample(t)
            telem = self.mujoco_env.step(
                pos_ref=p_ref,
                vel_ref=v_ref,
                acc_ref=a_ref,
                yaw_ref=yaw_ref,
                dist_along_path=dist_s,
            )

            p_meas = telem["pos"]
            v_meas = telem["vel"]
            q_meas = telem["quat"]
            w_meas = telem["omega"]
            roll_d, pitch_d, yaw_d = quat_to_rpy_deg(q_meas)

            self.logs["time"].append(t)
            self.logs["pos"].append(p_meas)
            self.logs["vel"].append(v_meas)
            self.logs["speed"].append(float(np.linalg.norm(v_meas)))
            self.logs["quat"].append(q_meas)
            self.logs["rpy"].append((roll_d, pitch_d, yaw_d))
            self.logs["omega"].append(w_meas)
            self.logs["pos_ref"].append(p_ref)
            self.logs["vel_ref"].append(v_ref)
            self.logs["pos_error"].append(telem["pos_error"])
            self.logs["quat_err_norm"].append(telem["quat_err_norm"])
            self.logs["actual_omega"].append(telem["actual_omega"])
            self.logs["v_wind"].append(telem["v_wind"])
            self.logs["wind_speed"].append(telem["wind_speed"])
            self.logs["active_zone"].append(telem["active_zone"])
            self.logs["dist"].append(dist_s)

        for k in ["pos", "vel", "speed", "quat", "rpy", "omega", "pos_ref", "vel_ref",
                  "pos_error", "quat_err_norm", "actual_omega", "v_wind", "wind_speed", "dist"]:
            self.logs[k] = np.array(self.logs[k])

    def _build_ui(self):
        """Builds Tkinter UI layout."""
        # Main frames
        self.top_frame = tk.Frame(self.root, bg="#1e1e2e")
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.bottom_frame = tk.Frame(self.root, bg="#181825", relief=tk.RAISED, bd=2)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        # 1. 3D Matplotlib Canvas
        self.fig = plt.figure(figsize=(10, 6.5), facecolor="#181825")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#11111b")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 2. Right Telemetry Sidebar
        self.sidebar = tk.Frame(self.top_frame, bg="#181825", width=380, padx=12, pady=10)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self._build_sidebar_hud()

        # 3. Bottom Controls & Scrubber
        self._build_bottom_controls()

        # Pre-render static obstacles and reference trajectory
        self._render_static_scene()

    def _build_sidebar_hud(self):
        """Creates rich live telemetry displays on sidebar."""
        title_lbl = tk.Label(
            self.sidebar,
            text="ENGINE: MUJOCO (PHYSICS LOG)",
            font=("Segoe UI", 12, "bold"),
            fg="#89b4fa",
            bg="#181825",
        )
        title_lbl.pack(anchor="w", pady=(0, 6))

        # Helper card creator
        def create_card(title: str, text_var: tk.StringVar, fg_color: str = "#cdd6f4"):
            card = tk.LabelFrame(
                self.sidebar,
                text=f" {title} ",
                font=("Segoe UI", 9, "bold"),
                fg=fg_color,
                bg="#181825",
                padx=8,
                pady=4,
            )
            card.pack(fill=tk.X, pady=4)
            lbl = tk.Label(
                card,
                textvariable=text_var,
                font=("Consolas", 9),
                fg="#cdd6f4",
                bg="#181825",
                justify=tk.LEFT,
                anchor="w",
            )
            lbl.pack(fill=tk.X)
            return card

        # StringVars
        self.hud_loc_var = tk.StringVar(value="X: 0.00 m\nY: 0.00 m\nZ: 0.00 m (Alt)")
        self.hud_vel_var = tk.StringVar(value="Vx: 0.00 m/s\nVy: 0.00 m/s\nVz: 0.00 m/s\nSpeed: 0.00 m/s")
        self.hud_rpy_var = tk.StringVar(value="Roll (φ):  0.0°\nPitch (θ): 0.0°\nYaw (ψ):   0.0°")
        self.hud_wind_var = tk.StringVar(value="Wx: 0.0 m/s | Wy: 0.0 m/s\nSpeed: 0.0 m/s\nZone: Calm")
        self.hud_act_var = tk.StringVar(value="RPM: [0, 0, 0, 0]\nPos Err: 0.00 m\nAtt Err: 0.000")

        create_card("📍 LIVE LOCATION & ALTITUDE", self.hud_loc_var, "#a6e3a1")
        create_card("🚀 LIVE LINEAR VELOCITY", self.hud_vel_var, "#89dceb")
        create_card("📐 ATTITUDE ANGLES (RPY)", self.hud_rpy_var, "#f9e2af")
        create_card("💨 SPATIAL VARIABLE WIND", self.hud_wind_var, "#fab387")
        create_card("⚙️ MOTOR RPMS & TRACKING ERROR", self.hud_act_var, "#f38ba8")

    def _build_bottom_controls(self):
        """Builds bottom playback toolbar and timeline scrubber."""
        # Scrubber bar
        slider_frame = tk.Frame(self.bottom_frame, bg="#181825")
        slider_frame.pack(fill=tk.X, padx=10, pady=2)

        self.time_lbl = tk.Label(
            slider_frame,
            text="Time: 0.00s / 0.00s",
            font=("Segoe UI", 10, "bold"),
            fg="#cdd6f4",
            bg="#181825",
            width=22,
            anchor="w",
        )
        self.time_lbl.pack(side=tk.LEFT)

        self.scrubber = ttk.Scale(
            slider_frame,
            from_=0,
            to=self.total_steps - 1,
            orient=tk.HORIZONTAL,
            command=self._on_scrubber_slide,
        )
        self.scrubber.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Buttons toolbar
        btn_frame = tk.Frame(self.bottom_frame, bg="#181825")
        btn_frame.pack(fill=tk.X, padx=10, pady=4)

        self.btn_play = tk.Button(
            btn_frame,
            text="▶ Play",
            font=("Segoe UI", 10, "bold"),
            bg="#a6e3a1",
            fg="#11111b",
            width=10,
            command=self._toggle_play,
        )
        self.btn_play.pack(side=tk.LEFT, padx=4)

        self.btn_reset = tk.Button(
            btn_frame,
            text="↺ Reset",
            font=("Segoe UI", 10, "bold"),
            bg="#f38ba8",
            fg="#11111b",
            width=10,
            command=self._reset_playback,
        )
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        self.btn_replan = tk.Button(
            btn_frame,
            text="🎲 Randomize & Re-plan",
            font=("Segoe UI", 10, "bold"),
            bg="#cba6f7",
            fg="#11111b",
            padx=10,
            command=self._replan_mission,
        )
        self.btn_replan.pack(side=tk.LEFT, padx=10)

        # Speed buttons
        speed_lbl = tk.Label(
            btn_frame, text="Speed:", font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825"
        )
        speed_lbl.pack(side=tk.LEFT, padx=(20, 4))

        for spd in [0.5, 1.0, 2.0, 4.0]:
            btn_s = tk.Button(
                btn_frame,
                text=f"{spd}x",
                font=("Segoe UI", 9),
                bg="#313244",
                fg="#cdd6f4",
                width=5,
                command=lambda s=spd: self._set_speed(s),
            )
            btn_s.pack(side=tk.LEFT, padx=2)

    def _render_static_scene(self):
        """Draws obstacles, ground grid, and reference trajectory."""
        self.ax.clear()
        self.ax.set_facecolor("#11111b")
        self.fig.patch.set_facecolor("#181825")

        self.ax.set_xlim([0, 25])
        self.ax.set_ylim([0, 25])
        self.ax.set_zlim([0, 14])

        self.ax.set_xlabel("X (m)", color="#cdd6f4", labelpad=8)
        self.ax.set_ylabel("Y (m)", color="#cdd6f4", labelpad=8)
        self.ax.set_zlabel("Z (m)", color="#cdd6f4", labelpad=8)
        self.ax.tick_params(colors="#a6adc8")
        self.ax.set_title("MuJoCo Engine Flight Simulation [Paper 1 + Paper 2 + Variable Wind]", color="#89b4fa", fontsize=11, pad=12)

        # Draw Ground Grid
        gx, gy = np.meshgrid(np.linspace(0, 25, 6), np.linspace(0, 25, 6))
        self.ax.plot_wireframe(gx, gy, np.zeros_like(gx), color="#313244", alpha=0.4, linewidth=0.5)

        # 3D Obstacles
        for obs in self.obstacles:
            cx, cy, cz = obs.center
            r = obs.radius
            if obs.kind == "cylinder":
                z_lin = np.linspace(0, obs.height, 12)
                theta_lin = np.linspace(0, 2 * np.pi, 16)
                theta_grid, z_grid = np.meshgrid(theta_lin, z_lin)
                x_grid = cx + r * np.cos(theta_grid)
                y_grid = cy + r * np.sin(theta_grid)
                self.ax.plot_surface(x_grid, y_grid, z_grid, color="#f38ba8", alpha=0.35, shade=True)
            else:
                u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:8j]
                xs = cx + r * np.cos(u) * np.sin(v)
                ys = cy + r * np.sin(u) * np.sin(v)
                zs = cz + r * np.cos(v)
                self.ax.plot_surface(xs, ys, zs, color="#fab387", alpha=0.35, shade=True)

        # Reference trajectory
        self.ax.plot(
            self.pos_spline[:, 0],
            self.pos_spline[:, 1],
            self.pos_spline[:, 2],
            "g--",
            linewidth=1.8,
            alpha=0.7,
            label="Paper 1 B-Spline Ref",
        )

        # Start / Goal markers
        self.ax.scatter(*self.start_pos, color="#a6e3a1", s=70, marker="o", label="Start")
        self.ax.scatter(*self.goal_pos, color="#f38ba8", s=70, marker="^", label="Goal")

        # Dynamic plot handles
        (self.line_actual,) = self.ax.plot([], [], [], color="#89dceb", linewidth=2.2, label="MuJoCo Actual Path")
        self.drone_arms = [self.ax.plot([], [], [], color="#cdd6f4", linewidth=3.0)[0] for _ in range(2)]
        self.drone_rotors = [self.ax.scatter([], [], [], color="#f9e2af", s=50) for _ in range(4)]
        self.drone_core = self.ax.scatter([], [], [], color="#89b4fa", s=90)

        # Wind Vector Quiver Arrow Handle (initialized as None)
        self.wind_quiver = None

        self.ax.legend(loc="upper left", facecolor="#181825", edgecolor="#313244", labelcolor="#cdd6f4", fontsize=8)
        self.canvas.draw_idle()

    def _update_3d_frame(self, step_idx: int, update_scrubber: bool = True):
        """Updates drone position, dynamic attitude, wind vector, and live HUD."""
        if step_idx >= self.total_steps:
            step_idx = self.total_steps - 1

        p = self.logs["pos"][step_idx]
        v = self.logs["vel"][step_idx]
        spd = self.logs["speed"][step_idx]
        r, pt, y = self.logs["rpy"][step_idx]
        q = self.logs["quat"][step_idx]
        wv = self.logs["v_wind"][step_idx]
        ws = self.logs["wind_speed"][step_idx]
        wz = self.logs["active_zone"][step_idx]
        rpm = self.logs["actual_omega"][step_idx]
        perr = self.logs["pos_error"][step_idx]
        qerr = self.logs["quat_err_norm"][step_idx]
        t = self.logs["time"][step_idx]

        # 1. Update Trajectory Trail
        flown_p = self.logs["pos"][: step_idx + 1]
        self.line_actual.set_data(flown_p[:, 0], flown_p[:, 1])
        self.line_actual.set_3d_properties(flown_p[:, 2])

        # 2. Update Drone 3D Orientation Frame
        w, qx, qy, qz = q
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*w), 2*(qx*qz + qy*w)],
            [2*(qx*qy + qz*w), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*w)],
            [2*(qx*qz - qy*w), 2*(qy*qz + qx*w), 1 - 2*(qx**2 + qy**2)]
        ])
        L = 0.45  # Visual arm length
        arm_x_b = np.array([[-L, 0, 0], [L, 0, 0]])
        arm_y_b = np.array([[0, -L, 0], [0, L, 0]])
        arm_x_w = p + (R @ arm_x_b.T).T
        arm_y_w = p + (R @ arm_y_b.T).T

        self.drone_arms[0].set_data(arm_x_w[:, 0], arm_x_w[:, 1])
        self.drone_arms[0].set_3d_properties(arm_x_w[:, 2])
        self.drone_arms[1].set_data(arm_y_w[:, 0], arm_y_w[:, 1])
        self.drone_arms[1].set_3d_properties(arm_y_w[:, 2])

        self.drone_core._offsets3d = ([p[0]], [p[1]], [p[2]])

        rotor_positions = p + (R @ np.array([[L, 0, 0], [0, L, 0], [-L, 0, 0], [0, -L, 0]]).T).T
        for i in range(4):
            self.drone_rotors[i]._offsets3d = ([rotor_positions[i, 0]], [rotor_positions[i, 1]], [rotor_positions[i, 2]])

        # 3. Dynamic 3D Wind Vector Arrow
        if self.wind_quiver is not None:
            self.wind_quiver.remove()

        arrow_scale = 0.45
        wx, wy, wz_val = wv
        self.wind_quiver = self.ax.quiver(
            p[0], p[1], p[2],
            wx * arrow_scale, wy * arrow_scale, wz_val * arrow_scale,
            color="#fab387",
            linewidth=2.5,
            arrow_length_ratio=0.35,
        )

        # 4. Update HUD Text Displays
        self.hud_loc_var.set(
            f"X: {p[0]:+7.2f} m\n"
            f"Y: {p[1]:+7.2f} m\n"
            f"Z: {p[2]:+7.2f} m (Alt: {p[2]:.2f}m)"
        )
        self.hud_vel_var.set(
            f"Vx: {v[0]:+6.2f} m/s\n"
            f"Vy: {v[1]:+6.2f} m/s\n"
            f"Vz: {v[2]:+6.2f} m/s\n"
            f"Speed: {spd:5.2f} m/s"
        )
        self.hud_rpy_var.set(
            f"Roll (φ):  {r:+6.1f}°\n"
            f"Pitch (θ): {pt:+6.1f}°\n"
            f"Yaw (ψ):   {y:+6.1f}°"
        )
        self.hud_wind_var.set(
            f"Wx: {wx:+5.1f} m/s | Wy: {wy:+5.1f} m/s\n"
            f"Speed: {ws:4.1f} m/s ({ws*3.6:4.1f} km/h)\n"
            f"Zone:  {wz}"
        )
        self.hud_act_var.set(
            f"RPM: [{int(rpm[0])}, {int(rpm[1])}, {int(rpm[2])}, {int(rpm[3])}]\n"
            f"Pos Err: {perr:5.2f} m\n"
            f"Att Err: {qerr:5.3f} rad"
        )

        self.time_lbl.config(text=f"Time: {t:5.2f}s / {self.t_total:5.2f}s")
        if update_scrubber:
            self.scrubber.set(step_idx)
        self.canvas.draw_idle()

    def _animation_tick(self):
        """Timer callback advancing playback frames."""
        if self.is_playing:
            step_inc = max(1, int(self.speed_multiplier * 2))
            self.current_step += step_inc
            if self.current_step >= self.total_steps:
                self.current_step = self.total_steps - 1
                self.is_playing = False
                self.btn_play.config(text="▶ Play", bg="#a6e3a1")

            self._update_3d_frame(self.current_step, update_scrubber=True)

        self.root.after(25, self._animation_tick)

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            if self.current_step >= self.total_steps - 1:
                self.current_step = 0
            self.btn_play.config(text="⏸ Pause", bg="#f9e2af")
        else:
            self.btn_play.config(text="▶ Play", bg="#a6e3a1")

    def _reset_playback(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Play", bg="#a6e3a1")
        self.current_step = 0
        self._update_3d_frame(0, update_scrubber=True)

    def _set_speed(self, spd: float):
        self.speed_multiplier = spd

    def _on_scrubber_slide(self, val):
        if not self.is_playing:
            idx = int(float(val))
            self.current_step = idx
            self._update_3d_frame(idx, update_scrubber=False)

    def _replan_mission(self):
        self.is_playing = False
        self.btn_play.config(text="▶ Play", bg="#a6e3a1")
        self.seed = np.random.randint(1, 99999)
        self._run_mujoco_engine_mission()
        self.scrubber.config(to=self.total_steps - 1)
        self._render_static_scene()
        self.current_step = 0
        self._update_3d_frame(0)


def launch_interactive_window():
    root = tk.Tk()
    app = MuJoCoInteractive3DApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_interactive_window()
