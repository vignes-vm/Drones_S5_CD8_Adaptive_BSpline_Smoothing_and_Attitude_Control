"""
Standalone Native 3D Graphics GUI Application for PyBullet UAV Simulation.
Opens an INSTANT 3D OpenGL-style rendering window on Windows using native Python GUI,
rendering real-time 3D flight trajectory, 3D obstacle blocks, quadrotor model with spinning rotors,
and interactive mouse orbit/zoom controls.

Usage:
  py -3.12 pybullet/interactive_3d_window.py
"""
import os
import sys
import time
import math
import numpy as np
import tkinter as tk
from tkinter import ttk

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.bspline import BSplinePath
from pybullet.src.quaternion_math import QuaternionMath
from pybullet.src.trajectory_tracker import PyBulletTrajectoryTracker


class Quadrotor3DWindow(tk.Tk):

    def __init__(self, seed=42):
        super().__init__()

        self.title("PyBullet 3D UAV Flight Simulation — Native 3D Window")
        self.geometry("1100x720")
        self.configure(bg="#0f172a")

        # 1. Generate Environment & Trajectory
        self.env_3d = Environment3D()
        self.env_3d.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=seed)

        astar = AStarPlanner3D(self.env_3d, grid_resolution=50)
        raw_wp = np.array(astar.plan(self.env_3d.start, self.env_3d.goal))
        baseline_wp = np.array(astar.smooth_waypoints(raw_wp, min_distance=100))
        self.spline = BSplinePath(baseline_wp, param_mode='uniform')
        self.path_len = self.spline.path_length()
        self.spline_pts = self.spline.sample_path(200)

        # 2. Simulation Physics State
        self.dt_sim = 0.005
        self.flight_speed = 25.0
        self.u_current = 0.001
        self.dt_u = (self.flight_speed / max(1.0, self.path_len)) * self.dt_sim

        # Physical 6-DOF Quadrotor State
        self.pos = np.array(self.env_3d.start, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega = np.zeros(3, dtype=float)
        self.mass = 1.0
        self.g = 9.81
        self.inertia = np.diag([6.5e-4, 6.5e-4, 1.2e-3])
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.motor_w = np.array([280.0, 280.0, 280.0, 280.0])

        self.tracker = PyBulletTrajectoryTracker(
            mass=self.mass, g=self.g, Pq=2.5, Pw=0.08, tau_limit=1.0,
            Kp_pos=[1.2, 1.2, 2.5], Kd_pos=[1.5, 1.5, 2.0]
        )

        # 3. Camera 3D Orbit Parameters
        self.cam_angle_x = 0.7
        self.cam_angle_y = 0.4
        self.cam_zoom = 0.4
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.is_playing = True
        self.trail_history = []

        # 4. Construct GUI Layout
        self._build_ui()

        # Mouse Binds for 3D Camera Orbit & Zoom
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)

        # 5. Start 60 FPS Animation Loop
        self.after(20, self._simulation_loop)

    def _build_ui(self):
        # Header Bar
        header = tk.Frame(self, bg="#1e293b", height=50)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header, text="🛸 PyBullet 3D Real-Time UAV Flight Viewport",
            font=("Segoe UI", 14, "bold"), fg="#f8fafc", bg="#1e293b"
        )
        title_lbl.pack(side=tk.LEFT, padx=15, pady=10)

        sub_lbl = tk.Label(
            header, text="Drag mouse to orbit 3D view • Scroll to zoom",
            font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b"
        )
        sub_lbl.pack(side=tk.RIGHT, padx=15, pady=10)

        # Control Toolbar
        toolbar = tk.Frame(self, bg="#0f172a", height=45)
        toolbar.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=8)

        self.play_btn = tk.Button(
            toolbar, text="⏸ Pause", font=("Segoe UI", 10, "bold"),
            fg="white", bg="#3b82f6", activebackground="#2563eb",
            relief=tk.FLAT, command=self._toggle_play, padx=12, pady=4
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        reset_btn = tk.Button(
            toolbar, text="↺ Reset Flight", font=("Segoe UI", 10),
            fg="#f8fafc", bg="#334155", activebackground="#475569",
            relief=tk.FLAT, command=self._reset_flight, padx=12, pady=4
        )
        reset_btn.pack(side=tk.LEFT, padx=5)

        rand_btn = tk.Button(
            toolbar, text="🎲 Randomize Map", font=("Segoe UI", 10),
            fg="#f8fafc", bg="#059669", activebackground="#047857",
            relief=tk.FLAT, command=self._randomize_map, padx=12, pady=4
        )
        rand_btn.pack(side=tk.LEFT, padx=5)

        # Progress Slider
        self.slider_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(
            toolbar, from_=0.0, to=1.0, variable=self.slider_var,
            orient=tk.HORIZONTAL, command=self._on_slider
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15)

        self.hud_text = tk.Label(
            toolbar, text="Pos: (0, 0, 20) | Motor RPM: 280",
            font=("Consolas", 10), fg="#38bdf8", bg="#0f172a"
        )
        self.hud_text.pack(side=tk.RIGHT, padx=10)

        # 3D Graphics Canvas
        self.canvas = tk.Canvas(self, bg="#020617", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

    def _on_mouse_down(self, event):
        self.is_dragging = True
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_mouse_drag(self, event):
        if not self.is_dragging: return
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.cam_angle_x += dx * 0.005
        self.cam_angle_y = max(0.05, min(math.pi / 2 - 0.05, self.cam_angle_y + dy * 0.005))
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def _on_mouse_up(self, event):
        self.is_dragging = False

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self.cam_zoom = min(1.5, self.cam_zoom * 1.1)
        else:
            self.cam_zoom = max(0.1, self.cam_zoom / 1.1)

    def _project_3d(self, x, y, z):
        w = self.canvas.winfo_width() or 1000
        h = self.canvas.winfo_height() or 600

        # Center focus at middle of map
        cx, cy, cz = 1000.0, 900.0, 100.0
        dx = x - cx
        dy = y - cy
        dz = z - cz

        cosX, sinX = math.cos(self.cam_angle_x), math.sin(self.cam_angle_x)
        cosY, sinY = math.cos(self.cam_angle_y), math.sin(self.cam_angle_y)

        x1 = dx * cosX - dy * sinX
        y1 = dx * sinX + dy * cosX
        z1 = dz

        y2 = y1 * cosY - z1 * sinY
        z2 = y1 * sinY + z1 * cosY

        scale = (500.0 * self.cam_zoom) / (z2 + 2500.0)
        px = w / 2.0 + x1 * scale
        py = h / 2.0 - y2 * scale

        return px, py

    def _simulation_loop(self):
        if self.is_playing:
            # Step physics 4 times per frame for 200 Hz integration rate
            for _ in range(4):
                pos_des = self.spline.evaluate(self.u_current)
                tangent = self.spline.evaluate_derivative(self.u_current, 1)
                tang_norm = np.linalg.norm(tangent)
                v_des = (tangent / tang_norm * self.flight_speed) if tang_norm > 1e-4 else np.zeros(3)

                thrust, torques, q_ref, _, _ = self.tracker.step_control(
                    pos_des, v_des, self.pos, self.vel, self.quat, self.omega, tangent=tangent
                )

                # Rigid body dynamics step
                R = QuaternionMath.to_rotation_matrix(self.quat)
                acc = (R @ np.array([0, 0, thrust]) / self.mass) + np.array([0, 0, -self.g])
                self.vel += acc * self.dt_sim
                self.pos += self.vel * self.dt_sim

                alpha = self.inv_inertia @ (torques - np.cross(self.omega, self.inertia @ self.omega))
                self.omega += alpha * self.dt_sim
                w_quat = np.array([0.0, self.omega[0], self.omega[1], self.omega[2]])
                dq = 0.5 * QuaternionMath.multiply(self.quat, w_quat)
                self.quat = QuaternionMath.normalize(self.quat + dq * self.dt_sim)

                if self.u_current < 0.999:
                    self.u_current = min(0.999, self.u_current + self.dt_u)

            self.trail_history.append(self.pos.copy())
            if len(self.trail_history) > 300:
                self.trail_history.pop(0)

            self.slider_var.set(self.u_current)

        self._render_scene()
        self.after(20, self._simulation_loop)

    def _render_scene(self):
        self.canvas.delete("all")

        # 1. Ground Plane Grid
        for g in range(0, 2001, 400):
            p1 = self._project_3d(g, 0, 0)
            p2 = self._project_3d(g, 1800, 0)
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#1e293b", width=1)

            p3 = self._project_3d(0, g, 0)
            p4 = self._project_3d(2000, g, 0)
            self.canvas.create_line(p3[0], p3[1], p4[0], p4[1], fill="#1e293b", width=1)

        # 2. Render 3D Obstacles
        for obs in self.env_3d.obstacles:
            if obs.shape == 'cylinder':
                r, h = obs.dimensions
                b_px = self._project_3d(obs.position[0], obs.position[1], 0)
                t_px = self._project_3d(obs.position[0], obs.position[1], h)
                self.canvas.create_line(b_px[0], b_px[1], t_px[0], t_px[1], fill="#10b981", width=3)
                self.canvas.create_oval(
                    b_px[0]-10, b_px[1]-5, b_px[0]+10, b_px[1]+5,
                    outline="#10b981", fill="#065f46"
                )
            elif obs.shape == 'cuboid':
                l, w, h = obs.dimensions
                b_px = self._project_3d(obs.position[0], obs.position[1], 0)
                t_px = self._project_3d(obs.position[0], obs.position[1], h)
                self.canvas.create_line(b_px[0], b_px[1], t_px[0], t_px[1], fill="#3b82f6", width=3)
                self.canvas.create_rectangle(
                    b_px[0]-12, b_px[1]-6, b_px[0]+12, b_px[1]+6,
                    outline="#3b82f6", fill="#1e3a8a"
                )

        # 3. B-Spline Path Curve
        for i in range(len(self.spline_pts) - 1):
            p1 = self._project_3d(*self.spline_pts[i])
            p2 = self._project_3d(*self.spline_pts[i+1])
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#ef4444", width=2, dash=(4, 2))

        # 4. Trajectory Trail
        if len(self.trail_history) > 1:
            for i in range(len(self.trail_history) - 1):
                p1 = self._project_3d(*self.trail_history[i])
                p2 = self._project_3d(*self.trail_history[i+1])
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#10b981", width=3)

        # 5. Render Quadrotor Model & Spin Rotors
        dr_px = self._project_3d(*self.pos)
        arm = 18
        rot_angle = time.time() * 30.0

        # Arms
        self.canvas.create_line(dr_px[0]-arm, dr_px[1]-arm/2, dr_px[0]+arm, dr_px[1]+arm/2, fill="#38bdf8", width=3)
        self.canvas.create_line(dr_px[0]-arm, dr_px[1]+arm/2, dr_px[0]+arm, dr_px[1]-arm/2, fill="#38bdf8", width=3)

        # Center body
        self.canvas.create_oval(dr_px[0]-6, dr_px[1]-6, dr_px[0]+6, dr_px[1]+6, fill="#0284c7", outline="#38bdf8")

        # 4 Spinning Rotors
        rotors = [
            (dr_px[0]-arm, dr_px[1]-arm/2), (dr_px[0]+arm, dr_px[1]+arm/2),
            (dr_px[0]-arm, dr_px[1]+arm/2), (dr_px[0]+arm, dr_px[1]-arm/2)
        ]
        for rx, ry in rotors:
            r_sz = 8 + math.sin(rot_angle) * 2
            self.canvas.create_oval(rx-r_sz, ry-r_sz/2, rx+r_sz, ry+r_sz/2, outline="#38bdf8", fill="#0284c7")

        # Start & Goal Markers
        s_px = self._project_3d(*self.env_3d.start)
        g_px = self._project_3d(*self.env_3d.goal)
        self.canvas.create_oval(s_px[0]-8, s_px[1]-8, s_px[0]+8, s_px[1]+8, fill="#22c55e", outline="white")
        self.canvas.create_text(s_px[0], s_px[1]-14, text="START", fill="#22c55e", font=("Segoe UI", 9, "bold"))

        self.canvas.create_oval(g_px[0]-8, g_px[1]-8, g_px[0]+8, g_px[1]+8, fill="#ef4444", outline="white")
        self.canvas.create_text(g_px[0], g_px[1]-14, text="GOAL", fill="#ef4444", font=("Segoe UI", 9, "bold"))

        # HUD Text
        eul = QuaternionMath.to_euler(self.quat)
        self.hud_text.config(
            text=f"Pos: ({self.pos[0]:.0f}, {self.pos[1]:.0f}, {self.pos[2]:.0f})m | "
                 f"Att: R:{math.degrees(eul[0]):.1f}° P:{math.degrees(eul[1]):.1f}° Y:{math.degrees(eul[2]):.1f}°"
        )

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="▶ Play" if not self.is_playing else "⏸ Pause")

    def _reset_flight(self):
        self.pos = np.array(self.env_3d.start, dtype=float)
        self.vel.fill(0.0)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega.fill(0.0)
        self.u_current = 0.001
        self.trail_history.clear()
        self.is_playing = True
        self.play_btn.config(text="⏸ Pause")

    def _on_slider(self, val):
        self.u_current = float(val)
        self.pos = self.spline.evaluate(self.u_current)

    def _randomize_map(self):
        new_seed = int(np.random.randint(1, 99999))
        print(f"Randomizing 3D obstacles with seed={new_seed}...")
        self.env_3d = Environment3D()
        self.env_3d.add_random_obstacles(n_cylinders=5, n_cuboids=5, seed=new_seed)

        astar = AStarPlanner3D(self.env_3d, grid_resolution=50)
        raw_wp = np.array(astar.plan(self.env_3d.start, self.env_3d.goal))
        baseline_wp = np.array(astar.smooth_waypoints(raw_wp, min_distance=100))
        self.spline = BSplinePath(baseline_wp, param_mode='uniform')
        self.path_len = self.spline.path_length()
        self.spline_pts = self.spline.sample_path(200)
        self.dt_u = (self.flight_speed / max(1.0, self.path_len)) * self.dt_sim

        self._reset_flight()


if __name__ == '__main__':
    app = Quadrotor3DWindow()
    app.mainloop()
