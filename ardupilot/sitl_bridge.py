"""
ArduPilot SITL MAVLink Companion Bridge.

Connects the 3D A* + Adaptive B-Spline Path Smoother and Quaternion Attitude Guidance
to an ArduPilot SITL (Software In The Loop) simulation instance over MAVLink.

Usage:
    # Run against live SITL on default MAVProxy port (UDP 14550 or 14551):
    python ardupilot/sitl_bridge.py --connection udpin:127.0.0.1:14550 --mode attitude

    # Run in dry-run mode (tests pipeline & plots without requiring SITL execution):
    python ardupilot/sitl_bridge.py --dry-run
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False

from environment.obstacles import Environment3D
from path_planning.astar import AStarPlanner3D
from path_planning.path_smoother import AdaptiveBSplineSmoother
from integration.trajectory_converter import TrajectoryConverter
from attitude_control.quaternion import Quaternion


class ArduPilotBridge:
    """
    Companion bridge between the B-Spline Path Smoother / Quaternion Guidance
    and an ArduPilot SITL quadrotor instance via MAVLink.
    """

    def __init__(self, connection_string="udpin:0.0.0.0:14550", target_system=1, target_component=1):
        self.connection_string = connection_string
        self.target_system = target_system
        self.target_component = target_component
        self.master = None

        # Telemetry & state tracking
        self.current_pos_ned = np.array([0.0, 0.0, 0.0])  # North, East, Down (m)
        self.current_q = Quaternion(1.0, 0.0, 0.0, 0.0)    # w, x, y, z
        self.current_attitude_euler = (0.0, 0.0, 0.0)      # roll, pitch, yaw (rad)
        self.current_rates = np.array([0.0, 0.0, 0.0])     # wx, wy, wz (rad/s)

        # Guidance configuration
        self.error_threshold = 0.50  # Max attitude error (rad) to advance u (0.5 rad ~ 28 deg)
        self.u_step = 0.005           # Parameter increment per control step
        self.target_thrust = 0.55     # Base hover thrust [0, 1]

    def connect(self, timeout=10.0):
        """Establish MAVLink connection to SITL."""
        if not PYMAVLINK_AVAILABLE:
            print("[ArduPilotBridge] Error: pymavlink is not installed.")
            return False

        print(f"[ArduPilotBridge] Connecting to SITL on {self.connection_string}...")
        try:
            self.master = mavutil.mavlink_connection(self.connection_string)
            print("[ArduPilotBridge] Waiting for MAVLink heartbeat...")
            heartbeat = self.master.wait_heartbeat(timeout=timeout)
            if heartbeat is None:
                print(f"[ArduPilotBridge] Connection timed out after {timeout}s.")
                return False

            self.target_system = self.master.target_system
            self.target_component = self.master.target_component
            print(f"[ArduPilotBridge] Connected to System {self.target_system}, Component {self.target_component}")
            return True
        except Exception as e:
            print(f"[ArduPilotBridge] Connection failed: {e}")
            return False

    def set_mode(self, mode_name="GUIDED"):
        """Change vehicle flight mode."""
        if not self.master:
            return False
        mode_id = self.master.mode_mapping().get(mode_name)
        if mode_id is None:
            print(f"[ArduPilotBridge] Unknown mode: {mode_name}")
            return False
        self.master.mav.set_mode_send(
            self.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"[ArduPilotBridge] Commanded mode change to {mode_name}")
        return True

    def arm_vehicle(self, arm=True):
        """Arm or disarm the vehicle."""
        if not self.master:
            return False
        cmd = 1 if arm else 0
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            cmd, 0, 0, 0, 0, 0, 0
        )
        action = "Arm" if arm else "Disarm"
        print(f"[ArduPilotBridge] Commanded {action}")
        return True

    def takeoff(self, altitude=20.0):
        """Command takeoff to specified target altitude (m)."""
        if not self.master:
            return False
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0, altitude
        )
        print(f"[ArduPilotBridge] Commanded takeoff to {altitude}m")
        return True

    def update_telemetry(self):
        """Fetch latest MAVLink telemetry messages (ATTITUDE, ATTITUDE_QUATERNION, LOCAL_POSITION_NED)."""
        if not self.master:
            return

        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break

            msg_type = msg.get_type()
            if msg_type == "ATTITUDE_QUATERNION":
                # q1, q2, q3, q4 correspond to w, x, y, z
                self.current_q = Quaternion(msg.q1, msg.q2, msg.q3, msg.q4)
                self.current_rates = np.array([getattr(msg, 'rollspeed', 0.0), getattr(msg, 'pitchspeed', 0.0), getattr(msg, 'yawspeed', 0.0)])
            elif msg_type == "ATTITUDE":
                self.current_attitude_euler = (msg.roll, msg.pitch, msg.yaw)
                self.current_rates = np.array([getattr(msg, 'rollspeed', 0.0), getattr(msg, 'pitchspeed', 0.0), getattr(msg, 'yawspeed', 0.0)])
                # Convert Euler to Quaternion if ATTITUDE_QUATERNION isn't emitted
                self.current_q = Quaternion.from_euler(msg.roll, msg.pitch, msg.yaw)
            elif msg_type == "LOCAL_POSITION_NED":
                # MAVLink NED: x=North, y=East, z=Down (negative altitude)
                self.current_pos_ned = np.array([msg.x, msg.y, msg.z])

    def send_attitude_target(self, ref_q: Quaternion, thrust=0.55):
        """
        Stream MAVLink SET_ATTITUDE_TARGET command (#82).
        Commands reference quaternion q_ref and collective thrust.
        """
        if not self.master:
            return

        # type_mask = 7 -> ignore body roll/pitch/yaw rates, use attitude quaternion + thrust
        type_mask = 0b00000111
        q_array = [ref_q.q0, ref_q.q1, ref_q.q2, ref_q.q3]

        self.master.mav.set_attitude_target_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.target_system,
            self.target_component,
            type_mask,
            q_array,
            0.0, 0.0, 0.0,  # body rates ignored
            thrust
        )

    def send_position_target(self, pos_xyz, vel_xyz=None, yaw=0.0):
        """
        Stream MAVLink SET_POSITION_TARGET_LOCAL_NED command (#84).
        Converts internal XYZ (East-North-Up or standard meters) to MAVLink NED.
        """
        if not self.master:
            return

        # Map internal XYZ to NED: North = Y, East = X, Down = -Z
        pos_ned = [pos_xyz[1], pos_xyz[0], -pos_xyz[2]]

        # Type mask: 0x0DF8 -> Position + Yaw enabled, Velocity/Accel ignored
        type_mask = 0b0000110111111000

        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            pos_ned[0], pos_ned[1], pos_ned[2],
            0.0, 0.0, 0.0,  # vx, vy, vz
            0.0, 0.0, 0.0,  # afx, afy, afz
            yaw, 0.0
        )

    def run_guidance_loop(self, spline_smoother, waypoints, mode="attitude", duration=25.0, dt=0.05, dry_run=False):
        """
        Main execution loop for ArduPilot guidance.
        Generates smooth B-Spline path and streams reference commands to SITL.
        """
        print(f"\n[ArduPilotBridge] Starting Guidance Loop (mode={mode}, dry_run={dry_run})...")

        # 1. Fit B-Spline path
        spline = spline_smoother.smooth(waypoints)
        converter = TrajectoryConverter()

        # Logging buffers
        log_t = []
        log_u = []
        log_att_err = []
        log_q_ref = []
        log_q_meas = []
        log_pos_ref = []
        log_pos_meas = []

        u = 0.0
        sim_time = 0.0
        n_steps = int(duration / dt)

        # Mock initial quad state for dry-run mode
        mock_q = Quaternion(1.0, 0.0, 0.0, 0.0)
        mock_pos = np.copy(waypoints[0])

        for step in range(n_steps):
            t_now = step * dt
            log_t.append(t_now)

            # Evaluate B-Spline target reference at u
            pos_ref, ref_q = converter.get_reference_at(spline, u)

            if not dry_run and self.master:
                # Update telemetry from live SITL
                self.update_telemetry()
                q_meas = self.current_q
                # Convert NED back to internal XYZ for position logging: X=East, Y=North, Z=-Down
                pos_meas = np.array([self.current_pos_ned[1], self.current_pos_ned[0], -self.current_pos_ned[2]])

                # Send command to SITL
                if mode == "attitude":
                    self.send_attitude_target(ref_q, thrust=self.target_thrust)
                else:
                    yaw = ref_q.to_euler()[2]
                    self.send_position_target(pos_ref, yaw=yaw)

                time.sleep(dt)  # Real-time synchronization
            else:
                # Dry-run mode: simulate simplified first-order tracking dynamics
                q_meas = mock_q
                pos_meas = mock_pos

                # Update mock state toward reference
                mock_q = TrajectoryConverter.slerp_static(mock_q, ref_q, t=0.15)
                mock_pos = mock_pos + 0.15 * (pos_ref - mock_pos)

            # Compute attitude error q_err = q_ref x q_meas^-1
            q_err = ref_q.error(q_meas)
            err_vec_norm = np.linalg.norm([q_err.q1, q_err.q2, q_err.q3])

            # Gate path parameter u based on attitude tracking error
            if err_vec_norm < self.error_threshold:
                u = min(1.0, u + self.u_step)

            # Log metrics
            log_u.append(u)
            log_att_err.append(err_vec_norm)
            log_q_ref.append([ref_q.q0, ref_q.q1, ref_q.q2, ref_q.q3])
            log_q_meas.append([q_meas.q0, q_meas.q1, q_meas.q2, q_meas.q3])
            log_pos_ref.append(pos_ref)
            log_pos_meas.append(pos_meas)

            if step % 50 == 0 or step == n_steps - 1:
                progress = u * 100.0
                print(f"  t={t_now:5.1f}s | u={u:5.3f} ({progress:4.1f}%) | AttErr={err_vec_norm:6.4f} | Pos=[{pos_meas[0]:6.1f}, {pos_meas[1]:6.1f}, {pos_meas[2]:6.1f}]m")

            if u >= 1.0:
                print(f"[ArduPilotBridge] Path complete (u=1.0) at t={t_now:.2f}s!")
                break

        # Plot & Save Execution Summary
        self.save_plots(log_t, log_u, log_att_err, log_q_ref, log_q_meas, log_pos_ref, log_pos_meas)
        print("[ArduPilotBridge] Guidance loop finished successfully.")

    def save_plots(self, log_t, log_u, log_att_err, log_q_ref, log_q_meas, log_pos_ref, log_pos_meas):
        """Generate and save performance tracking plots."""
        os.makedirs("ardupilot/results", exist_ok=True)
        log_t = np.array(log_t)
        log_pos_ref = np.array(log_pos_ref)
        log_pos_meas = np.array(log_pos_meas)
        log_q_ref = np.array(log_q_ref)
        log_q_meas = np.array(log_q_meas)

        # Plot 1: 3D Path Tracking
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(log_pos_ref[:, 0], log_pos_ref[:, 1], log_pos_ref[:, 2], 'r--', label='B-Spline Reference')
        ax.plot(log_pos_meas[:, 0], log_pos_meas[:, 1], log_pos_meas[:, 2], 'b-', label='ArduPilot Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('ArduPilot SITL Flight - B-Spline Path Tracking')
        ax.legend()
        plt.tight_layout()
        plt.savefig('ardupilot/results/ardupilot_3d_flight.png', dpi=200)
        plt.close()

        # Plot 2: Path Progress & Attitude Error Convergence
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(log_t, log_u, 'g-', lw=2)
        ax1.set_ylabel('Path Parameter u')
        ax1.set_title('ArduPilot SITL - Path Progress & Quaternion Attitude Error')
        ax1.grid(True)

        ax2.plot(log_t, log_att_err, 'r-', lw=1.5)
        ax2.axhline(self.error_threshold, color='k', linestyle='--', label=f'Threshold ({self.error_threshold} rad)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('||q_err_vec|| (rad)')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.savefig('ardupilot/results/ardupilot_progress_and_error.png', dpi=200)
        plt.close()

        print("Saved plots to ardupilot/results/ directory.")


def main():
    parser = argparse.ArgumentParser(description="ArduPilot SITL MAVLink Companion Bridge")
    parser.add_argument("--connection", type=str, default="udpin:0.0.0.0:14550", help="MAVLink connection string")
    parser.add_argument("--mode", type=str, choices=["attitude", "position"], default="attitude", help="Guidance command mode")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration (s)")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode without live SITL")
    args = parser.parse_args()

    # Build environment & sample path
    env = Environment3D()
    planner = AStarPlanner3D(env, grid_resolution=50.0)
    raw_path = planner.plan(env.start, env.goal)
    waypoints = planner.smooth_waypoints(raw_path)

    smoother = AdaptiveBSplineSmoother(n_inner_knots=6, kappa_max=0.5)

    bridge = ArduPilotBridge(connection_string=args.connection)

    if not args.dry_run:
        connected = bridge.connect(timeout=5.0)
        if not connected:
            print("\n[Notice] Could not connect to live SITL instance. Falling back to DRY-RUN mode.")
            args.dry_run = True
        else:
            print("\n[ArduPilotBridge] Arming vehicle & taking off to 20m...")
            bridge.set_mode("GUIDED")
            time.sleep(0.5)
            bridge.arm_vehicle(True)
            time.sleep(0.5)
            bridge.takeoff(20.0)
            time.sleep(2.0)  # Wait briefly for takeoff init

    bridge.run_guidance_loop(smoother, waypoints, mode=args.mode, duration=args.duration, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
