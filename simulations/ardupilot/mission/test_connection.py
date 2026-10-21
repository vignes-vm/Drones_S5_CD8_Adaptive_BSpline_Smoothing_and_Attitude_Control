#!/usr/bin/env python3
"""
Prompt 2 — SITL MAVLink connection test.

Starts ArduCopter SITL in the background (via sim_vehicle.py), exposes MAVLink on
udp:127.0.0.1:14550 (via a headless MAVProxy hub), connects with pymavlink, prints
the vehicle's mode / armed status / GPS position, then cleanly shuts everything down.

Run:  ~/ardupilot_sim/venv/bin/python ~/ardupilot_sim/mission/test_connection.py
"""
import os
import sys
import time
import socket
import signal
import subprocess

HOME       = os.path.expanduser("~/ardupilot_sim")
ARDUPILOT  = os.path.join(HOME, "ardupilot")
VENV_BIN   = os.path.join(HOME, "venv", "bin")
CONN       = "udp:127.0.0.1:14550"          # endpoint the test connects to
SITL_TCP   = ("127.0.0.1", 5760)            # SITL primary MAVLink port

from pymavlink import mavutil

procs = []


def _port_open(host_port, timeout=1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex(host_port) == 0


def cleanup():
    """Terminate SITL + MAVProxy and any children sim_vehicle spawned."""
    for p in procs:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
    time.sleep(1.5)
    for p in procs:
        if p.poll() is None:
            p.kill()
    # sim_vehicle.py forks the arducopter binary; make sure it's gone
    for pat in ("sim_vehicle.py", "build/sitl/bin/arducopter", "mavproxy.py"):
        subprocess.run(["pkill", "-f", pat],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    env = os.environ.copy()
    env["PATH"] = VENV_BIN + os.pathsep + env.get("PATH", "")

    print("[test] Starting ArduCopter SITL (sim_vehicle.py, headless)...")
    sitl = subprocess.Popen(
        [sys.executable, os.path.join(ARDUPILOT, "Tools/autotest/sim_vehicle.py"),
         "-v", "ArduCopter", "--no-mavproxy", "--no-rebuild"],
        cwd=ARDUPILOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(sitl)

    # wait for SITL to open its primary MAVLink TCP port
    print("[test] Waiting for SITL on tcp:5760 ...")
    for _ in range(60):
        if _port_open(SITL_TCP):
            break
        time.sleep(1)
    else:
        print("[test] FAIL: SITL did not open tcp:5760 in time")
        return 1
    print("[test] SITL is up.")

    print(f"[test] Starting MAVProxy hub -> {CONN} ...")
    mav = subprocess.Popen(
        [os.path.join(VENV_BIN, "mavproxy.py"),
         "--master=tcp:127.0.0.1:5760", f"--out={CONN}", "--daemon"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(mav)

    print(f"[test] Connecting pymavlink to {CONN} ...")
    m = mavutil.mavlink_connection(CONN, source_system=200)
    hb = m.wait_heartbeat(timeout=60)
    if hb is None:
        print("[test] FAIL: no heartbeat received")
        return 1
    print(f"[test] HEARTBEAT ok — system {m.target_system}, component {m.target_component}")

    # ask for telemetry streams so GPS/position arrive
    m.mav.request_data_stream_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

    # collect a GPS/position fix — wait for a real 3D lock (SITL EKF needs
    # ~20-30 s after boot to acquire GPS and set home)
    print("[test] Waiting for GPS lock (fix_type >= 3)...")
    gps, pos = None, None
    deadline = time.time() + 60
    while time.time() < deadline:
        msg = m.recv_match(blocking=True, timeout=5)
        if msg is None:
            continue
        t = msg.get_type()
        if t == "GPS_RAW_INT":
            gps = msg
        elif t == "GLOBAL_POSITION_INT":
            pos = msg
        if gps is not None and gps.fix_type >= 3 and pos is not None and pos.lat != 0:
            break

    print("\n=========== VEHICLE STATE ===========")
    print(f" Mode        : {m.flightmode}")
    print(f" Armed       : {bool(m.motors_armed())}")
    if gps is not None:
        print(f" GPS fix     : {gps.fix_type} ({gps.satellites_visible} sats)")
        print(f" GPS lat/lon : {gps.lat/1e7:.7f}, {gps.lon/1e7:.7f}")
    if pos is not None:
        print(f" Altitude    : {pos.alt/1000:.1f} m (AMSL), {pos.relative_alt/1000:.1f} m (rel)")
    print("=====================================\n")

    print("[test] Closing MAVLink connection.")
    m.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        print("[test] Stopping SITL + MAVProxy ...")
        cleanup()
        print("[test] Clean shutdown complete.")
    sys.exit(rc)
