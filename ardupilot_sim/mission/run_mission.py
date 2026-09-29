#!/usr/bin/env python3
"""
Prompt 4 — fly the B-Spline mission end-to-end in ArduCopter SITL and log telemetry.

Flow:
  1. start SITL (subprocess) at the same home as generate_mission.py
  2. connect MAVLink, wait heartbeat
  3. GUIDED -> arm -> takeoff (MAV_CMD_NAV_TAKEOFF) to first waypoint altitude
  4. upload bspline_mission.waypoints (MISSION_COUNT / MISSION_ITEM protocol)
  5. AUTO -> fly the mission autonomously
  6. poll + log at 5 Hz into logs/flight_log_<timestamp>.json
  7. detect completion (final MISSION_ITEM_REACHED or disarm), stop logging
  8. LAND, disarm, terminate SITL
  9. print summary (time, waypoints reached, max alt, unexpected failsafe/mode changes)

Run:  ~/ardupilot_sim/venv/bin/python ~/ardupilot_sim/mission/run_mission.py
"""
import os
import sys
import time
import json
import signal
import socket
import subprocess
from datetime import datetime, timezone

from pymavlink import mavutil

HERE      = os.path.dirname(os.path.abspath(__file__))
HOME_ROOT = os.path.expanduser("~/ardupilot_sim")
ARDUPILOT = os.path.join(HOME_ROOT, "ardupilot")
VENV_BIN  = os.path.join(HOME_ROOT, "venv", "bin")
LOG_DIR   = os.path.join(HOME_ROOT, "logs")
WP_FILE   = os.path.join(HERE, "bspline_mission.waypoints")

# same home as generate_mission.py
HOME_LAT, HOME_LON = 13.0827, 80.2707
SITL_TCP = ("127.0.0.1", 5760)
CONN     = "tcp:127.0.0.1:5760"
SPEEDUP  = 10           # SITL faster-than-real-time to keep the run short
WPNAV_SPEED = 1500      # cm/s (15 m/s) cruise for the mission

procs = []


def port_open(hp, timeout=1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex(hp) == 0


def cleanup():
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
    for pat in ("sim_vehicle.py", "build/sitl/bin/arducopter", "mavproxy.py"):
        subprocess.run(["pkill", "-f", pat],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_wpl(path):
    """Parse a QGC WPL 110 file into a list of item dicts."""
    items = []
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines[0].startswith("QGC WPL 110"), "not a QGC WPL 110 file"
    for ln in lines[1:]:
        c = ln.split("\t")
        items.append(dict(seq=int(c[0]), current=int(c[1]), frame=int(c[2]),
                          command=int(c[3]),
                          p1=float(c[4]), p2=float(c[5]), p3=float(c[6]), p4=float(c[7]),
                          x=float(c[8]), y=float(c[9]), z=float(c[10]),
                          autocontinue=int(c[11])))
    return items


def upload_mission(m, items, timeout=40):
    """Upload via MISSION_COUNT / MISSION_REQUEST(_INT) / MISSION_ITEM_INT / MISSION_ACK."""
    print(f"[run] Uploading {len(items)} mission items...")
    m.mav.mission_count_send(m.target_system, m.target_component, len(items))
    deadline = time.time() + timeout
    acked = False
    while time.time() < deadline:
        msg = m.recv_match(type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"],
                           blocking=True, timeout=5)
        if msg is None:
            continue
        t = msg.get_type()
        if t == "MISSION_ACK":
            acked = True
            print(f"[run] Mission upload ACK: type={msg.type}")
            break
        seq = msg.seq
        it = items[seq]
        m.mav.mission_item_int_send(
            m.target_system, m.target_component, seq, it["frame"], it["command"],
            0, it["autocontinue"], it["p1"], it["p2"], it["p3"], it["p4"],
            int(round(it["x"] * 1e7)), int(round(it["y"] * 1e7)), it["z"])
    if not acked:
        raise RuntimeError("mission upload not ACKed")


def set_mode(m, mode):
    mode_id = m.mode_mapping()[mode]
    m.mav.set_mode_send(m.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)


def wait_mode(m, mode, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if m.flightmode == mode:
            return True
    return False


def cmd_long(m, command, *params):
    p = list(params) + [0.0] * (7 - len(params))
    m.mav.command_long_send(m.target_system, m.target_component, command, 0, *p)


def wait_armable_and_arm(m, timeout=90):
    print("[run] Waiting for EKF/GPS ready, then arming...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        set_mode(m, "GUIDED")
        m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        cmd_long(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=2)
        m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if m.motors_armed():
            print("[run] Armed.")
            return True
        time.sleep(1)
    return False


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    items = parse_wpl(WP_FILE)
    last_wp_seq = items[-1]["seq"]
    # first real waypoint altitude (skip home item 0)
    takeoff_alt = next(it["z"] for it in items if it["seq"] >= 1)

    env = os.environ.copy()
    env["PATH"] = VENV_BIN + os.pathsep + env.get("PATH", "")

    print("[run] Starting SITL...")
    sitl = subprocess.Popen(
        [sys.executable, os.path.join(ARDUPILOT, "Tools/autotest/sim_vehicle.py"),
         "-v", "ArduCopter", "--no-mavproxy", "--no-rebuild",
         f"--custom-location={HOME_LAT},{HOME_LON},0,0", "--speedup", str(SPEEDUP)],
        cwd=ARDUPILOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(sitl)

    for _ in range(60):
        if port_open(SITL_TCP):
            break
        time.sleep(1)
    else:
        print("[run] FAIL: SITL did not start")
        return 1

    m = mavutil.mavlink_connection(CONN, source_system=200)
    m.wait_heartbeat(timeout=30)
    print(f"[run] Connected (sys {m.target_system}). Requesting streams.")
    m.mav.request_data_stream_send(m.target_system, m.target_component,
                                   mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1)
    # cruise speed param
    m.mav.param_set_send(m.target_system, m.target_component, b"WPNAV_SPEED",
                         float(WPNAV_SPEED), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    # wait for a GPS fix so arming/EKF is happy
    t0 = time.time()
    while time.time() - t0 < 60:
        g = m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=5)
        if g and g.fix_type >= 3:
            break

    # upload the mission (before flight)
    upload_mission(m, items)

    # GUIDED -> arm -> takeoff
    if not wait_armable_and_arm(m):
        print("[run] FAIL: could not arm")
        return 1
    print(f"[run] Takeoff to {takeoff_alt:.0f} m ...")
    cmd_long(m, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, takeoff_alt)
    # wait until near takeoff altitude
    t0 = time.time()
    while time.time() - t0 < 60:
        p = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=3)
        if p and p.relative_alt / 1000.0 >= takeoff_alt * 0.95:
            break
    print("[run] Reached takeoff altitude. Switching to AUTO.")
    set_mode(m, "AUTO")
    wait_mode(m, "AUTO", timeout=10)
    # copter also needs a mission-start nudge in some cases
    cmd_long(m, mavutil.mavlink.MAV_CMD_MISSION_START, 0, last_wp_seq)

    # ---- 5 Hz logging loop ----
    log = {"metadata": {"start_time": datetime.now(timezone.utc).isoformat(),
                        "mission_file": os.path.basename(WP_FILE),
                        "vehicle": "ArduCopter SITL",
                        "home": {"lat": HOME_LAT, "lon": HOME_LON},
                        "speedup": SPEEDUP},
           "telemetry": []}
    mission_start = time.time()
    state = dict(lat=0.0, lon=0.0, alt=0.0, roll=0.0, pitch=0.0, yaw=0.0,
                 vx=0.0, vy=0.0, vz=0.0, groundspeed=0.0, current_wp=0,
                 mode=m.flightmode, armed=bool(m.motors_armed()))
    mode_changes = []
    failsafe_events = []
    reached = set()
    done = False
    was_armed = True
    got_position = False   # don't log until the first real GLOBAL_POSITION_INT
    max_run = 400  # wall-clock safety cap (s)

    while not done and (time.time() - mission_start) < max_run:
        # drain all pending messages, update state
        while True:
            msg = m.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "GLOBAL_POSITION_INT":
                state.update(lat=msg.lat / 1e7, lon=msg.lon / 1e7,
                             alt=msg.relative_alt / 1000.0,
                             vx=msg.vx / 100.0, vy=msg.vy / 100.0, vz=msg.vz / 100.0)
                got_position = True
            elif t == "ATTITUDE":
                state.update(roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw)
            elif t == "VFR_HUD":
                state["groundspeed"] = msg.groundspeed
            elif t == "MISSION_CURRENT":
                state["current_wp"] = msg.seq
            elif t == "HEARTBEAT":
                newmode = m.flightmode
                if newmode != state["mode"]:
                    mode_changes.append((round(time.time() - mission_start, 1),
                                         state["mode"], newmode))
                    state["mode"] = newmode
                state["armed"] = bool(m.motors_armed())
            elif t == "MISSION_ITEM_REACHED":
                reached.add(msg.seq)
                if msg.seq >= last_wp_seq:
                    done = True
            elif t == "STATUSTEXT":
                txt = msg.text.lower() if isinstance(msg.text, str) else str(msg.text)
                if "failsafe" in txt or "crash" in txt or "ekf" in txt and "fail" in txt:
                    failsafe_events.append((round(time.time() - mission_start, 1), msg.text))

        # disarm mid-air / end also ends the mission
        if was_armed and not state["armed"]:
            done = True
        was_armed = state["armed"]

        if not got_position:
            time.sleep(0.2)
            continue

        log["telemetry"].append({
            "t": round(time.time() - mission_start, 3),
            "lat": state["lat"], "lon": state["lon"], "alt": state["alt"],
            "roll": state["roll"], "pitch": state["pitch"], "yaw": state["yaw"],
            "vx": state["vx"], "vy": state["vy"], "vz": state["vz"],
            "groundspeed": state["groundspeed"], "current_wp": state["current_wp"],
            "mode": state["mode"], "armed": state["armed"]})
        time.sleep(0.2)

    # ---- land + disarm ----
    print("[run] Mission finished — landing.")
    set_mode(m, "LAND")
    t0 = time.time()
    while time.time() - t0 < 120:
        m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if not m.motors_armed():
            break
    print("[run] Disarmed.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"flight_log_{ts}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # ---- summary ----
    tele = log["telemetry"]
    alts = [e["alt"] for e in tele]
    gspd = [e["groundspeed"] for e in tele]
    expected_wp = last_wp_seq  # excludes home item 0
    unexpected = [mc for mc in mode_changes
                  if mc[2] not in ("GUIDED", "AUTO", "LAND")]
    print("\n" + "=" * 52)
    print("  MISSION SUMMARY")
    print("=" * 52)
    print(f"  Total flight time   : {tele[-1]['t']:.1f} s (wall, speedup {SPEEDUP}x)")
    print(f"  Telemetry samples   : {len(tele)}")
    print(f"  Waypoints reached   : {len(reached)} / {expected_wp}")
    print(f"  Max altitude        : {max(alts):.1f} m")
    print(f"  Min altitude        : {min(alts):.1f} m")
    print(f"  Max groundspeed     : {max(gspd):.1f} m/s")
    print(f"  Mode changes        : {mode_changes}")
    print(f"  Unexpected modes    : {unexpected if unexpected else 'none'}")
    print(f"  Failsafe events     : {failsafe_events if failsafe_events else 'none'}")
    print(f"  Log file            : {log_path}")
    print(f"  Telemetry entries   : {len(tele)}")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        print("[run] Stopping SITL ...")
        cleanup()
        print("[run] Clean shutdown complete.")
    sys.exit(rc)
