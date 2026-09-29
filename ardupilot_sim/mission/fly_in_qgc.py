#!/usr/bin/env python3
"""
Fly the mission on an ALREADY-RUNNING SITL + MAVProxy hub, so QGroundControl
(connected on udp:14550) shows the drone tracing the B-Spline path on its map.

Assumes:
  - SITL is running (sim_vehicle --no-mavproxy) at the mission home
  - a MAVProxy hub forwards tcp:5760 -> udp:14550 (QGC) and udp:14551 (this)
  - QGroundControl is open (auto-connects to 14550)

This script connects to udp:14551, uploads the mission, arms, takes off, and
switches to AUTO — then just monitors. It does NOT stop SITL/QGC afterwards.
"""
import time
from pymavlink import mavutil

# reuse the mission helpers
from run_mission import (parse_wpl, upload_mission, set_mode, wait_mode,
                         cmd_long, wait_armable_and_arm, WP_FILE)

CONN = "udp:127.0.0.1:14551"


def main():
    items = parse_wpl(WP_FILE)
    last_wp = items[-1]["seq"]
    takeoff_alt = next(it["z"] for it in items if it["seq"] >= 1)

    print(f"[qgc] Connecting to {CONN} ...")
    m = mavutil.mavlink_connection(CONN, source_system=201)
    m.wait_heartbeat(timeout=30)
    print(f"[qgc] Connected (sys {m.target_system}).")
    m.mav.request_data_stream_send(m.target_system, m.target_component,
                                   mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1)

    # wait for GPS so QGC shows the vehicle + arming works
    t0 = time.time()
    while time.time() - t0 < 60:
        g = m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=5)
        if g and g.fix_type >= 3:
            break

    upload_mission(m, items)
    print("[qgc] Mission uploaded — it should now be visible on the QGC map.")

    if not wait_armable_and_arm(m):
        print("[qgc] could not arm"); return 1
    print(f"[qgc] Takeoff to {takeoff_alt:.0f} m ...")
    cmd_long(m, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, takeoff_alt)
    t0 = time.time()
    while time.time() - t0 < 60:
        p = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=3)
        if p and p.relative_alt / 1000.0 >= takeoff_alt * 0.95:
            break
    print("[qgc] Switching to AUTO — watch the drone fly the path in QGC.")
    set_mode(m, "AUTO"); wait_mode(m, "AUTO", 10)
    cmd_long(m, mavutil.mavlink.MAV_CMD_MISSION_START, 0, last_wp)

    # monitor progress (QGC shows the live map; we just print milestones)
    reached = -1
    t0 = time.time()
    while time.time() - t0 < 600:
        msg = m.recv_match(type=["MISSION_ITEM_REACHED", "MISSION_CURRENT"],
                           blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.get_type() == "MISSION_ITEM_REACHED":
            if msg.seq // 20 != reached // 20:   # print every ~20 wps
                print(f"[qgc] reached waypoint {msg.seq}/{last_wp}")
            reached = msg.seq
            if msg.seq >= last_wp:
                print("[qgc] Mission complete — drone at final waypoint.")
                break
    print("[qgc] Leaving SITL + QGC running so you can keep exploring the map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
