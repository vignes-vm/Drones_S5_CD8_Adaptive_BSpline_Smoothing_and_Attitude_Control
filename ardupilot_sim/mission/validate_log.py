#!/usr/bin/env python3
"""
Prompt 5 — validate a flight log before visualization.

Loads the most recent logs/flight_log_*.json and runs integrity checks:
  1. no NaN / null values in any telemetry field
  2. timestamps monotonically increasing
  3. >= 90% of expected waypoints reached (current_wp vs mission waypoint count)
  4. altitude never dropped below MIN_SAFE_ALT mid-mission (crash / premature land)

Prints PASS/FAIL per check. Exits non-zero if ANY check fails — do not proceed
to visualization until this passes.
"""
import os
import glob
import json
import math
import sys

HERE     = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(os.path.expanduser("~/ardupilot_sim"), "logs")
WP_FILE  = os.path.join(HERE, "bspline_mission.waypoints")
MIN_SAFE_ALT = 5.0          # metres
NUM_FIELDS = ["t", "lat", "lon", "alt", "roll", "pitch", "yaw",
              "vx", "vy", "vz", "groundspeed"]


def latest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "flight_log_*.json")))
    return logs[-1] if logs else None


def mission_waypoint_count():
    """Number of NAV waypoints (excludes the home item 0)."""
    with open(WP_FILE) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return len(lines) - 1 - 1   # minus header, minus home item


def is_bad(v):
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def main():
    log_path = latest_log()
    if not log_path:
        print("FAIL: no flight_log_*.json found")
        return 1
    print(f"Validating: {os.path.basename(log_path)}\n")
    data = json.load(open(log_path))
    tele = data.get("telemetry", [])
    results = []

    # --- Check 1: no NaN/null ---
    bad = []
    for i, e in enumerate(tele):
        for k in NUM_FIELDS:
            if k not in e or is_bad(e[k]):
                bad.append((i, k, e.get(k)))
    ok1 = len(bad) == 0
    results.append(("No NaN/null values", ok1,
                    "" if ok1 else f"{len(bad)} bad field(s), e.g. {bad[:3]}"))

    # --- Check 2: timestamps monotonically increasing ---
    ts = [e["t"] for e in tele]
    non_mono = [(i, ts[i - 1], ts[i]) for i in range(1, len(ts)) if ts[i] < ts[i - 1]]
    ok2 = len(non_mono) == 0
    results.append(("Timestamps monotonic", ok2,
                    "" if ok2 else f"{len(non_mono)} regressions, e.g. {non_mono[:3]}"))

    # --- Check 3: >=90% waypoints reached ---
    expected = mission_waypoint_count()
    max_wp = max((e["current_wp"] for e in tele), default=0)
    frac = max_wp / expected if expected else 0.0
    ok3 = frac >= 0.90
    results.append((">=90% waypoints reached", ok3,
                    f"reached wp {max_wp}/{expected} ({frac*100:.1f}%)"))

    # --- Check 4: altitude never below MIN_SAFE_ALT mid-mission ---
    low = [(round(e["t"], 2), round(e["alt"], 2), e["current_wp"])
           for e in tele if e["alt"] < MIN_SAFE_ALT]
    ok4 = len(low) == 0
    results.append((f"Altitude >= {MIN_SAFE_ALT} m mid-mission", ok4,
                    "" if ok4 else f"{len(low)} sample(s) below, e.g. {low[:3]}"))

    # --- report ---
    print(f"{'CHECK':<34} RESULT   DETAIL")
    print("-" * 72)
    all_ok = True
    for name, ok, detail in results:
        all_ok &= ok
        print(f"{name:<34} {'PASS' if ok else 'FAIL':<8} {detail}")
    print("-" * 72)
    print(f"telemetry entries: {len(tele)}")
    if all_ok:
        print("\nALL CHECKS PASSED — log is ready for visualization.")
        return 0
    print("\nVALIDATION FAILED — fix the mission script before visualizing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
