# Running the ArduPilot SITL B-Spline Mission Pipeline

All commands assume the project venv. **Activate it first in every terminal:**

```bash
source ~/ardupilot_sim/venv/bin/activate
```

Paths:
- Project root: `~/ardupilot_sim`
- ArduPilot repo: `~/ardupilot_sim/ardupilot` (symlink to the built clone)
- Home coordinate (placeholder — Chennai): `lat 13.0827, lon 80.2707`
  (change in `mission/generate_mission.py` for your real test site)

---

## 1. Quick connection test (sanity check)

Starts SITL, connects over MAVLink, prints mode/armed/GPS, cleans up.

```bash
cd ~/ardupilot_sim
python mission/test_connection.py
```

## 2. Generate the mission (plan → B-Spline → waypoints)

3D A* → cubic B-Spline (scipy) → 20 m arc-length waypoints → QGC WPL 110.

```bash
cd ~/ardupilot_sim
python mission/generate_mission.py
```

Outputs:
- `mission/bspline_mission.waypoints` (mission file)
- `mission/path_metadata.json` (metrics, obstacles, spline)

## 3. Fly the mission headless + log telemetry

Starts SITL, uploads the mission, flies AUTO, logs at 5 Hz, lands, summarises.

```bash
cd ~/ardupilot_sim
python mission/run_mission.py
```

Output: `logs/flight_log_<timestamp>.json`

## 4. Validate the flight log

Integrity gate before visualization (NaN/monotonic/waypoints/altitude).

```bash
cd ~/ardupilot_sim
python mission/validate_log.py
```

## 5. Visualize + analyze (offline)

Reads the newest log + metadata; writes plots and a report.

```bash
cd ~/ardupilot_sim
python visualization/visualize_flight.py   # 4 PNGs + summary_report.txt
python visualization/map_view.py           # offline OSM map -> flight_map.html
```

Outputs in `visualization/output/`:
`static_3d_flight.png`, `attitude_over_time.png`, `tracking_error.png`,
`mission_progress.png`, `summary_report.txt`, `flight_map.html`

---

## 6. Watch it live in QGroundControl

Run these in **separate terminal tabs** (each with the venv activated).
**Order matters:** SITL → hub → QGC → fly script.

**Terminal 1 — SITL (simulator):**
```bash
cd ~/ardupilot_sim/ardupilot
Tools/autotest/sim_vehicle.py -v ArduCopter --no-mavproxy --no-rebuild \
    --custom-location=13.0827,80.2707,0,0 --speedup 3
```
Wait until it prints `Starting ArduCopter` and opens port 5760.

**Terminal 2 — MAVProxy hub (fans MAVLink to QGC + the fly script):**
```bash
mavproxy.py --master=tcp:127.0.0.1:5760 \
    --out=udp:127.0.0.1:14550 \
    --out=udp:127.0.0.1:14551 --daemon
```

**Launch QGroundControl** (auto-connects to udp:14550):
```bash
open -a QGroundControl
```

**Terminal 3 — upload + fly (QGC shows it live on the map):**
```bash
cd ~/ardupilot_sim
python mission/fly_in_qgc.py
```

---

## Stop everything

```bash
pkill -f sim_vehicle; pkill -f arducopter; pkill -f mavproxy
```
…then quit QGroundControl.

---

## Full end-to-end (headless) in one block

```bash
source ~/ardupilot_sim/venv/bin/activate
cd ~/ardupilot_sim
python mission/generate_mission.py \
  && python mission/run_mission.py \
  && python mission/validate_log.py \
  && python visualization/visualize_flight.py
```

## Troubleshooting

- **`fly_in_qgc.py` says "no heartbeat"** → SITL/hub not up yet; wait a few
  seconds and rerun (order must be SITL → hub → script).
- **`externally-managed-environment` on pip** → you forgot to activate the venv.
- **Mission too slow to watch** → raise `--speedup` (e.g. 5) in Terminal 1.
- **QGC shows no vehicle** → confirm the hub is forwarding to `udp:14550` and
  QGC's comm link is the default UDP (14550).
