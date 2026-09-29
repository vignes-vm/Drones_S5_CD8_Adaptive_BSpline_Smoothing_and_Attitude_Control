#!/usr/bin/env python3
"""
Map view — overlay the flight on an interactive OpenStreetMap.

Standalone: reads only the most recent logs/flight_log_*.json and
mission/path_metadata.json, and writes visualization/output/flight_map.html
(open it in a browser; OSM tiles load online).

  - actual GPS track (red)
  - planned B-Spline path (blue, converted local->GPS about home)
  - obstacles (orange circles at their footprint radius)
  - start / end markers

NOTE: the home coordinate is the placeholder in path_metadata.json (Chennai).
Change HOME in generate_mission.py to your real test site and re-run the whole
pipeline to place this map at the true location.
"""
import os
import glob
import json
import numpy as np
import folium

HOME_ROOT = os.path.expanduser("~/ardupilot_sim")
LOG_DIR   = os.path.join(HOME_ROOT, "logs")
META_FILE = os.path.join(HOME_ROOT, "mission", "path_metadata.json")
OUT_DIR   = os.path.join(HOME_ROOT, "visualization", "output")
EARTH_M_PER_DEG = 111320.0


def latest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "flight_log_*.json")))
    if not logs:
        raise SystemExit("no flight_log_*.json found")
    return logs[-1]


def local_to_latlon(x, y, home_lat, home_lon):
    """local (East, North) metres -> (lat, lon)."""
    lat = home_lat + (y / EARTH_M_PER_DEG)
    lon = home_lon + (x / (EARTH_M_PER_DEG * np.cos(np.radians(home_lat))))
    return lat, lon


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log = json.load(open(latest_log()))
    meta = json.load(open(META_FILE))
    home_lat = meta["home"]["lat"]
    home_lon = meta["home"]["lon"]

    tele = log["telemetry"]
    actual = [(e["lat"], e["lon"]) for e in tele]

    planned_local = np.array(meta["sampled_path_local"])
    planned = [local_to_latlon(p[0], p[1], home_lat, home_lon)
               for p in planned_local]

    m = folium.Map(location=[home_lat, home_lon], zoom_start=15,
                   tiles="OpenStreetMap")

    folium.PolyLine(planned, color="blue", weight=3, opacity=0.8,
                    tooltip="Planned B-Spline").add_to(m)
    folium.PolyLine(actual, color="red", weight=3, opacity=0.9,
                    tooltip="Actual GPS trajectory").add_to(m)

    for o in meta["obstacles"]:
        olat, olon = local_to_latlon(o["x"], o["y"], home_lat, home_lon)
        folium.Circle([olat, olon], radius=o["radius"], color="orange",
                      fill=True, fill_opacity=0.35,
                      tooltip=f"obstacle r={o['radius']:.0f}m h={o['height']:.0f}m"
                      ).add_to(m)

    folium.Marker(actual[0], tooltip="Start",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(actual[-1], tooltip="End",
                  icon=folium.Icon(color="black")).add_to(m)

    # fit view to the whole track
    lats = [p[0] for p in actual]; lons = [p[1] for p in actual]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    out = os.path.join(OUT_DIR, "flight_map.html")
    m.save(out)
    print(f"Saved interactive map: {out}")
    print(f"  actual points: {len(actual)}  | obstacles: {len(meta['obstacles'])}")
    print(f"  home (placeholder): {home_lat}, {home_lon}")
    print("  open in a browser (OSM tiles load online).")


if __name__ == "__main__":
    main()
