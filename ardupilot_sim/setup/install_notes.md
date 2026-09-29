# ArduPilot SITL — Install Notes

macOS (Apple Silicon, M4) · Clean, documented SITL setup for the Adaptive B-Spline
mission pipeline.

- **Date:** 2026-09-29
- **Host:** macOS 26.6.2 (Tahoe), arm64 (Apple Silicon M4)
- **Toolchain:** Apple clang 21.0.0, Homebrew 7.0.7, Xcode CLT present
- **ArduPilot commit (reproducibility):** `26c7363f64faac95a16d352a043563a2dc68ec8e`
  - `git describe`: `ArduPilot-4.6.0-beta1-8826-g26c7363f64`
- **SITL binary:** `~/ardupilot_sim/ardupilot/build/sitl/bin/arducopter` (6.1 MB, Mach-O arm64) ✅

---

## Project layout

```
~/ardupilot_sim/
├── ardupilot/        -> symlink to ~/Developer/ardupilot (existing, built)
├── setup/            (this file)
├── mission/          (Prompts 2–4, not yet created)
├── logs/
└── visualization/
```

## Step 0 — Folder structure

```bash
mkdir -p ~/ardupilot_sim/{setup,mission,logs,visualization}
```

## Step 1 — Locate existing ArduPilot repo

Searched `~/ardupilot`, `~/Documents`, `~/Developer`, `~/Desktop`, `~/Downloads`,
`~/projects`, `~/college`:

```bash
for loc in ~/ardupilot ~/Documents ~/Developer ~/Desktop ~/Downloads ~/projects ~/college; do
  find "$loc" -maxdepth 4 -type d -name ardupilot 2>/dev/null
done
# -> /Users/vignesvm/Developer/ardupilot   (already cloned AND already built)
```

**Decision:** A complete ArduPilot clone with all submodules initialized *and a
working ArduCopter SITL build* already exists at `~/Developer/ardupilot`. Rather
than re-download ~1–2 GB and rebuild, `~/ardupilot_sim/ardupilot` is a **symlink**
to it. This is the reproducible, non-messy choice — one canonical repo, one build.

```bash
ln -sfn ~/Developer/ardupilot ~/ardupilot_sim/ardupilot
```

> A fresh `git clone --recurse-submodules` (and `--depth 1` shallow variant) into
> `~/ardupilot_sim/ardupilot` was attempted first, but the existing built repo was
> reused instead (faster, already patched & verified). To force a truly fresh clone
> later: `rm ~/ardupilot_sim/ardupilot && git clone --recurse-submodules
> https://github.com/ArduPilot/ardupilot.git ~/ardupilot_sim/ardupilot`.

## Step 2 — Prerequisites (`install-prereqs-mac.sh`) — notes & deviations

The official script `Tools/environment_install/install-prereqs-mac.sh` was
reviewed. On macOS 26 / Apple Silicon it has three rough edges, so it was **not**
run verbatim; the equivalent, cleaner steps were used and are documented here:

1. **Homebrew system tools** — already present, needed by the build/scripts:
   `coreutils` (`grealpath`), `gawk`, `wget`. Xcode CLT present. ✅
2. **`python3 -m pip install ...` step → PEP 668 (`externally-managed-environment`).**
   The script installs Python packages into the system Python, which modern macOS
   blocks. Fixed by using a **virtualenv** for all Python deps (no `sudo pip`, no
   `--break-system-packages`). Build deps installed into the venv:
   `empy==3.3.4` (pinned — waf's code generator; empy 4.x breaks the API),
   `pymavlink`, `future`, `lxml`. (MAVProxy + the MAVLink stack go in Prompt 2's venv.)
3. **`wxPython` / `billiard` (graphics) build** — heavy/fragile on Apple Silicon.
   Skipped (equivalent to `SKIP_AP_GRAPHIC_ENV=1`). SITL and headless MAVLink work
   fine without it; only MAVProxy's `--map`/`--console` GUI needs wxPython.
4. **`pyenv` + `~/.zshrc` edits, and the STM32 `sudo` ARM toolchain** — skipped.
   Not needed for SITL (STM32 toolchain is only for real flight-board firmware).

## Step 3 — Build ArduCopter SITL

```bash
cd ~/ardupilot_sim/ardupilot        # -> ~/Developer/ardupilot
./waf configure --board sitl
./waf copter
```

### Errors encountered and fixes (macOS 26 / Apple clang 21)

The stock tree does not link under Apple's new linker (Xcode 16+ removed the
classic linker). Three source patches were applied (each carries an explanatory
comment in-tree):

| # | Error | Root cause | Fix |
|---|-------|-----------|-----|
| 1 | `ld: pointer not aligned in AP_FWVersion::fwver` | `PACKED` struct places `const char*` at a non-8-aligned offset; ld-prime rejects it | Removed `PACKED` from `libraries/AP_Common/AP_FWVersion.h` (packed layout only matters to external firmware-parsing tools, not SITL) |
| 2 | `-ld_classic` ignored / link fails | Xcode 16+ removed the classic linker | Skip `-dead_strip`/`-ld_classic` for clang ≥ 16 in `Tools/ardupilotwaf/boards.py` |
| 3 | `undefined ___real_malloc` (from `__wrap_malloc`) | On macOS `--wrap,malloc` is never passed, so `__wrap_malloc` is dead code previously hidden by `-dead_strip` | Guarded it out on `__APPLE__` in `libraries/AP_Common/c++.cpp` |

Result:

```
BUILD SUMMARY
Target          Text (B)  Data (B)  ...  Total Flash Used (B)
bin/arducopter   4325376    278528  ...             4603904
'copter' finished successfully
```

## Step 4 — Verify the binary

```bash
ls -lh build/sitl/bin/arducopter
# -rwxr-xr-x  6.1M  build/sitl/bin/arducopter
file build/sitl/bin/arducopter
# Mach-O 64-bit executable arm64
build/sitl/bin/arducopter --help    # prints options ✅
```

## Step 5 — Confirm SITL launches

```bash
cd ~/ardupilot_sim/ardupilot
Tools/autotest/sim_vehicle.py -v ArduCopter --console --map
```

**Confirmed launching** (verified headless via `--no-mavproxy`, then a pymavlink
client on `tcp:127.0.0.1:5760`):

```
RiTW: Starting ArduCopter : .../build/sitl/bin/arducopter --model + --speedup 1 ...
SIM_VEHICLE LAUNCH OK - heartbeat, mode STABILIZE
```

> **Note on `--console --map`:** these two flags open MAVProxy's wxPython GUI
> windows, which require `wxPython` (deliberately skipped above). The **simulator
> itself launches and serves MAVLink correctly**; for the GUI console/map either
> `pip install wxPython` into the venv (slow build on Apple Silicon) or use
> QGroundControl on `udp:127.0.0.1:14550` instead. Headless runs use:
> `Tools/autotest/sim_vehicle.py -v ArduCopter --no-mavproxy`.

## MAVLink endpoints (when SITL runs)

- `tcp:127.0.0.1:5760` — primary (GCS)
- `udp:127.0.0.1:14550`, `14551` — via MAVProxy `--out` (for QGC / scripts)

## Status

- [x] macOS dev environment verified
- [x] ArduPilot repo located & reused (`~/Developer/ardupilot`, symlinked)
- [x] Build deps handled cleanly in a venv (no PEP 668 breakage)
- [x] ArduCopter SITL builds (3 documented clang-21 patches)
- [x] `build/sitl/bin/arducopter` present & runnable
- [x] `sim_vehicle.py` launches SITL, heartbeat confirmed

**Next (not started — awaiting go-ahead):** Prompt 2 — Python MAVLink venv at
`~/ardupilot_sim/venv` + `mission/test_connection.py`.
