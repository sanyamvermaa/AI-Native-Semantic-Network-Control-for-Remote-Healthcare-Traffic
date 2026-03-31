# Project Context

## Overview
This repository is organized around two separate workflows:

1. Dataset generation module
2. Closed-loop module

The modules are intentionally isolated so development can continue independently.

## Current Structure

- data/
  - datasets/
  - logs/
- models/
- plots/
- scripts/
  - setup_namespaces.sh
  - dataset_generation/
    - dynamic_traffic_generator.py
    - generate_dataset.py
    - health_sender.py
    - health_receiver.py
    - run_experiment.sh
    - train_model.py
    - tune_xgboost.py
    - plot.py
    - bd.py
  - closed_loop/
    - health_sender.py         ← updated (see changelog)
    - health_receiver.py       ← updated (see changelog)
    - ward_controller.py       ← updated (see changelog)
    - dashboard.py             ← updated (see changelog)
    - common.py                ← updated (see changelog)
    - run_closed_loop_stress_auto.sh

## Module Boundaries

### Dataset generation
- Uses scripts inside scripts/dataset_generation only.
- Has its own local copies of sender and receiver.
- Writes telemetry and generated datasets under data/.
- Intended for synthetic data creation, feature engineering, and model training input preparation.

### Closed-loop
- Uses scripts inside scripts/closed_loop only.
- Operates as a separate runtime/stress workflow.
- Does not depend on dataset_generation sender or receiver copies.

## Entry Points

### Dataset generation flow
1. scripts/setup_namespaces.sh
2. scripts/dataset_generation/run_experiment.sh
3. scripts/dataset_generation/dynamic_traffic_generator.py
4. scripts/dataset_generation/generate_dataset.py
5. scripts/dataset_generation/train_model.py
6. scripts/dataset_generation/tune_xgboost.py

### Closed-loop flow
1. scripts/setup_namespaces.sh
2. scripts/closed_loop/run_closed_loop_stress_auto.sh

## Notes
- Namespace setup remains shared via scripts/setup_namespaces.sh.
- Sender and receiver are duplicated by design across both modules.
- Keep all new dataset-generation logic inside scripts/dataset_generation.
- Keep all new closed-loop logic inside scripts/closed_loop.
- neurokit2 and numpy are required pip dependencies for health_sender.py (closed-loop).
  Install with: pip install neurokit2 numpy

---

## Changelog — Closed-Loop Module (all fixes applied, files ready to deploy)

### common.py

**Bug fixed: `policy_command` triggered SEMANTIC_SUMMARY on stable network**

Root cause: `choose_health_state` returns `"UNKNOWN"` whenever a 250ms telemetry
window contains zero labelled packets (startup race, brief loss spike). The tuple
`("Stable", "UNKNOWN")` was not in the policy dict, so it fell through to the
catch-all default which was `"SEMANTIC_SUMMARY"`.

Fixes applied:
- Added `if health == "UNKNOWN": health = "NORMAL"` at the top of `policy_command`.
  Rationale: empty window on a stable network = benefit of the doubt, keep transmitting.
  Exception: `("Critical", "UNKNOWN")` naturally resolves to `SEMANTIC_SUMMARY` via the
  `Critical/NORMAL` row — bad channel AND no signal is a legitimate suppression trigger.
- Changed the catch-all fallback from `"SEMANTIC_SUMMARY"` to `"DOWNSAMPLED_ECG"`.
  Any truly unexpected (net, health) combo gets conservative downsampling, not full suppression.

Full 9-case policy table (unchanged):

  ("Stable",   "NORMAL")   → FULL_ECG
  ("Stable",   "ALERT")    → FULL_ECG_PRIORITY
  ("Stable",   "CRITICAL") → SEMANTIC_CRITICAL
  ("Unstable", "NORMAL")   → DOWNSAMPLED_ECG
  ("Unstable", "ALERT")    → SEMANTIC_ALERT
  ("Unstable", "CRITICAL") → SEMANTIC_CRITICAL
  ("Critical", "NORMAL")   → SEMANTIC_SUMMARY
  ("Critical", "ALERT")    → SEMANTIC_ALERT
  ("Critical", "CRITICAL") → SEMANTIC_CRITICAL


### ward_controller.py

**Bug fixed: immediate SEMANTIC_SUMMARY at startup (Temperature "NO SIGNAL")**

Root cause: `last_packet_ts = 0.0` (Unix epoch). The window-timeout check
`now - last_packet_ts > window_timeout` fired at t=0 before any packets arrived,
sending SEMANTIC_SUMMARY to all devices. Temperature in SUMMARY mode requires
`summary_interval=120s` AND 10 buffer samples to emit anything → silent for 2+ minutes.

Fix: `last_packet_ts = time.time()` — countdown starts from boot, not epoch zero.


### health_sender.py

**Feature: Physiological signal generation (replaces random.randint)**

Two new classes replace flat random value generation:

**`PhysiologicalModel` (all non-ECG devices, ECG fallback)**
Ornstein-Uhlenbeck mean-reverting stochastic process:
  X(t+dt) = X(t) + θ(μ−X)·dt + σ·√dt·N(0,1)

OU parameters tuned per device type (theta=reversion speed, sigma=noise):
  ECG:           θ=0.30, σ=4.0   (~3s time-constant, realistic HRV noise)
  SpO2:          θ=0.50, σ=0.3   (tightly regulated, small slow variation)
  BloodPressure: θ=0.25, σ=2.5   (moderate beat-to-beat variation)
  Temperature:   θ=0.02, σ=0.2   (changes over many minutes)
  Respiration:   θ=0.35, σ=0.8   (moderate breath-to-breath variation)

Burst events shift the OU attractor (μ) to the burst-range midpoint so values
converge gradually rather than jumping instantly. Recovery is symmetric.

**`ECGNeuroKitSource` (ECG devices when neurokit2 is installed)**
Uses `nk.ecg_simulate()` to generate a 60s ECG waveform at 100 Hz, then extracts
R-peaks via `nk.ecg_peaks()` and converts RR intervals to instantaneous HR (BPM).
Each 60s buffer is streamed sample-by-sample. Beat-to-beat variation is ≤2 BPM
(physiologically correct HRV). The OU model drives the long-term mean HR so
gradual drift and burst events still work. Gracefully falls back to OU if neurokit2
is not available.

Import block updated to include `math` and optional neurokit2/numpy imports with
`_NK2_AVAILABLE` flag.


### health_receiver.py

**Fix: `window_sec` added to receiver→ward UDP payload**

`ward_controller` reads `payload.get("window_sec") or 0.25` for latency calculation.
The field was never sent; the fallback value happened to be correct (0.25 =
`TELEMETRY_INTERVAL`) but it is now sent explicitly.


### dashboard.py

**Bug fixed: Temperature always showing "NO SIGNAL"**

Root cause: stale threshold was hardcoded at 3.0s for all devices. Temperature
transmits every 30s so it was always flagged stale.

Fix: Added `DEVICE_STALE_TIMEOUT` dict at module level with per-device timeouts:
  ECG:           5s   (100 Hz sender)
  SpO2:          10s  (1 Hz sender)
  BloodPressure: 5s   (100 Hz sender)
  Temperature:   90s  (30s sender — 3× interval)
  Respiration:   10s  (1 Hz sender)

`api_devices` now looks up `DEVICE_STALE_TIMEOUT.get(resolved_type, 5.0)` per device
and also returns `stale_timeout` in the JSON for frontend reference.

**Bug fixed: `null` seconds_ago shown as "Updated: 0.0s ago"**

`Number(null) === 0` in JS, so before `ward_mode_state.json` existed the UI falsely
showed "Updated: 0.0s ago". Fixed with explicit `!== null` check. When null, shows
"Updated: no data".

**Bug fixed: stale ward badge still showed last known network state**

When ward/receiver was down >8s, the state badge still showed the last known state
in colour. Now shows "⚠ NO DATA" in grey when `seconds_ago > 8` or `null`.

**Bug fixed: charts always started blank on page load**

`updateTelemetry` read only `d.current` and ignored the 120-row history arrays
(`d.packet_loss_rate`, `d.avg_delay`, `d.jitter`) returned by `/api/telemetry`.
Added `seedChartsFromHistory()` that pre-populates all three charts on the first
successful fetch. Subsequent ticks use the existing rolling-append logic.

**Improvement: health state badge added to top bar**

`healthStateBadge` element added next to the network state badge. Populated from
`d.health_state` in `updateState` — NORMAL/ALERT/CRITICAL visible at a glance.
