# Project Context

## Overview
This repository implements an **AI-Native Semantic Network Control** system for Remote Healthcare Traffic.
It is organized around three modules that are intentionally isolated so development can continue independently.

1. **Dataset generation module** — synthetic telemetry generation, ML model training
2. **Closed-loop module** — real-time adaptive control runtime
3. **Semantic AI pipeline** — encoder/decoder + patient fusion (Steps 1–6 complete)

---

## Current Structure

```
data/
  datasets/       — synthetic telemetry CSVs (final_network_dataset.csv, etc.)
  logs/           — runtime logs (sender CSVs, telemetry, command_log, ward_mode_state.json)
  eval_logs/      — enriched logs for semantic fidelity evaluation (generated)
models/
  semantic/       — trained TorchScript models (enc_ECG.pt, dec_ECG.pt,
                    enc_BloodPressure.pt, dec_BloodPressure.pt,
                    patient_fusion.pt, metadata.json)
  best_network_model.pkl / .json       — XGBoost network state classifier (best)
  xgboost_network_model.pkl / .json
  xgboost_tuned_model.pkl / .json
  lgbm_network_model.pkl / .json
  robust_network_model.pkl / .json
  label_classes.json                   — ["Critical", "Stable", "Unstable"]
plots/
  semantic_fidelity/  — Step 6 output figures + summary_report.txt
scripts/
  setup_namespaces.sh
  dataset_generation/
    dynamic_traffic_generator.py
    generate_dataset.py
    health_sender.py
    health_receiver.py
    run_experiment.sh
    train_model.py
    tune_xgboost.py
    plot.py
    bd.py
  closed_loop/
    health_sender.py         — adaptive sender with OU/neurokit2 physiology + semantic codec
    health_receiver.py       — ML network classifier + semantic decoder + telemetry publisher
    ward_controller.py       — 9-case policy + PatientFusion escalation + ACK-retry
    dashboard.py             — Flask dashboard served at :5050
    common.py                — shared DEVICE_LAYOUT, policy_command, helpers
    run_closed_loop_stress_auto.sh
  semantic/
    train_semantic_codec.py     — Step 1
    semantic_encoder.py         — Runtime encoder/decoder wrapper
    channel_quantizer.py        — Variable-fidelity latent quantization
    test_step2.py               — Step 2
    inspect_payload.py          — Step 3
    test_step4.py               — Step 4
    train_patient_fusion.py     — Step 5
    patient_fusion.py           — Runtime patient fusion inference wrapper
  evaluation/
    enrich_logs_for_fidelity.py
    semantic_fidelity.py        — Step 6
    baseline_sender.py
    analyze_results.py
    run_closedloop_eval.sh
    run_baseline_eval.sh
```

---

## Module Boundaries

### Dataset generation
- Uses scripts inside `scripts/dataset_generation` only.
- Has its own local copies of sender and receiver.
- Writes telemetry and generated datasets under `data/`.

### Closed-loop
- Uses scripts inside `scripts/closed_loop` only.
- Operates as a separate runtime/stress workflow.
- Does not depend on `dataset_generation` sender or receiver copies.

### Semantic pipeline
- All training scripts in `scripts/semantic/`.
- Runtime wrappers (`semantic_encoder.py`, `patient_fusion.py`) imported by closed_loop senders/receivers.
- Evaluation scripts in `scripts/evaluation/`.
- Models saved to `models/semantic/`.

---

## Entry Points

### Dataset generation flow
1. `scripts/setup_namespaces.sh`
2. `scripts/dataset_generation/run_experiment.sh`
3. `scripts/dataset_generation/dynamic_traffic_generator.py`
4. `scripts/dataset_generation/generate_dataset.py`
5. `scripts/dataset_generation/train_model.py`
6. `scripts/dataset_generation/tune_xgboost.py`

### Closed-loop flow
1. `scripts/setup_namespaces.sh`
2. `scripts/closed_loop/run_closed_loop_stress_auto.sh [duration] [stage_sec]`

### Semantic pipeline flow (run in order)
1. `wsl python3 scripts/semantic/train_semantic_codec.py`
2. `wsl python3 scripts/semantic/test_step2.py`
3. `wsl python3 scripts/semantic/inspect_payload.py`
4. `wsl python3 scripts/semantic/test_step4.py`
5. `wsl python3 scripts/semantic/train_patient_fusion.py`
6. `wsl python3 scripts/evaluation/enrich_logs_for_fidelity.py`
   `wsl python3 scripts/evaluation/semantic_fidelity.py --logs-dir data/eval_logs`

---

## Closed-Loop Architecture

```
8 SENDERS ──UDP:9000──► RECEIVER ──UDP:5006──► WARD CONTROLLER
  ▲                        │                          │
  │                    ML inference               9-case policy
  │                    (XGBoost,             + PatientFusion escalation
  │                    24 features)              + ACK-retry
  │                    semantic decode                │
  └────────────────────────────── UDP:600x ◄──────────┘
                                  (per-device command)
```

### Device Fleet (DEVICE_LAYOUT in common.py)

| ID | Type          | Sample Rate | Semantic Capable |
|----|---------------|-------------|------------------|
|  0 | ECG           | 100 Hz      | ✓ (encoder+decoder) |
|  1 | ECG           | 100 Hz      | ✓ |
|  2 | SpO2          | 1 Hz        | ✗ (raw only) |
|  3 | SpO2          | 1 Hz        | ✗ |
|  4 | BloodPressure | 100 Hz      | ✓ |
|  5 | BloodPressure | 100 Hz      | ✓ |
|  6 | Temperature   | 1/30 Hz     | ✗ |
|  7 | Respiration   | 1 Hz        | ✗ |

---

## Multi-Transmission Mode System

This is the **core design** of the closed-loop system. There are two distinct layers:

### Layer 1 — Policy Commands (ward_controller → senders)
The ward controller issues one of **6 commands** based on the 3×3 network × health state matrix:

| Policy Command       | Network State | Health State    | Tx Mode     | Interval Mult | Notes |
|----------------------|---------------|-----------------|-------------|---------------|-------|
| `FULL_ECG`           | Stable        | NORMAL          | RAW         | 1.0×          | Every sample transmitted |
| `FULL_ECG_PRIORITY`  | Stable        | ALERT           | RAW         | 0.8×          | Slightly faster rate |
| `DOWNSAMPLED_ECG`    | Unstable      | NORMAL          | DELTA       | 4.0× (ECG)    | ECG 100Hz→25Hz; non-ECG hold rate |
| `SEMANTIC_ALERT`     | Unstable/Critical | ALERT       | DELTA       | 1.5×          | Rate slowed + delta filter |
| `SEMANTIC_CRITICAL`  | Any           | CRITICAL        | CRITICAL_ONLY | 1.0×        | Suppress non-urgent; ECG/BP use ML encoder |
| `SEMANTIC_SUMMARY`   | Critical      | NORMAL / timeout | SUMMARY    | 8.0×          | Time-gated compressed summaries |

**Overrides:**
- If `receiver_overloaded=True` → always `SEMANTIC_SUMMARY` (drain protection)
- If no packet for `window_timeout` (3s) → `SEMANTIC_SUMMARY` (silence handling)
- PatientFusion `deterioration_prob > 0.75` on Stable network → escalate health to ALERT → re-run policy

### Layer 2 — Transmission Modes (sender-side filtering)

| Tx Mode       | Behaviour |
|---------------|-----------|
| `RAW`         | Every sample transmitted unfiltered |
| `DELTA`       | Transmit only if value changed ≥5% from last sent |
| `SUMMARY`     | Time-gated: one compressed JSON summary per `summary_interval` seconds |
| `CRITICAL_ONLY` | Transmit only if `clinical_importance ≥ 0.7`; ECG/BP use ML latent encoding |

**Semantic encoding fast-path (ECG + BloodPressure only):**
- Sender accumulates 200-sample window → `SemanticEncoder.encode()` → 16-dim latent z
- `encode_payload()` quantizes z to N dims based on command: 16/16/8/8/4/2
- Receiver calls `decode_payload()` → `SemanticEncoder.decode()` → `clinical_state` + confidence

### Stale Timeout per Device Type

| Device Type   | Stale After |
|---------------|-------------|
| ECG           | 5 s         |
| BloodPressure | 5 s         |
| SpO2          | 10 s        |
| Respiration   | 10 s        |
| Temperature   | 90 s        |

---

## Network State Classifier (Receiver)

The receiver uses **XGBoost** (loaded from `models/best_network_model.pkl`) with **24 engineered features**:

**Base features (7):** `bandwidth_usage_bps`, `throughput_bps`, `packet_loss_rate`, `jitter`, `avg_delay`, `active_devices`, `packets_per_window`

**Engineered features (17):** rolling means/stds (W=10 windows), deltas, acceleration, trend slopes (S=8 windows), cross-products (`loss×delay`, `loss×jitter`)

**3 classes:** Stable / Unstable / Critical  
**Confidence thresholds:** Stable ≥ 0.60, Unstable ≥ 0.55, Critical ≥ 0.60 (otherwise step up severity)

**Network condition thresholds** (from `common.py`):
- Critical:  loss ≥ 8% OR delay ≥ 130ms OR jitter ≥ 35ms
- Unstable:  loss ≥ 3% OR delay ≥ 35ms OR jitter ≥ 8ms
- Stable:    below all above

**Stress test profiles** (from `run_closed_loop_stress_auto.sh`):
- Stable:   loss 0–0.8%, delay 5–25ms, jitter 0–4ms
- Unstable: loss 3–7%, delay 40–120ms, jitter 8–25ms
- Critical: loss 12–22%, delay 180–380ms, jitter 50–90ms

---

## Health State Determination (Receiver)

1. Receiver assigns `label` per packet: NORMAL / ALERT / CRITICAL (from sender or decoded)
2. EMA-smoothed label counts (α=0.3) per window → `health_state = argmax(EMA)`
3. Low decode confidence (<0.55) forces health_state to NORMAL (unless already CRITICAL)
4. PatientFusion in ward_controller can escalate health_state to ALERT

**Clinical thresholds (sender labelling):**
- ECG (bpm): NORMAL 60–100 | ALERT 100–130 | CRITICAL >130 or <40
- BloodPressure (mmHg): NORMAL 110–130 | ALERT 140+ | CRITICAL 160+ or <80
- SpO2 (%): NORMAL 95–100 | ALERT <94 | CRITICAL <90
- Temperature (Cx10): NORMAL 360–373 | ALERT 376+ | CRITICAL 385+
- Respiration (br/min): NORMAL 12–20 | ALERT 22+ | CRITICAL 28+ | CRITICAL <5

---

## Semantic Pipeline — Steps & Final Results

### Step 1 — train_semantic_codec.py [PASS]
**Purpose:** Train per-device-type encoder/decoder (ECG + BloodPressure) VAE-style. Window=200, LatentDim=16.

| Device        | Accuracy | F1[NORMAL] | F1[ALERT] | F1[CRITICAL] |
|---------------|----------|------------|-----------|--------------|
| ECG           | 0.990    | 0.983      | 0.990     | 0.991        |
| BloodPressure | 0.924    | 0.934      | 0.763     | 0.954        |

Models: `models/semantic/enc_ECG.pt`, `dec_ECG.pt`, `enc_BloodPressure.pt`, `dec_BloodPressure.pt`, `metadata.json`

### Step 2 — test_step2.py [ALL CHECKS PASSED]
Round-trip encode→decode for all 3 phases × 2 device types. 0 dangerous misclassifications.

### Step 3 — inspect_payload.py [COMPLETE]
6 command levels verified. Latent dims by command: 16/16/8/8/4/2.

### Step 4 — test_step4.py [PASS]
1080 round-trips (30 windows × 3 phases × 2 devices × 6 commands).
0 dangerous misses (CRITICAL decoded as NORMAL). 0 hi-fi false alarms.

### Step 5 — train_patient_fusion.py [PASS]
Cross-modal ECG+BP fusion → joint state + deterioration probability.

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.788 |
| F1 macro   | 0.772 |
| F1[NORMAL] | 0.877 |
| F1[ALERT]  | 0.590 |
| F1[CRITICAL]| 0.848 |

Model: `models/semantic/patient_fusion.pt`

### Step 6 — semantic_fidelity.py [COMPLETE]
End-to-end evaluation across 12 (network_condition × command) combinations.

**Clinical state F1 heatmap (weighted F1):**

| Command           | Stable | Unstable | Critical |
|-------------------|--------|----------|----------|
| FULL_ECG          | 0.152  | 0.353    | 0.367    |
| SEMANTIC_ALERT    | 0.276  | 0.441    | 0.376    |
| SEMANTIC_CRITICAL | 0.528  | 0.315    | **0.528**|
| SEMANTIC_SUMMARY  | 0.376  | 0.190    | 0.432    |

Overall clinical state F1: 0.380 · SLA compliance (SEMANTIC_CRITICAL <1000ms): 100% · Mean decode confidence: 0.574

Outputs: `plots/semantic_fidelity/` (4 figures + summary_report.txt)

---

## Semantic Pipeline — Key Constants

| Constant           | Value |
|--------------------|-------|
| WINDOW_SIZE (codec)| 200 samples |
| LATENT_DIM         | 16 |
| CHUNK_SIZE (eval)  | 300 rows |
| ECG devices        | dev10 (id=10), dev11 (id=11) [semantic eval only] |
| BP devices         | dev12 (id=12), dev13 (id=13) [semantic eval only] |
| Slot map           | {0:0, 1:1, 4:2, 5:3, 10:4, 11:5, 12:6, 13:7} |
| Sample rate        | 100 Hz (dt=0.01s) |
| Telemetry interval | 250 ms (4 windows/sec) |
| Drain budget       | 150 ms per telemetry window |
| MIN_HISTORY_WINDOWS| 8 (before ML model is used) |

---

## Dashboard (scripts/closed_loop/dashboard.py)

Served at `http://localhost:5050` — Flask + vanilla JS, polling every 1s.

### API Endpoints
| Endpoint            | Returns |
|---------------------|---------|
| `GET /`             | HTML dashboard |
| `GET /api/state`    | Current network/health/command from `ward_mode_state.json` |
| `GET /api/devices`  | Per-device value, label, stale status from `sender_log_dev*.csv` |
| `GET /api/telemetry`| Last 120 rows of `network_telemetry.csv` (arrays + current) |
| `GET /api/commands` | Last 20 rows of `command_log.csv` (reversed, newest first) |
| `GET /api/semantic_stats` | Aggregated tx-mode counters from `sender_semantic_stats_dev*.csv` |

### Dashboard Sections
1. **Top bar** — network state badge, active policy command, health state, latency, online device count, suppression %
2. **4 metric cards** — Packet Loss, Avg Delay, Jitter, Throughput (each with live sparkline chart)
3. **Transmission Mode Counters** — split-panel:
   - Left: live per-tx-mode counters (RAW/DELTA/SUMMARY/CRITICAL_ONLY) from sender stats
   - Right: static policy command → tx mode mapping reference table
4. **Patient Device Fleet** — 8 device cards with value, clinical label, staleness, mode tag
5. **System Architecture** — data flow diagram with accurate labels
6. **Command History** — last 20 commands with time, net state, health state, command+tx-mode tag, latency

---

## Notes
- Namespace setup shared via `scripts/setup_namespaces.sh`.
- Sender/receiver are duplicated by design across both modules.
- Keep all new dataset-generation logic inside `scripts/dataset_generation`.
- Keep all new closed-loop logic inside `scripts/closed_loop`.
- All semantic training/evaluation commands run under WSL Python 3.12.3 with PyTorch 2.11.0+cu130.
- Run all commands from the project root directory.

---

## Reproducibility Engineering (May 2026)

### Problem
Early evaluation runs had non-deterministic network stress between baseline and closed-loop experiments.
Two root causes were identified and fixed:

**1. Physiological seed non-determinism:**
Both `baseline_sender.py` and `health_sender.py` now accept `--seed-base 20260101`.
Each device seeds as `random.seed(SEED_BASE + device_id)`, ensuring all 8 devices produce identical vital-sign sequences across both runs.

**2. Network stress non-determinism:**
`dwell()` used bash `$RANDOM`, which is seeded by PID+time — not reproducible across separate processes.
Attempts to seed with `RANDOM=20260101` failed because bash's internal PRNG is PID-influenced.

**Final fix — scenario manifest replay:**
- `scripts/evaluation/generate_stress_manifest.py` pre-generates all 102 stress steps using Python's `random.seed(20260101)` (a pure, portable LCG).
- Output: `scripts/evaluation/scenario_manifest.csv` — committed to the repository.
- Both `run_natural_stress_baseline.sh` and `run_natural_stress_closedloop.sh` now call `replay_manifest()` which reads the CSV line-by-line (`tail -n +2 | tr -d '\r' | while IFS=','...`).
- This is the only correct approach: both scripts read the **same file**, making network conditions physically identical.

**Verification:**
```
verify_stress_match.py → 102/102 exact match on Loss, Delay, Jitter (MaxDiff=0.000)
```

### New Scripts Added

| Script | Purpose |
|--------|---------|
| `scripts/evaluation/generate_stress_manifest.py` | Generates `scenario_manifest.csv` with seeded Python RNG |
| `scripts/evaluation/scenario_manifest.csv` | Pre-computed 102-step stress scenario (seed=20260101) |
| `scripts/evaluation/verify_stress_match.py` | Validates that two run dirs had identical network stress |
| `scripts/dataset_generation/audit_xgboost_leakage.py` | Detects label leakage and reports honest OOD accuracy |
| `scripts/evaluation/analyze_results_final.py` | Generates all 12 publication figures + summary_report.txt |
| `scripts/evaluation/run_natural_stress_baseline.sh` | Baseline run with manifest replay (replaces old version) |
| `scripts/evaluation/run_natural_stress_closedloop.sh` | Closed-loop run with manifest replay (replaces old version) |

### How to Re-run the Controlled Experiment

```bash
# 1. Generate manifest (only needed once, already committed)
python3 scripts/evaluation/generate_stress_manifest.py

# 2. Run baseline (~11 min)
bash scripts/evaluation/run_natural_stress_baseline.sh 660

# 3. Run closed-loop (~11 min)
bash scripts/evaluation/run_natural_stress_closedloop.sh 660

# 4. Pin dirs and generate all figures
BASELINE_DIR=$(ls -dt outputs/evaluation/baseline_natural_* | head -1)
CL_DIR=$(ls -dt outputs/evaluation/closedloop_natural_* | head -1)

# 5. Sanity check — both must show 102
grep -c "STRESS" "$BASELINE_DIR/stress.log"
grep -c "STRESS" "$CL_DIR/stress.log"

# 6. Verify stress match (must be 102/102 before proceeding)
PYTHONIOENCODING=utf-8 python3 scripts/evaluation/verify_stress_match.py \
    --baseline-dir "$BASELINE_DIR" --closedloop-dir "$CL_DIR"

# 7. Generate all 12 figures + summary report
PYTHONIOENCODING=utf-8 python3 scripts/evaluation/analyze_results_final.py \
    --baseline-dir "$BASELINE_DIR" --closedloop-dir "$CL_DIR" \
    --output-dir outputs/evaluation/figures_latest_run \
    --format png pdf --dpi 200
```

---

## XGBoost Model Audit (audit_xgboost_leakage.py)

### Finding: Label Leakage (Not Overfitting)
The 99% training accuracy is explained by **deterministic label leakage**:
- Labels are generated by the same `choose_network_state(loss, delay, jitter)` heuristic used to create features.
- A single decision stump on `loss×delay` and `loss×jitter` achieves ~97% accuracy.
- The model learns a **perfect decision rule** — no generalisation is required.

### Is This a Problem?
**No**, for system function. The model is the correct tool because:
- It adds **rolling temporal features** (means, slopes, acceleration) that the bare heuristic lacks.
- This provides **noise robustness** — single-spike misclassification prevention.
- **Anticipatory transitions** — slope/trend features detect deterioration 2–3 windows earlier.

### Honest Accuracy Numbers to Cite

| Evaluation | Accuracy |
|---|---|
| Synthetic training set (DO NOT CITE) | 99.0% — label leakage |
| 5-fold TimeSeriesSplit on live baseline telemetry | **99.67% ± 0.21%** |
| Out-of-Distribution (synthetic → live) | **98.9%** |
| Confusion matrix (live data): Critical recall | **97.7%** |

**Citation rule:** Always cite the live CV score (99.67%) and OOD score (98.9%). Describe the model as a "soft heuristic approximator with temporal smoothing."

---

## Final Controlled Experiment Results (May 10, 2026)

**Run pair:**
- Baseline: `baseline_natural_20260510_113738` (642 s)
- Closed-loop: `closedloop_natural_20260510_131407` (642 s)
- Stress verification: **102/102 steps identical** (MaxDiff Loss=0.000, Delay=0.000, Jitter=0.000)

### 1. Network Condition (same input stress, different adaptation)

| Metric | Baseline | Closed-Loop | Improvement |
|--------|----------|-------------|-------------|
| Packet Loss | 7.49% | 6.98% | −6.8% |
| Avg Delay | 85.6 ms | 71.1 ms | −16.9% |
| Jitter | 26.0 ms | 23.8 ms | −8.5% |
| Throughput | 23.9 kbps | 15.2 kbps | −36.4% (semantic saving) |
| Stable windows | 1081 / 2570 | **1305** / 2568 | **+20.7%** |
| Critical windows | 922 / 2570 | **824** / 2568 | **−10.6%** |

### 2. Throughput Savings by Network State

| State | Baseline | Closed-Loop | Bandwidth Saving |
|-------|----------|-------------|-----------------|
| Stable | 24.4 kbps | 13.6 kbps | **−44.5%** |
| Unstable | 25.7 kbps | 20.2 kbps | **−21.3%** |
| Critical | 22.0 kbps | 15.0 kbps | **−31.9%** |

### 3. ML Predictor Performance (Closed-Loop)
- Model active: **2537 / 2568 windows (98.8%)**
- Mean confidence: **0.9793** | Min: 0.5000
- Model-low-confidence fallbacks: 24 windows (0.9%)

### 4. Adaptation Latency
| Percentile | Latency |
|---|---|
| p50 | 266 ms |
| p90 | 754 ms |
| p95 | 756 ms |
| SEMANTIC_CRITICAL SLA (≤1000 ms) | **100.0%** |

### 5. Semantic Suppression
- Total samples generated: 181,710
- Sent: 128,971 (71.0%)
- **Suppressed (semantic gating): 42,422 (23.3%)**
- Net bandwidth reduction vs. raw transmission: ~29%

---

## Known Limitations — Cannot Be Code-Fixed

### #6 — No real hardware; synthetic OU physiology only
`PhysiologicalModel` uses Ornstein–Uhlenbeck. Real deployment requires hardware sensor drivers.

### #12 — Single-machine loopback; no true multi-hop wireless path
Network namespaces emulate a two-node topology. Real traffic traverses cellular/Wi-Fi with RF fading.

### #15 — `tc netem` parameters are fixed at experiment start
Dynamic mid-experiment netem changes require root + privileged subprocess bridge.

### #16 — Semantic codec trained only on ECG and BloodPressure
SpO2, Temperature, Respiration bypass semantic pipeline (`SEMANTIC_CAPABLE_TYPES = {"ECG", "BloodPressure"}`).

### #20 — No inter-device bandwidth fairness scheduling
Controller issues per-device commands but no fair-queuing/admission-control across total link capacity.

### #21 — PHY/MAC layer not modelled
UDP application layer only. No 802.11/LTE MAC model.

### #23 — No patient context in semantic payloads
16-dim z-vector encodes signal window only — no demographics/history.

### #25 — No cryptographic authentication (HMAC/TLS)
UDP packets unsigned. No HMAC-SHA256 or DTLS. No sensitive data transmitted.

### #28 — Ward controller decisions are rule-based, not learned
Static threshold rules. RL/model-predictive controller would improve performance.
