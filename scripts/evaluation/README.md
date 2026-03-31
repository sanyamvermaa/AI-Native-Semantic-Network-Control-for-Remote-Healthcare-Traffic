# Evaluation Scripts — README

## Overview

Three-step workflow to generate all paper figures comparing the **baseline
(static, no adaptation)** system against the **closed-loop (semantic) system**.

Both runs use identical netem stress profiles, the same physiological model,
and the same device layout — the only independent variable is whether the
semantic adaptation layer is active.

---

## Prerequisites

```bash
pip install matplotlib pandas numpy
# Already installed if you ran the training pipeline
```

Make scripts executable (Linux / WSL):

```bash
chmod +x scripts/evaluation/run_baseline_eval.sh
chmod +x scripts/evaluation/run_closedloop_eval.sh
```

---

## Step 1 — Baseline run (static, no adaptation)

```bash
bash scripts/evaluation/run_baseline_eval.sh 180 20
```

- **Duration**: 180 s total
- **Stage**: 20 s per netem profile
- **Profile cycle**: Stable → Unstable → Critical → Unstable → Stable → Critical

Output directory printed at end, e.g.:

```
outputs/evaluation/baseline_20260331_143000/
```

Files produced:

| File | Description |
|------|-------------|
| `network_telemetry.csv` | Per-250ms bandwidth, loss, delay, jitter |
| `sender_log_dev*.csv` | Every generated vital-sign sample with label |
| `sender_semantic_stats_dev*.csv` | Per-device transmission stats (all RAW) |
| `receiver.log` | Receiver stdout |
| `stress.log` | Netem profile applied at each stage |

---

## Step 2 — Closed-loop run (adaptive semantic system)

```bash
bash scripts/evaluation/run_closedloop_eval.sh 180 20
```

Same duration and netem cycle as Step 1.

Output directory, e.g.:

```
outputs/evaluation/closedloop_20260331_145000/
```

Additional files vs baseline:

| File | Description |
|------|-------------|
| `command_log.csv` | Every ward-controller decision + latency_ms |
| `ward_controller.log` | Ward controller stdout |

---

## Step 3 — Generate all 10 figures

```bash
python scripts/evaluation/analyze_results.py \
    --baseline-dir   outputs/evaluation/baseline_20260331_143000 \
    --closedloop-dir outputs/evaluation/closedloop_20260331_145000 \
    --output-dir     outputs/evaluation/figures
```

---

## Output: 10 figures + summary report

| File | Description |
|------|-------------|
| `01_bandwidth_over_time.png` | Throughput vs time, shaded by network state |
| `02_packet_loss_comparison.png` | Loss time-series: same netem, different response |
| `03_semantic_suppression.png` | Mode distribution; sent vs suppressed per mode |
| `04_adaptation_latency.png` | CDF of command-delivery latency |
| `05_command_distribution.png` | Pie chart: 9-case policy in action |
| `06_clinical_delivery.png` | Transmission rate + CRITICAL_ONLY retention |
| `07_bandwidth_savings_bar.png` | Mean throughput by state; % saving annotated |
| `08_telemetry_timeline.png` | Loss / delay / jitter: both runs overlaid |
| `09_network_state_agreement.png` | ML predictor vs rule-based heuristic |
| `10_suppression_over_time.png` | Rolling throughput with command-change markers |
| `summary_report.txt` | Numbers ready to drop into the results section |

---

## Folder layout after all 3 steps

```
scripts/
  evaluation/
    baseline_sender.py          ← static sender (no adaptation)
    run_baseline_eval.sh        ← Step 1
    run_closedloop_eval.sh      ← Step 2
    analyze_results.py          ← Step 3
    README.md

outputs/
  evaluation/
    baseline_<timestamp>/       ← Step 1 output
    closedloop_<timestamp>/     ← Step 2 output
    figures/                    ← 10 PNGs + summary_report.txt
```

---

## Troubleshooting

**Namespace setup fails (already exists from a previous run):**

```bash
sudo ip netns del sender_ns   2>/dev/null
sudo ip netns del receiver_ns 2>/dev/null
```

**Check receiver started correctly:**

```bash
tail -30 outputs/evaluation/baseline_.../receiver.log
```

**Check senders are transmitting:**

```bash
tail -20 outputs/evaluation/baseline_.../sender_0_ECG.log
```

**analyze_results.py shows empty plots:**  
Both `--baseline-dir` and `--closedloop-dir` must contain `network_telemetry.csv`.
Verify both experiments ran to completion before running Step 3.

---

## What makes the paper argument valid

`baseline_sender.py` uses the **identical** OU physiological model and clinical
device intervals as `health_sender.py`.  The only difference between the two
runs is whether the ward controller is running and whether senders adapt.

Therefore, any bandwidth saving shown in Figure 07 or any improvement in
clinical-to-noise ratio shown in Figure 06 is **entirely attributable to the
semantic layer** — not to different device profiles or different signal models.
