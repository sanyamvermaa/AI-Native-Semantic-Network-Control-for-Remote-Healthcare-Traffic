"""
generate_synthetic_dataset.py
──────────────────────────────
Generates a clean synthetic training dataset for the network state classifier.

Key design principles vs. the original realistic_network_dataset.csv:
  1. Features are in MILLISECONDS (matching what health_receiver.py feeds the model).
  2. Labels are computed directly from choose_network_state() — NOT from scenario metadata.
     So label = what-the-heuristic-says = model will learn to match the heuristic.
  3. Temporal continuity: each synthetic "run" interpolates smoothly through phases
     so rolling features (mean, std, slope) are meaningful.
  4. Values match the natural_stress closedloop scenario ranges exactly.

Output: data/datasets/synthetic_control_dataset.csv
"""

import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV   = PROJECT_ROOT / "data" / "datasets" / "synthetic_control_dataset.csv"

# ── Heuristic (must match common.py choose_network_state) ────────────────────
def choose_network_state(loss_rate, delay_ms, jitter_ms):
    if loss_rate >= 0.08 or delay_ms >= 130 or jitter_ms >= 35:
        return "Critical"
    elif loss_rate >= 0.03 or delay_ms >= 35 or jitter_ms >= 8:
        return "Unstable"
    return "Stable"

# ── Stress scenario phases (loss %, delay ms, jitter ms) ─────────────────────
# Matches the natural_stress closedloop script exactly.
PHASES = [
    # (n_windows, loss_pct, delay_ms, jitter_ms, noise_loss, noise_delay, noise_jitter)
    (320, 0.3,  12.0,  2.0,  0.2,  3.0,  0.8),   # Phase 1 — Stable morning
    (160, 5.0,  70.0, 14.0,  0.8,  8.0,  2.0),   # Phase 2 — Unstable drift
    ( 80, 2.0,  35.0,  7.0,  0.5,  5.0,  1.5),   # Phase 3 — Partial recovery
    (200, 14.0,200.0, 55.0,  2.0, 20.0,  8.0),   # Phase 4 — Escalation
    ( 64, 18.0,260.0, 70.0,  2.5, 25.0, 10.0),   # Phase 5 — Peak critical
    (200, 4.0,  55.0, 11.0,  0.8,  8.0,  2.0),   # Phase 6 — Slow recovery
    ( 96, 0.5,  15.0,  3.0,  0.3,  3.0,  0.8),   # Phase 7 — Stable recovery
    ( 36, 20.0,220.0, 72.0,  3.0, 25.0, 12.0),   # Phase 8 — Sharp spike
    ( 80, 1.0,  20.0,  4.0,  0.4,  3.0,  1.0),   # Phase 9 — Fast recovery
    ( 64, 0.2,  10.0,  2.0,  0.2,  2.0,  0.5),   # Phase 10 — Stable finish
]

def generate_run(run_id, bw_base=20_000.0):
    """Generate one full stress-scenario run, returning a list of window dicts."""
    rows = []
    t = run_id * 1200.0   # each run starts at different timestamp

    for n_win, loss_pct, delay_ms, jitter_ms, nl, nd, nj in PHASES:
        # Interpolate into this phase over ~10% of its windows (smooth transition)
        prev_loss  = rows[-1]["packet_loss_rate"] * 100.0 if rows else loss_pct
        prev_delay = rows[-1]["avg_delay"]         if rows else delay_ms
        prev_jitter= rows[-1]["jitter"]             if rows else jitter_ms

        for i in range(n_win):
            frac = min(1.0, i / max(1, n_win // 8))  # first 12% of windows: ramp
            l = (prev_loss  + (loss_pct  - prev_loss)  * frac) / 100.0
            d = (prev_delay + (delay_ms  - prev_delay) * frac)
            j = (prev_jitter+ (jitter_ms - prev_jitter)* frac)

            # Add noise (clamp to non-negative)
            l_noise = np.random.normal(0, nl / 100.0)
            d_noise = np.random.normal(0, nd)
            j_noise = np.random.normal(0, nj)

            loss  = float(np.clip(l + l_noise,  0.0, 0.99))
            delay = float(np.clip(d + d_noise,  0.0, 500.0))
            jitter= float(np.clip(j + j_noise,  0.0, 150.0))

            bw = bw_base * random.uniform(0.8, 1.2)
            thr = bw * (1.0 - loss)
            pkts = int(random.uniform(25, 55))

            rows.append({
                "timestamp":         round(t, 3),
                "active_devices":    random.randint(4, 12),
                "bandwidth_usage_bps": round(bw, 2),
                "throughput_bps":    round(thr, 2),
                "packet_loss_rate":  round(loss, 6),
                "jitter":            round(jitter, 4),
                "avg_delay":         round(delay, 4),
                "packets_per_window": pkts,
            })
            t += 0.25  # 250ms telemetry interval

    return rows


# ── Run generator N times to build a large balanced dataset ──────────────────
N_RUNS = 60   # ~60 × ~1300 windows = ~78,000 rows before label filter
all_rows = []
for run_id in range(N_RUNS):
    all_rows.extend(generate_run(run_id))

print(f"[+] Generated {len(all_rows)} raw windows across {N_RUNS} runs")

df = pd.DataFrame(all_rows)

# ── Compute rolling/engineered features (same as train_model.py) ─────────────
W = 10
df = df.sort_values("timestamp").reset_index(drop=True)

df["rolling_loss_mean"]       = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"]        = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"]     = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"]      = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"]  = df["throughput_bps"].rolling(W).std()
df["loss_delta"]   = df["packet_loss_rate"].diff()
df["jitter_delta"] = df["jitter"].diff()
df["delay_delta"]  = df["avg_delay"].diff()
df["loss_accel"]   = df["loss_delta"].diff()
df["loss_trend_3"]       = df["packet_loss_rate"].rolling(3).mean() - df["packet_loss_rate"].rolling(8).mean()
df["throughput_trend_3"] = df["throughput_bps"].rolling(3).mean() - df["throughput_bps"].rolling(8).mean()
df["delay_trend_3"]      = df["avg_delay"].rolling(3).mean() - df["avg_delay"].rolling(8).mean()

xs = np.arange(8)
df["delay_slope"] = df["avg_delay"].rolling(8).apply(lambda y: np.polyfit(xs, y, 1)[0], raw=True)
df["loss_slope"]  = df["packet_loss_rate"].rolling(8).apply(lambda y: np.polyfit(xs, y, 1)[0], raw=True)

df["loss_x_delay"]  = df["packet_loss_rate"] * df["avg_delay"]
df["loss_x_jitter"] = df["packet_loss_rate"] * df["jitter"]

df = df.dropna().reset_index(drop=True)
print(f"[+] After NaN drop: {len(df)} rows")

# ── Assign labels using the heuristic (NOT scenario metadata) ─────────────────
df["network_condition"] = df.apply(
    lambda r: choose_network_state(r["packet_loss_rate"], r["avg_delay"], r["jitter"]),
    axis=1
)

print(f"[+] Label distribution (heuristic):\n{df['network_condition'].value_counts().to_string()}\n")

# ── Balance classes (contiguous block undersampling to preserve time series) ──
counts = df["network_condition"].value_counts()
min_count = counts.min()
print(f"[+] Balancing to {min_count} per class ...")

balanced_parts = []
for cls in ["Stable", "Unstable", "Critical"]:
    sub = df[df["network_condition"] == cls].copy()
    if len(sub) > min_count:
        # Keep contiguous blocks: take every Nth row uniformly spread
        indices = np.linspace(0, len(sub) - 1, min_count, dtype=int)
        sub = sub.iloc[indices].reset_index(drop=True)
    balanced_parts.append(sub)

df_balanced = pd.concat(balanced_parts).sort_values("timestamp").reset_index(drop=True)
print(f"[+] Balanced dataset: {len(df_balanced)} rows")
print(df_balanced["network_condition"].value_counts().to_string())

# ── Save ─────────────────────────────────────────────────────────────────────
df_balanced.to_csv(OUTPUT_CSV, index=False)
print(f"\n[+] Saved to {OUTPUT_CSV}")
