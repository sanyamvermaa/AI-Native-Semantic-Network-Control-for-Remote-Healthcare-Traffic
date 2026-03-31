#!/usr/bin/env python3
"""
Generate balanced synthetic sender logs for semantic codec training.

Produces CSV files in the same format as health_sender.py log output:
    seq, timestamp, device_id, device_type, value, unit, label

Strategy per device type:
  ECG          — NeuroKit2 nk.ecg_simulate() heart-rate extraction,
                 OU process around three operating points (normal/alert/critical)
  BloodPressure — OU process around three operating points

Target: 10,000 rows per phase × 3 phases = 30,000 rows per device type,
        balanced ~33% NORMAL / ~33% ALERT / ~33% CRITICAL.

Output: data/logs/sender_log_dev{id}_synthetic_{type}.csv
        (These are picked up automatically by train_semantic_codec.py
         because load_device_logs globs on sender_log_dev*_{type}.csv)

Usage (from project root, WSL):
    python3 scripts/semantic/generate_training_data.py
"""

import csv
import os
import random
import sys
import time
from typing import Iterator, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CLOSED_LOOP  = os.path.join(_PROJECT_ROOT, "scripts", "closed_loop")
sys.path.insert(0, _CLOSED_LOOP)

from health_sender import CLINICAL_THRESHOLDS, DEVICE_PROFILES  # type: ignore
print("DEBUG: BloodPressure CLINICAL_THRESHOLDS =", CLINICAL_THRESHOLDS.get("BloodPressure", {}))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROWS_PER_PHASE  = 10_000   # per class (NORMAL / ALERT / CRITICAL)
CHUNK_SIZE      = 300      # smaller → more frequent transitions
NUM_CHUNKS      = ROWS_PER_PHASE // CHUNK_SIZE   # = 33 cycles through N/A/C
DEVICE_ID_START = 10       # avoid colliding with real device ids 0-7
OUTPUT_DIR      = os.path.join(_PROJECT_ROOT, "data", "logs")

# ECG: NeuroKit2 heart-rate extraction sampling rate
ECG_NK_SF = 100  # sample rate for nk.ecg_simulate (Hz)

# ---------------------------------------------------------------------------
# OU process
# ---------------------------------------------------------------------------

def ou_stream(
    mu: float,
    sigma: float,
    theta: float,
    n: int,
    x0: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Ornstein-Uhlenbeck process: mean-reverting with Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()
    x = mu if x0 is None else x0
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x += theta * (mu - x) + sigma * rng.standard_normal()
        out[i] = x
    return out


def _label(value: float, device_type: str) -> str:
    t = CLINICAL_THRESHOLDS[device_type]
    if (t.get("critical") is not None and value >= t["critical"]) or \
       (t.get("low_critical") is not None and value <= t["low_critical"]):
        return "CRITICAL"
    if (t.get("warn") is not None and value >= t["warn"]) or \
       (t.get("low_warn") is not None and value <= t["low_warn"]):
        return "ALERT"
    return "NORMAL"

# ---------------------------------------------------------------------------
# ECG generator  (heart-rate in bpm via NeuroKit2)
# ---------------------------------------------------------------------------

def _ecg_hr_from_nk(n_samples: int, hr_bpm: float, rng) -> np.ndarray:
    """
    Use NeuroKit2 to simulate an ECG at hr_bpm, then detect R-peaks and
    compute instantaneous HR per sample.  Returns array of length n_samples.

    Falls back to OU around hr_bpm if NeuroKit2 fails.
    """
    try:
        import neurokit2 as nk
        # Simulate enough seconds to cover n_samples at ECG_NK_SF
        duration_s = max(int(n_samples / ECG_NK_SF) + 30, 60)
        ecg_sig = nk.ecg_simulate(
            duration     = duration_s,
            sampling_rate= ECG_NK_SF,
            heart_rate   = hr_bpm,
            noise        = 0.05,
            random_state = int(rng.integers(0, 9999)),
        )
        _, info = nk.ecg_peaks(ecg_sig, sampling_rate=ECG_NK_SF)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 3:
            raise ValueError("too few R-peaks")

        # Instantaneous HR per R-R interval
        rr_intervals = np.diff(rpeaks) / ECG_NK_SF   # seconds
        inst_hr      = 60.0 / rr_intervals            # bpm

        # Interpolate to n_samples
        xp = rpeaks[1:].astype(float)
        total_pts = len(ecg_sig)
        xi = np.linspace(0, total_pts - 1, n_samples)
        hr_interp = np.interp(xi, xp, inst_hr)

        # Add small OU jitter (~2 bpm sigma) for realism
        jitter = ou_stream(0.0, 2.0, 0.3, n_samples, x0=0.0, rng=rng)
        return (hr_interp + jitter).astype(np.float32)

    except Exception as exc:
        # Fallback: plain OU
        print(f"    [WARN] NeuroKit2 ECG sim failed ({exc}), using OU fallback")
        sigma = max(hr_bpm * 0.04, 2.0)
        return ou_stream(hr_bpm, sigma, 0.08, n_samples, rng=rng)


def generate_ecg_phase(
    phase: str,
    n: int,
    device_id: int,
    seq_start: int,
    ts_start: float,
    rng,
) -> List[Tuple]:
    """
    Generate n ECG rows for a given clinical phase.

    phase    hr_target (bpm)   OU sigma
    NORMAL   75                3
    ALERT    110               5   (tachycardia warning zone 100-129)
    CRITICAL 140               6   (severe tachycardia ≥130)
    """
    INTERVAL = DEVICE_PROFILES["ECG"]["interval"]  # 0.01 s

    targets = {
        "NORMAL":   (75.0,  5.0, 0.08),
        "ALERT":    (112.0, 6.0, 0.08),
        "CRITICAL": (145.0, 7.0, 0.06),
    }
    mu, sigma, theta = targets[phase]
    # Per-chunk jitter — each call gets a slightly different operating point
    # so train/val/test windows all have different statistical fingerprints
    mu    += float(rng.uniform(-5.0, 5.0))
    sigma  = max(sigma + float(rng.uniform(-0.8, 1.2)), 1.0)
    theta  = max(theta + float(rng.uniform(-0.02, 0.02)), 0.01)
    mu     = float(np.clip(mu, 20.0, 220.0))

    print(f"    ECG {phase}: NeuroKit2 HR synthesis at {mu:.1f} bpm ...")
    raw = _ecg_hr_from_nk(n, mu, rng)

    # Clip to physiologically plausible range
    raw = np.clip(raw, 20.0, 220.0)

    rows = []
    ts   = ts_start
    for i, v in enumerate(raw):
        label = _label(float(v), "ECG")
        rows.append((
            seq_start + i + 1,
            round(ts, 6),
            device_id,
            "ECG",
            round(float(v), 1),
            "bpm",
            label,
        ))
        ts += INTERVAL
    return rows


# ---------------------------------------------------------------------------
# BloodPressure generator  (pure OU — clinically IBP waveform is sinusoidal
#  but for HR-level features an OU mean-reverting model is sufficient)
# ---------------------------------------------------------------------------

def generate_bp_phase(
    phase: str,
    n: int,
    device_id: int,
    seq_start: int,
    ts_start: float,
    rng,
) -> List[Tuple]:
    """
    Improved version — no hypo mixing, safer targets, more realistic noise.
    """
    INTERVAL = DEVICE_PROFILES["BloodPressure"]["interval"]

    targets = {
        "NORMAL":   (115.0, 4.0, 0.10),
        "ALERT":    (148.0, 3.0, 0.10),
        "CRITICAL": (195.0, 5.0, 0.07),
    }
    mu, sigma, theta = targets[phase]
    # Per-chunk jitter — breaks statistical fingerprint
    mu    += float(rng.uniform(-4.0, 4.0))
    sigma  = max(sigma + float(rng.uniform(-0.5, 0.8)), 0.5)
    theta  = max(theta + float(rng.uniform(-0.02, 0.02)), 0.01)

    # NO hypo mixing — pure hypertensive CRITICAL
    raw = ou_stream(mu, sigma, theta, n, rng=rng)

    raw = np.clip(raw, 40.0, 250.0)

    rows = []
    ts   = ts_start
    for i, v in enumerate(raw):
        label = _label(float(v), "BloodPressure")
        rows.append((
            seq_start + i + 1,
            round(ts, 6),
            device_id,
            "BloodPressure",
            round(float(v), 1),
            "mmHg",
            label,
        ))
        ts += INTERVAL
    return rows


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

HEADER = ["seq", "timestamp", "device_id", "device_type", "value", "unit", "label"]

def write_csv(rows: List[Tuple], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"  Wrote {len(rows):,} rows → {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _interleave_chunks(phase_rows: dict) -> List[Tuple]:
    """
    Given a dict {phase: [rows...]}, interleave in CHUNK_SIZE blocks:
      N[0:CHUNK], A[0:CHUNK], C[0:CHUNK], N[CHUNK:2*CHUNK], ...
    This keeps local temporal coherence within each chunk so that sliding
    windows of width <= CHUNK_SIZE can land entirely within one phase.
    """
    result = []
    for chunk_i in range(NUM_CHUNKS):
        s, e = chunk_i * CHUNK_SIZE, (chunk_i + 1) * CHUNK_SIZE
        for phase in ["NORMAL", "ALERT", "CRITICAL"]:
            result.extend(phase_rows[phase][s:e])
    return result


def main() -> None:
    rng = np.random.default_rng(seed=42)
    random.seed(42)

    ts_base = time.time()  # current wall-clock as base timestamp

    # --- ECG: two synthetic devices (ids 10, 11) ---
    for dev_offset, dev_id in enumerate([10, 11]):
        print(f"\n=== Synthetic ECG device {dev_id} ===")
        interval = DEVICE_PROFILES["ECG"]["interval"]

        # Generate each phase as one large array — only 3 NeuroKit2 calls per
        # device (same as before).  Interleaving is done in pure Python after.
        phase_rows: dict = {}
        seq = 0
        ts  = ts_base + dev_offset * 0.001
        for phase in ["NORMAL", "ALERT", "CRITICAL"]:
            rows = generate_ecg_phase(phase, ROWS_PER_PHASE, dev_id, seq, ts, rng)
            phase_rows[phase] = rows
            seq += ROWS_PER_PHASE
            if rows:
                ts = rows[-1][1] + interval

        # Interleave in CHUNK_SIZE blocks so sliding windows see frequent
        # NORMAL→ALERT→CRITICAL transitions.  Do NOT shuffle individual rows —
        # the cyclic structure is what gives pure-class windows within chunks.
        all_rows = _interleave_chunks(phase_rows)

        # Re-assign monotonic seq and timestamp
        ts = ts_base + dev_offset * 0.001
        all_rows = [
            (i + 1, round(ts + i * interval, 6), r[2], r[3], r[4], r[5], r[6])
            for i, r in enumerate(all_rows)
        ]

        from collections import Counter
        dist = Counter(r[6] for r in all_rows)
        print(f"  Label dist: { {k: v for k,v in dist.items()} }")

        out_path = os.path.join(OUTPUT_DIR, f"sender_log_dev{dev_id}_ECG.csv")
        write_csv(all_rows, out_path)

    # --- BloodPressure: two synthetic devices (ids 12, 13) ---
    for dev_offset, dev_id in enumerate([12, 13]):
        print(f"\n=== Synthetic BloodPressure device {dev_id} ===")
        interval = DEVICE_PROFILES["BloodPressure"]["interval"]

        phase_rows = {}
        seq = 0
        ts  = ts_base + dev_offset * 0.001
        for phase in ["NORMAL", "ALERT", "CRITICAL"]:
            rows = generate_bp_phase(phase, ROWS_PER_PHASE, dev_id, seq, ts, rng)
            phase_rows[phase] = rows
            seq += ROWS_PER_PHASE
            if rows:
                ts = rows[-1][1] + interval

        all_rows = _interleave_chunks(phase_rows)

        ts = ts_base + dev_offset * 0.001
        all_rows = [
            (i + 1, round(ts + i * interval, 6), r[2], r[3], r[4], r[5], r[6])
            for i, r in enumerate(all_rows)
        ]

        from collections import Counter
        dist = Counter(r[6] for r in all_rows)
        print(f"  Label dist: { {k: v for k,v in dist.items()} }")

        out_path = os.path.join(OUTPUT_DIR, f"sender_log_dev{dev_id}_BloodPressure.csv")
        write_csv(all_rows, out_path)

    print("\n=== Generation complete ===")
    print("Run: python3 scripts/semantic/train_semantic_codec.py")


if __name__ == "__main__":
    main()
