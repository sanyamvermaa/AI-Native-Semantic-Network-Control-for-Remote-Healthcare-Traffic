#!/usr/bin/env python3
"""
baseline_sender.py — Static vital-sign sender (no semantic adaptation).

Identical device profiles and OU physiological model to the closed-loop
health_sender.py, but transmits every sample in RAW mode at the base
interval with no ward-controller connection and no semantic filtering.

This is the control condition for evaluation: the only independent variable
between baseline and closed-loop runs is whether the semantic adaptation
layer is active.

Log output format is identical to health_sender.py so analyze_results.py
can compare both runs with the same code.

DETERMINISTIC SEED
──────────────────
Both this script and health_sender.py seed Python's `random` module with:
    PHYSIO_SEED_BASE + device_id

This guarantees that every device produces the *same* physiological
vital-sign sequence across the baseline run and the closed-loop run.
The only remaining experimental variable is therefore the semantic
adaptation layer itself — not random differences in generated vitals.

To use a different scenario epoch, pass --seed-base <int> (same value
must be used in both scripts).
"""

import argparse
import csv
import math
import os
import random
import socket
import time
from typing import Dict, Optional, Tuple

# Optional: neurokit2 — same as health_sender.py for fair comparison
try:
    import neurokit2 as _nk   # noqa: F401
    import numpy as _np       # noqa: F401
    _NK2_AVAILABLE = True
except ImportError:
    _NK2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Device profiles — identical to scripts/closed_loop/health_sender.py
# ---------------------------------------------------------------------------

DEVICE_PROFILES: Dict[str, Dict] = {
    "ECG": {
        "interval":         0.010,
        "normal":           (60, 100),
        "burst":            (110, 150),
        "unit":             "bpm",
        "burst_prob":       0.002,
        "burst_dur":        6,
        "summary_interval": 5.0,
    },
    "SpO2": {
        "interval":         1.0,
        "normal":           (95, 100),
        "burst":            (80, 94),
        "unit":             "%",
        "burst_prob":       0.002,
        "burst_dur":        5,
        "summary_interval": 30.0,
    },
    "BloodPressure": {
        "interval":         0.010,
        "normal":           (110, 130),
        "burst":            (160, 200),
        "unit":             "mmHg",
        "burst_prob":       0.001,
        "burst_dur":        8,
        "summary_interval": 10.0,
    },
    "Temperature": {
        "interval":         30.0,
        "normal":           (360, 373),
        "burst":            (380, 402),
        "unit":             "Cx10",
        "burst_prob":       0.001,
        "burst_dur":        10,
        "summary_interval": 120.0,
    },
    "Respiration": {
        "interval":         1.0,
        "normal":           (12, 20),
        "burst":            (25, 40),
        "unit":             "br/min",
        "burst_prob":       0.002,
        "burst_dur":        5,
        "summary_interval": 30.0,
    },
}

# ---------------------------------------------------------------------------
# Clinical thresholds — identical to scripts/closed_loop/health_sender.py
# ---------------------------------------------------------------------------

CLINICAL_THRESHOLDS: Dict[str, Dict] = {
    "ECG":           {"warn": 100, "critical": 130, "low_warn": 50,  "low_critical": 40},
    "SpO2":          {"warn": 94,  "critical": 90,  "low_warn": None, "low_critical": None},
    "BloodPressure": {"warn": 140, "critical": 160, "low_warn": 90,  "low_critical": 80},
    "Temperature":   {"warn": 376, "critical": 385, "low_warn": 355, "low_critical": 350},
    "Respiration":   {"warn": 22,  "critical": 28,  "low_warn": 8,   "low_critical": 5},
}

LOG_FLUSH_INTERVAL   = 1.0
STATS_FLUSH_INTERVAL = 1.0
MIN_INTERVAL         = 0.002


# ---------------------------------------------------------------------------
# PhysiologicalModel — same OU process as health_sender.py
# ---------------------------------------------------------------------------

class PhysiologicalModel:
    """
    Ornstein-Uhlenbeck mean-reverting stochastic process.
    Identical to the one in health_sender.py for a fair comparison.
    """

    _OU: Dict[str, Tuple[float, float]] = {
        "ECG":           (0.30, 4.0),
        "SpO2":          (0.50, 0.3),
        "BloodPressure": (0.25, 2.5),
        "Temperature":   (0.02, 0.2),
        "Respiration":   (0.35, 0.8),
    }

    def __init__(
        self,
        device_type: str,
        normal_range: Tuple[int, int],
        burst_range: Tuple[int, int],
        interval: float,
    ) -> None:
        self.device_type = device_type
        self.normal_lo = float(normal_range[0])
        self.normal_hi = float(normal_range[1])
        self.burst_lo  = float(burst_range[0])
        self.burst_hi  = float(burst_range[1])
        self.interval  = interval
        self.theta, self.sigma = self._OU.get(device_type, (0.25, 1.5))
        self._x  = (self.normal_lo + self.normal_hi) / 2.0
        self._mu = self._x

    def set_burst(self, active: bool) -> None:
        self._mu = (
            (self.burst_lo  + self.burst_hi)  / 2.0 if active
            else (self.normal_lo + self.normal_hi) / 2.0
        )

    def step(self) -> int:
        dt = self.interval
        self._x += (
            self.theta * (self._mu - self._x) * dt
            + self.sigma * math.sqrt(dt) * random.gauss(0.0, 1.0)
        )
        return int(round(self._x))


# ---------------------------------------------------------------------------
# ECGNeuroKitSource — same as health_sender.py
# ---------------------------------------------------------------------------

class ECGNeuroKitSource:
    """HRV-derived heart rate via neurokit2 — identical to health_sender.py."""

    _DURATION = 60
    _SR       = 100

    def __init__(self, base_hr: float = 72.0) -> None:
        from typing import List
        self._ou = PhysiologicalModel("ECG", (60, 100), (110, 150), 1.0 / self._SR)
        self._ou._x  = base_hr
        self._ou._mu = base_hr
        self._hr_series: List[int] = []
        self._pos = 0
        self._refill()

    def _refill(self) -> None:
        import neurokit2 as nk
        import numpy as np
        target_hr = max(40.0, min(220.0, self._ou._x))
        try:
            ecg = nk.ecg_simulate(
                duration=self._DURATION, sampling_rate=self._SR,
                heart_rate=target_hr, noise=0.05,
                random_state=random.randint(0, 9999),
            )
            _, info = nk.ecg_peaks(np.array(ecg), sampling_rate=self._SR)
            r_peaks = info["ECG_R_Peaks"]
            if len(r_peaks) > 1:
                rr_sec = np.diff(r_peaks).astype(float) / self._SR
                hr_beats = 60.0 / rr_sec
                hr_per_sample = np.full(self._DURATION * self._SR, target_hr, dtype=float)
                for i, rp in enumerate(r_peaks[:-1]):
                    hr_per_sample[rp:r_peaks[i + 1]] = hr_beats[i]
                self._hr_series = [int(round(v)) for v in hr_per_sample]
            else:
                self._hr_series = [int(round(target_hr))] * (self._DURATION * self._SR)
        except Exception:
            self._hr_series = [int(round(target_hr))] * (self._DURATION * self._SR)
        self._pos = 0

    def step(self, burst_active: bool) -> int:
        self._ou.set_burst(burst_active)
        self._ou.step()
        if self._pos >= len(self._hr_series):
            self._refill()
        val = self._hr_series[self._pos]
        self._pos += 1
        return val


# ---------------------------------------------------------------------------
# Clinical importance helper
# ---------------------------------------------------------------------------

def clinical_importance(device_type: str, value: int) -> float:
    thresh = CLINICAL_THRESHOLDS.get(device_type, {})
    hi_crit = thresh.get("critical")
    hi_warn = thresh.get("warn")
    lo_crit = thresh.get("low_critical")
    lo_warn = thresh.get("low_warn")
    if hi_crit is not None and value >= hi_crit:
        return 1.0
    if lo_crit is not None and value <= lo_crit:
        return 1.0
    if hi_warn is not None and value >= hi_warn:
        return 0.7
    if lo_warn is not None and value <= lo_warn:
        return 0.7
    return 0.2


# ---------------------------------------------------------------------------
# Default base dir (self-contained; does not import from common.py)
# ---------------------------------------------------------------------------

def _default_base_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get(
        "HEALTH_DATA_BASE_DIR",
        os.path.join(os.path.dirname(os.path.dirname(here)), "data", "logs"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Shared seed base — MUST match the value in health_sender.py.
# Change this integer to get a different-but-still-reproducible scenario epoch.
PHYSIO_SEED_BASE = 20260101


def main() -> None:
    parser = argparse.ArgumentParser(description="Static baseline vital-sign sender (no adaptation)")
    parser.add_argument("--device-id",     type=int, required=True)
    parser.add_argument("--device-type",   type=str, default="ECG", choices=list(DEVICE_PROFILES.keys()))
    parser.add_argument("--receiver-ip",   type=str, default="10.0.0.2")
    parser.add_argument("--receiver-port", type=int, default=9000)
    parser.add_argument("--base-dir",      type=str, default=None)
    parser.add_argument(
        "--seed-base", type=int, default=PHYSIO_SEED_BASE,
        help="Base random seed.  device_id is added to this value.  "
             "Must match the value passed to health_sender.py for a "
             "fair controlled experiment."
    )
    args = parser.parse_args()

    # ── Deterministic per-device seed ──────────────────────────────────────
    # Ensures this run produces an identical physiological sequence to the
    # corresponding closed-loop run, so the ONLY variable is adaptation.
    _seed = args.seed_base + args.device_id
    random.seed(_seed)
    print(f"[Baseline {args.device_id}|{args.device_type}] Physiological seed: {_seed}")
    # Note: neurokit2 / numpy will still have independent seeds.  We seed
    # numpy here too for the ECGNeuroKitSource path.
    try:
        import numpy as _np_seed
        _np_seed.random.seed(_seed)
    except ImportError:
        pass

    if args.base_dir is None:
        args.base_dir = _default_base_dir()

    profile  = DEVICE_PROFILES[args.device_type]
    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    # Physiological signal source — same as health_sender.py
    if args.device_type == "ECG" and _NK2_AVAILABLE:
        _ecg_src = ECGNeuroKitSource(base_hr=80.0)
        def _next_value(burst_active: bool) -> int:
            return _ecg_src.step(burst_active)
        print(f"[Baseline {args.device_id}|ECG] Using neurokit2 HRV-derived heart rate.")
    else:
        _physio = PhysiologicalModel(
            args.device_type, profile["normal"], profile["burst"], profile["interval"]
        )
        def _next_value(burst_active: bool) -> int:
            _physio.set_burst(burst_active)
            return _physio.step()
        if args.device_type == "ECG":
            print(f"[Baseline {args.device_id}|ECG] neurokit2 not found; using OU process.")
        else:
            print(f"[Baseline {args.device_id}|{args.device_type}] Using OU physiological model.")

    log_file   = os.path.join(base_dir, f"sender_log_dev{args.device_id}_{args.device_type}.csv")
    stats_file = os.path.join(base_dir, f"sender_semantic_stats_dev{args.device_id}_{args.device_type}.csv")

    tx_sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver = (args.receiver_ip, args.receiver_port)

    log_f  = open(log_file,   "w", newline="", encoding="utf-8")
    stat_f = open(stats_file, "w", newline="", encoding="utf-8")

    log_writer   = csv.writer(log_f)
    stats_writer = csv.writer(stat_f)

    log_writer.writerow(["seq", "timestamp", "device_id", "device_type", "value", "unit", "label"])
    stats_writer.writerow([
        "timestamp", "device_id", "device_type",
        "current_command", "current_mode",
        "total_samples", "total_sent", "total_suppressed",
        "raw_sent", "raw_suppressed",
        "delta_sent", "delta_suppressed",
        "summary_sent", "summary_suppressed",
        "critical_only_sent", "critical_only_suppressed",
    ])
    log_f.flush()
    stat_f.flush()

    burst_mode     = False
    burst_end_time = 0.0
    sample_seq     = 0
    tx_seq         = 0
    raw_sent       = 0

    last_log_flush   = time.time()
    last_stats_flush = time.time()

    interval = max(MIN_INTERVAL, profile["interval"])

    print(
        f"[Baseline {args.device_id}|{args.device_type}] Sender started -> {receiver}, "
        f"mode=RAW (static) interval={interval}s"
    )

    try:
        while True:
            now = time.time()

            # Burst logic — identical to health_sender.py
            if burst_mode and now > burst_end_time:
                burst_mode = False
            if not burst_mode and random.random() < profile["burst_prob"]:
                burst_mode     = True
                burst_end_time = now + profile["burst_dur"]

            # Physiological value — same OU model
            value = _next_value(burst_mode)

            # Label via clinical importance — identical to health_sender.py
            importance = clinical_importance(args.device_type, value)
            if burst_mode:
                label = "CRITICAL"
            elif importance >= 0.7:
                label = "ALERT"
            else:
                label = "NORMAL"

            sample_seq += 1

            # Always transmit in RAW mode — no semantic filtering
            payload = (
                f"{args.device_id},{sample_seq},{now:.6f},{args.device_type},{value},{label}"
                .ljust(64)
                .encode("utf-8")
            )
            try:
                tx_sock.sendto(payload, receiver)
                tx_seq += 1
                raw_sent += 1
            except Exception:
                pass

            log_writer.writerow([
                sample_seq, now,
                args.device_id, args.device_type,
                value, profile["unit"], label,
            ])

            if now - last_log_flush >= LOG_FLUSH_INTERVAL:
                log_f.flush()
                last_log_flush = now

            if now - last_stats_flush >= STATS_FLUSH_INTERVAL:
                # Write accumulated stats: all packets are in RAW mode
                stats_writer.writerow([
                    round(now, 6),
                    args.device_id, args.device_type,
                    "FULL_ECG", "RAW",          # static sender always uses FULL_ECG / RAW
                    sample_seq, tx_seq, 0,       # total_suppressed = 0
                    raw_sent, 0,                 # raw_suppressed = 0
                    0, 0,                        # delta: none
                    0, 0,                        # summary: none
                    0, 0,                        # critical_only: none
                ])
                stat_f.flush()
                last_stats_flush = now

            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        log_f.close()
        stat_f.close()
        tx_sock.close()
        print(f"[Baseline {args.device_id}|{args.device_type}] Sender stopped. Sent {tx_seq} packets.")


if __name__ == "__main__":
    main()
