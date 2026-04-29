"""
Adaptive closed-loop semantic sender.

Each sender transmits vital signs to the receiver and listens for control
commands from ward_controller to adapt both rate and payload semantics in real time.

Transmission modes:
  RAW           — every sample, unfiltered
  DELTA         — only when value changes beyond a threshold
  SUMMARY       — one compressed semantic packet per summary_interval seconds
  CRITICAL_ONLY — suppress normal readings; only transmit clinically urgent ones

CHANGELOG (fixes applied):
  [PROFILE] SpO2 interval 0.020s → 1.0s  (clinical 1 Hz)
  [PROFILE] BloodPressure interval 0.050s → 0.010s  (arterial-line 100 Hz, documented)
  [PROFILE] Temperature interval 0.100s → 30.0s  (body temp changes over minutes)
  [PROFILE] Respiration interval 0.020s → 1.0s  (clinical 1 Hz)
  [PROFILE] Added summary_interval per device so SUMMARY rate is time-gated,
            not sample-gated, eliminating the broken double-gating logic.
  [SEMANTIC] SEMANTIC_CRITICAL multiplier 0.5 → 1.0  (was doubling traffic on bad network)
  [SEMANTIC] SEMANTIC_ALERT multiplier 1.0 → 1.5  (slight rate reduction + delta filter)
  [SEMANTIC] DOWNSAMPLED_ECG multiplier for non-ECG devices overridden to 1.0
             (command is ECG-specific; other devices hold their natural rate)
  [SEMANTIC] DOWNSAMPLED_ECG integer ratio 3.5 → 4.0  (clean 100Hz→25Hz)
  [LOGIC]   Burst-end check now happens before value assignment, fixing the off-by-one
            where the last burst tick still emitted a burst value after end time.
  [LOGIC]   Label assignment now uses clinical_importance() instead of fragile
            high * 0.95 float comparison that mislabelled SpO2 values.
  [LOGIC]   SUMMARY mode now time-gated (last_summary_emit + summary_interval)
            instead of sample-gated (sample_seq % window), removing the double-gate
            that made Temperature send a summary every ~16 minutes.
  [LOGIC]   mode_counters["suppressed"] no longer inflated by intentional SUMMARY
            skips; only counts packets that were semantically filtered.
  [PERF]    log_file opened once and flushed on interval, not on every packet
            (was 100+ file opens/second for ECG).
  [PERF]    stats_file also opened once and flushed on interval.
"""

import argparse
import collections
import csv
import json
import math
import os
import random
import socket
import sys
import time
from typing import Deque, Dict, List, Optional, Tuple

from common import default_base_dir

# Optional: neurokit2 gives realistic ECG waveform → HRV-derived heart rate.
# Falls back to the OU process if not installed.
try:
    import neurokit2 as _nk   # noqa: F401 — only used inside ECGNeuroKitSource
    import numpy as _np       # noqa: F401
    _NK2_AVAILABLE = True
except ImportError:
    _NK2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional semantic codec imports
# ---------------------------------------------------------------------------
_SEMANTIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "semantic")
if _SEMANTIC_DIR not in sys.path:
    sys.path.insert(0, _SEMANTIC_DIR)

try:
    from semantic_encoder import SemanticEncoder          # type: ignore
    from channel_quantizer import encode_payload, LATENT_DIMS_BY_COMMAND  # type: ignore
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Device profiles
# ---------------------------------------------------------------------------
# BloodPressure note: simulating invasive arterial-line (IBP) at 100 Hz.
#   For cuff-based NIBP, interval would be 60–300s. Document your assumption.
# Temperature note: 30s interval reflects continuous wearable patch monitoring.
#   Hospital spot-checks would be 60–300s; adjust to your simulation fidelity.

DEVICE_PROFILES: Dict[str, Dict] = {
    "ECG": {
        "interval":         0.010,   # 100 Hz — simulation rate (clinical: 250–500 Hz)
        "normal":           (60, 100),
        "burst":            (110, 150),
        "unit":             "bpm",
        "burst_prob":       0.002,
        "burst_dur":        6,
        "summary_interval": 5.0,     # seconds between SUMMARY packets in SUMMARY mode
    },
    "SpO2": {
        "interval":         1.0,     # 1 Hz — clinical standard for pulse oximetry
        "normal":           (95, 100),
        "burst":            (80, 94),
        "unit":             "%",
        "burst_prob":       0.002,
        "burst_dur":        5,
        "summary_interval": 30.0,
    },
    "BloodPressure": {
        "interval":         0.010,   # 100 Hz — arterial-line (IBP) simulation
        "normal":           (110, 130),
        "burst":            (160, 200),
        "unit":             "mmHg",
        "burst_prob":       0.001,
        "burst_dur":        8,
        "summary_interval": 10.0,
    },
    "Temperature": {
        "interval":         30.0,    # 1 sample/30s — body temp changes over minutes
        "normal":           (360, 373),
        "burst":            (380, 402),
        "unit":             "Cx10",
        "burst_prob":       0.001,
        "burst_dur":        10,
        "summary_interval": 120.0,
    },
    "Respiration": {
        "interval":         1.0,     # 1 Hz — standard respiratory rate monitoring
        "normal":           (12, 20),
        "burst":            (25, 40),
        "unit":             "br/min",
        "burst_prob":       0.002,
        "burst_dur":        5,
        "summary_interval": 30.0,
    },
}

# ---------------------------------------------------------------------------
# Clinical thresholds (used for importance scoring and ALERT labelling)
# ---------------------------------------------------------------------------

CLINICAL_THRESHOLDS: Dict[str, Dict] = {
    "ECG":           {"warn": 100, "critical": 130, "low_warn": 50,  "low_critical": 40},
    # SpO2: danger is LOW values only — below 94% is warning, below 90% is critical.
    # warn/critical are None because 100% SpO2 is fine — no upper danger bound.
    "SpO2":          {"warn": None, "critical": None, "low_warn": 94, "low_critical": 90},
    "BloodPressure": {"warn": 140, "critical": 160, "low_warn": 90,  "low_critical": 80},
    "Temperature":   {"warn": 376, "critical": 385, "low_warn": 355, "low_critical": 350},
    "Respiration":   {"warn": 22,  "critical": 28,  "low_warn": 8,   "low_critical": 5},
}

# ---------------------------------------------------------------------------
# Command → (transmission mode, interval multiplier)
# ---------------------------------------------------------------------------
# Multipliers are applied to base_interval.  For DOWNSAMPLED_ECG, non-ECG
# devices are held at 1.0× (handled in the command-processing block below).

COMMAND_SEMANTIC_MAP: Dict[str, Tuple[str, float]] = {
    "FULL_ECG":           ("RAW",           1.0),   # full rate, every sample
    "FULL_ECG_PRIORITY":  ("RAW",           0.8),   # slightly faster than normal
    "DOWNSAMPLED_ECG":    ("DELTA",         4.0),   # ECG: 100Hz→25Hz (integer ratio)
                                                    # non-ECG: overridden to 1.0× below
    "SEMANTIC_ALERT":     ("DELTA",         1.5),   # rate slows + delta filter
    "SEMANTIC_CRITICAL":  ("CRITICAL_ONLY", 1.0),   # hold rate; suppression does the work
    "SEMANTIC_SUMMARY":   ("SUMMARY",       8.0),   # slow base rate + time-gated summaries
}

LOG_FLUSH_INTERVAL  = 1.0   # seconds between log file flushes
STATS_FLUSH_INTERVAL = 1.0  # seconds between stats file flushes
MIN_INTERVAL        = 0.002 # hard floor on transmission interval (500 Hz max)

# Devices that have trained semantic encoder/decoder models.
# SpO2, Temperature, Respiration have no trained models — they stay raw-only.
SEMANTIC_CAPABLE_TYPES: frozenset = frozenset({"ECG", "BloodPressure"})

# Per-device SemanticBuffer window sizes (number of samples to accumulate).
# ECG/BP at 100 Hz: 200 samples = 2 s of context, matching the codec window.
# SpO2/Respiration at 1 Hz: 30 samples = 30 s.
# Temperature at 1/30 Hz: 4 samples = 2 minutes.
SEMANTIC_BUFFER_WINDOWS: Dict[str, int] = {
    "ECG":           200,
    "BloodPressure": 200,
    "SpO2":           30,
    "Temperature":     4,
    "Respiration":    30,
}


# ---------------------------------------------------------------------------
# SemanticBuffer
# ---------------------------------------------------------------------------

class SemanticBuffer:
    """
    Accumulates raw samples and supports:
      - clinical_importance()    : continuous score 0.0–1.0 for a single reading
      - should_transmit_delta()  : true if value changed enough to warrant sending
      - get_semantic_summary()   : compress window into min/max/mean/trend packet
    """

    def __init__(self, device_type: str, window: int = 20) -> None:
        self.device_type = device_type
        self.window = window
        self.buffer: Deque[float] = collections.deque(maxlen=window)
        self.last_transmitted_value: Optional[float] = None
        self.thresholds = CLINICAL_THRESHOLDS.get(device_type, {})

    def clinical_importance(self, value: float) -> float:
        """
        Return a continuous importance score in [0.0, 1.0].

        Piecewise linear interpolation:
          deep normal                 → 0.2
          normal ← → low_warn/warn   → interpolated 0.2 → 0.7
          low_warn/warn ← → critical → interpolated 0.7 → 1.0
          at or beyond critical       → 1.0
        """
        hi_crit = self.thresholds.get("critical")
        hi_warn = self.thresholds.get("warn")
        lo_crit = self.thresholds.get("low_critical")
        lo_warn = self.thresholds.get("low_warn")

        # High-side critical
        if hi_crit is not None and value >= hi_crit:
            return 1.0
        # Low-side critical
        if lo_crit is not None and value <= lo_crit:
            return 1.0
        # High-side: between warn and critical → interpolate 0.7 → 1.0
        if hi_crit is not None and hi_warn is not None and value >= hi_warn:
            span = max(hi_crit - hi_warn, 1e-9)
            return 0.7 + 0.3 * (value - hi_warn) / span
        # Low-side: between low_crit and low_warn → interpolate 0.7 → 1.0
        if lo_crit is not None and lo_warn is not None and value <= lo_warn:
            span = max(lo_warn - lo_crit, 1e-9)
            return 0.7 + 0.3 * (lo_warn - value) / span
        # High-side: approaching warn from normal → interpolate 0.2 → 0.7
        if hi_warn is not None:
            # Use center of normal range as the 0.2 anchor
            normal_center = (self.thresholds.get("low_warn") or 0)
            normal_center = (normal_center + hi_warn) / 2.0
            span = max(hi_warn - normal_center, 1e-9)
            if value > normal_center:
                frac = min(1.0, (value - normal_center) / span)
                return 0.2 + 0.5 * frac
        # Low-side: approaching low_warn from normal → interpolate 0.2 → 0.7
        if lo_warn is not None:
            normal_center_lo = (self.thresholds.get("warn") or lo_warn) if self.thresholds.get("warn") else lo_warn
            normal_center_lo = (lo_warn + normal_center_lo) / 2.0
            span = max(normal_center_lo - lo_warn, 1e-9)
            if value < normal_center_lo:
                frac = min(1.0, (normal_center_lo - value) / span)
                return 0.2 + 0.5 * frac
        return 0.2

    def push(self, value: float) -> None:
        self.buffer.append(value)

    def should_transmit_delta(self, value: float, threshold_pct: float = 0.05) -> bool:
        """True if value changed by >= threshold_pct relative to last transmitted."""
        if self.last_transmitted_value is None:
            return True
        baseline = max(1e-9, abs(self.last_transmitted_value))
        return abs(value - self.last_transmitted_value) / baseline >= threshold_pct

    def get_semantic_summary(self) -> Dict[str, object]:
        """Compress buffer into a semantic summary dict; empty dict if buffer is empty."""
        vals = list(self.buffer)
        if not vals:
            return {}

        hi_crit = self.thresholds.get("critical")
        lo_crit = self.thresholds.get("low_critical")

        # Spike detection: any sample exceeds a critical threshold mid-window
        spike = False
        if hi_crit is not None and any(v >= hi_crit for v in vals):
            spike = True
        if lo_crit is not None and any(v <= lo_crit for v in vals):
            spike = True

        if spike:
            trend = "SPIKE"
        else:
            first, last = vals[0], vals[-1]
            if last > first * 1.03:
                trend = "RISING"
            elif last < first * 0.97:
                trend = "FALLING"
            else:
                trend = "STABLE"

        peak = max(vals, key=self.clinical_importance)
        peak_imp = self.clinical_importance(peak)
        return {
            "semantic":        True,
            "min":             min(vals),
            "max":             max(vals),
            "mean":            round(sum(vals) / len(vals), 2),
            "trend":           trend,
            "n_samples":       len(vals),
            "importance":      round(peak_imp, 3),
            "peak_importance": round(peak_imp, 3),
        }



# ---------------------------------------------------------------------------
# PhysiologicalModel — Ornstein-Uhlenbeck mean-reverting process
# ---------------------------------------------------------------------------

class PhysiologicalModel:
    """
    Replaces random.randint() with a mean-reverting (Ornstein-Uhlenbeck) process
    so vitals evolve smoothly and naturally instead of jumping each tick.

    Discrete OU update:
        X(t+dt) = X(t) + θ(μ - X(t))·dt + σ·√dt·N(0,1)

    θ  — reversion speed  (how fast value returns to mean after disturbance)
    σ  — diffusion noise  (moment-to-moment variation)
    μ  — target mean      (shifts to burst-range midpoint during clinical events)

    Burst events cause a gradual rise then natural return — not an instant jump.
    """

    # (theta, sigma) tuned per device type and sampling interval
    _OU: Dict[str, Tuple[float, float]] = {
        "ECG":           (0.30, 4.0),   # HR: ~3 s time-constant, realistic HRV noise
        "SpO2":          (0.50, 0.3),   # tightly regulated — small, slow variation
        "BloodPressure": (0.25, 2.5),   # moderate BP beat-to-beat variation
        "Temperature":   (0.02, 0.2),   # body temp changes over many minutes
        "Respiration":   (0.35, 0.8),   # moderate breath-to-breath variation
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
        # Start at midpoint of normal range
        self._x  = (self.normal_lo + self.normal_hi) / 2.0
        self._mu = self._x

    def set_burst(self, active: bool) -> None:
        """Shift the OU attractor; value converges naturally — no instant jump."""
        self._mu = (
            (self.burst_lo  + self.burst_hi)  / 2.0 if active
            else (self.normal_lo + self.normal_hi) / 2.0
        )

    def step(self) -> float:
        """Advance one dt and return the next vital value as a float."""
        dt = self.interval
        self._x += (
            self.theta * (self._mu - self._x) * dt
            + self.sigma * math.sqrt(dt) * random.gauss(0.0, 1.0)
        )
        return self._x


# ---------------------------------------------------------------------------
# ECGNeuroKitSource — realistic HRV-derived heart rate via neurokit2
# ---------------------------------------------------------------------------

class ECGNeuroKitSource:
    """
    Uses neurokit2 to simulate a full ECG waveform at 100 Hz, then extracts
    beat-to-beat R-R intervals and converts them to instantaneous heart rate
    (BPM).  This gives realistic HRV instead of OU noise alone.

    The OU model still modulates the *mean* HR so long-term drift (exercise,
    stress, clinical events) is preserved.  The ECG waveform adds short-term
    HRV on top.

    Falls back to a plain OU model if neurokit2 is unavailable.
    """

    _DURATION = 60   # seconds of ECG to pre-generate per buffer
    _SR       = 100  # sampling rate — must match ECG device interval (0.01 s)

    def __init__(self, base_hr: float = 72.0) -> None:
        self._ou = PhysiologicalModel("ECG", (60, 100), (110, 150), 1.0 / self._SR)
        self._ou._x  = base_hr
        self._ou._mu = base_hr
        self._hr_series: List[float] = []
        self._pos = 0
        self._last_burst: bool = False
        self._refill()

    def _refill(self) -> None:
        import neurokit2 as nk
        import numpy as np

        target_hr = max(40.0, min(220.0, self._ou._x))
        try:
            ecg = nk.ecg_simulate(
                duration      = self._DURATION,
                sampling_rate = self._SR,
                heart_rate    = target_hr,
                noise         = 0.05,
                random_state  = random.randint(0, 9999),
            )
            _, info = nk.ecg_peaks(np.array(ecg), sampling_rate=self._SR)
            r_peaks = info["ECG_R_Peaks"]

            if len(r_peaks) > 1:
                rr_sec    = np.diff(r_peaks).astype(float) / self._SR   # seconds between beats
                hr_beats  = 60.0 / rr_sec                               # instantaneous BPM per beat
                # Forward-fill so every sample has an HR value
                hr_per_sample = np.full(self._DURATION * self._SR, target_hr, dtype=float)
                for i, rp in enumerate(r_peaks[:-1]):
                    nxt = r_peaks[i + 1]
                    hr_per_sample[rp:nxt] = hr_beats[i]
                self._hr_series = list(hr_per_sample.astype(float))
            else:
                self._hr_series = [float(target_hr)] * (self._DURATION * self._SR)
        except Exception:
            # Graceful fallback: flat HR series at current OU value
            self._hr_series = [float(target_hr)] * (self._DURATION * self._SR)

        self._pos = 0

    def step(self, burst_active: bool) -> float:
        # If burst state changed, flush the pre-generated buffer so _refill() is
        # triggered immediately with the new target HR instead of waiting up to
        # _DURATION seconds for the old buffer to drain.
        if burst_active != self._last_burst:
            self._last_burst = burst_active
            self._pos = len(self._hr_series)   # force _refill on next line

        # Let OU advance the long-term mean HR (drives next buffer's target)
        self._ou.set_burst(burst_active)
        self._ou.step()

        if self._pos >= len(self._hr_series):
            self._refill()

        val = self._hr_series[self._pos]
        self._pos += 1
        return val


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_payload(
    device_id:   int,
    seq:         int,
    now:         float,
    device_type: str,
    value:       int,
    label:       str,
    mode:        str,
    sem_buffer:  SemanticBuffer,
) -> Optional[bytes]:
    """
    Build a UDP payload for the current transmission mode.
    Returns None if the packet should be suppressed (semantic filtering).
    Caller is responsible for SUMMARY time-gating; this function does not gate.
    """

    if mode == "RAW":
        sem_buffer.last_transmitted_value = value
        return f"{device_id},{seq},{now:.6f},{device_type},{value:.2f},{label}".ljust(64).encode("utf-8")

    if mode == "DELTA":
        if not sem_buffer.should_transmit_delta(value, threshold_pct=0.05):
            return None   # value unchanged — suppress
        sem_buffer.last_transmitted_value = value
        return f"{device_id},{seq},{now:.6f},{device_type},{value:.2f},{label}".ljust(64).encode("utf-8")

    if mode == "SUMMARY":
        # Buffer fullness is not strictly required, but emit nothing until
        # we have at least half a window for a meaningful summary.
        if len(sem_buffer.buffer) < max(1, sem_buffer.window // 2):
            return None
        summary = sem_buffer.get_semantic_summary()
        if not summary:
            return None
        packet = {
            "device_id":   device_id,
            "seq":         seq,
            "ts":          round(now, 6),
            "device_type": device_type,
            "label":       label,
            "value":       value,
            **summary,
        }
        return json.dumps(packet, separators=(",", ":")).encode("utf-8")

    if mode == "CRITICAL_ONLY":
        if sem_buffer.clinical_importance(value) >= 0.7:
            sem_buffer.last_transmitted_value = value
            raw = f"{device_id},{seq},{now:.6f},{device_type},{value},{label},SEMANTIC_FILTERED"
            return raw.ljust(96).encode("utf-8")
        return None   # clinically unimportant — suppress

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive closed-loop vital-sign sender")
    parser.add_argument("--device-id",           type=int, required=True)
    parser.add_argument("--device-type",          type=str, default="ECG", choices=DEVICE_PROFILES.keys())
    parser.add_argument("--receiver-ip",          type=str, default="10.0.0.2")
    parser.add_argument("--receiver-port",        type=int, default=9000)
    parser.add_argument("--base-dir",             type=str, default=default_base_dir())
    parser.add_argument("--control-ip",           type=str, default="0.0.0.0")
    parser.add_argument("--control-base-port",    type=int, default=6000)
    parser.add_argument("--ward-ack-ip",          type=str, default=os.getenv("WARD_CONTROLLER_IP", "10.0.0.1"))
    parser.add_argument("--ward-ack-port",        type=int, default=int(os.getenv("WARD_CONTROLLER_PORT", "5006")))
    parser.add_argument("--correlated-burst-file", type=str, default="",
                        help="Path to a sentinel file; if it exists and is <5s old, "
                             "force a correlated burst regardless of per-device probability. "
                             "Touch this file from a stress script to simulate multi-organ events.")
    parser.add_argument("--initial-command",      type=str, default="FULL_ECG",
                        choices=list(COMMAND_SEMANTIC_MAP.keys()),
                        help="Starting transmission command before first ward_controller message")
    args = parser.parse_args()

    profile  = DEVICE_PROFILES[args.device_type]
    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    # --- Physiological signal source (smooth OU or neurokit2 ECG) ---
    if args.device_type == "ECG" and _NK2_AVAILABLE:
        _ecg_src = ECGNeuroKitSource(base_hr=80.0)
        def _next_value(burst_active: bool) -> int:
            return _ecg_src.step(burst_active)
        print(f"[Device {args.device_id}|ECG] Using neurokit2 HRV-derived heart rate.")
    else:
        _physio = PhysiologicalModel(
            args.device_type,
            profile["normal"],
            profile["burst"],
            profile["interval"],
        )
        def _next_value(burst_active: bool) -> int:
            _physio.set_burst(burst_active)
            return _physio.step()
        if args.device_type == "ECG":
            print(f"[Device {args.device_id}|ECG] neurokit2 not found; using OU process.")
        else:
            print(f"[Device {args.device_id}|{args.device_type}] Using OU physiological model.")

    log_file   = os.path.join(base_dir, f"sender_log_dev{args.device_id}_{args.device_type}.csv")
    stats_file = os.path.join(base_dir, f"sender_semantic_stats_dev{args.device_id}_{args.device_type}.csv")

    # --- Sockets ---
    tx_sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver = (args.receiver_ip, args.receiver_port)
    ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ack_target = (args.ward_ack_ip, args.ward_ack_port)

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_port = args.control_base_port + args.device_id
    ctrl_sock.bind((args.control_ip, ctrl_port))
    ctrl_sock.setblocking(False)
    try:
        ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
    except OSError:
        pass

    # --- Open log files once (not per packet) ---
    log_f  = open(log_file,   "w", newline="", encoding="utf-8")
    stat_f = open(stats_file, "w", newline="", encoding="utf-8")

    log_writer  = csv.writer(log_f)
    stats_writer = csv.writer(stat_f)

    log_writer.writerow(["seq", "timestamp", "device_id", "device_type", "value", "unit", "label", "encoded"])
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

    # --- State ---
    current_command  = args.initial_command
    _init_mode, _init_mult = COMMAND_SEMANTIC_MAP.get(current_command, ("RAW", 1.0))
    if current_command == "DOWNSAMPLED_ECG" and args.device_type != "ECG":
        _init_mult = 1.0
    tx_mode          = _init_mode
    current_interval = max(MIN_INTERVAL, profile["interval"] * _init_mult)
    summary_interval  = profile["summary_interval"]   # time-gate for SUMMARY mode
    last_summary_emit = 0.0
    last_command_time = time.time()
    command_timeout_s = 30.0

    sem_buffer = SemanticBuffer(
        args.device_type,
        window=SEMANTIC_BUFFER_WINDOWS.get(args.device_type, 30),
    )

    # Semantic encoder (ML-based codec; None when _SEMANTIC_AVAILABLE is False or models absent)
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(_repo_root, "models", "semantic")
    sem_encoder = SemanticEncoder(args.device_type, models_dir) if _SEMANTIC_AVAILABLE else None

    burst_mode     = False
    burst_end_time = 0.0
    sample_seq     = 0
    tx_seq         = 0

    # Counters: only incremented when we actually attempt build_payload
    mode_counters: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"sent": 0, "suppressed": 0}
    )

    last_log_flush   = time.time()
    last_stats_flush = time.time()

    # --- Stats flush helper ---
    def flush_semantic_stats(ts_now: float) -> None:
        total_sent       = sum(c["sent"]       for c in mode_counters.values())
        total_suppressed = sum(c["suppressed"] for c in mode_counters.values())

        def g(m: str, k: str) -> int:
            return int(mode_counters.get(m, {}).get(k, 0))

        stats_writer.writerow([
            round(ts_now, 6),
            args.device_id, args.device_type,
            current_command, tx_mode,
            sample_seq, total_sent, total_suppressed,
            g("RAW",           "sent"), g("RAW",           "suppressed"),
            g("DELTA",         "sent"), g("DELTA",         "suppressed"),
            g("SUMMARY",       "sent"), g("SUMMARY",       "suppressed"),
            g("CRITICAL_ONLY", "sent"), g("CRITICAL_ONLY", "suppressed"),
        ])
        stat_f.flush()

    print(
        f"[Device {args.device_id}|{args.device_type}] Sender started -> {receiver}, "
        f"control_port={ctrl_port}, base_interval={profile['interval']}s"
    )

    try:
        while True:
            now = time.time()

            # --- Drain control socket ---
            try:
                while True:
                    data, _ = ctrl_sock.recvfrom(2048)
                    msg = json.loads(data.decode("utf-8", errors="replace"))
                    if not isinstance(msg, dict):
                        continue
                    last_command_time = now
                    new_command = str(msg.get("command") or current_command)

                    # ACK command receptions so ward_controller can retry only when needed.
                    cmd_id = msg.get("command_id")
                    if cmd_id is not None:
                        try:
                            ack_packet = {
                                "type": "command_ack",
                                "device_id": args.device_id,
                                "command_id": int(cmd_id),
                                "command": new_command,
                                "timestamp": round(now, 6),
                            }
                            ack_sock.sendto(json.dumps(ack_packet).encode("utf-8"), ack_target)
                        except Exception:
                            pass

                    if new_command != current_command:
                        current_command = new_command
                        tx_mode, multiplier = COMMAND_SEMANTIC_MAP.get(current_command, ("RAW", 1.0))

                        # DOWNSAMPLED_ECG is ECG-specific; hold rate for all other devices
                        if current_command == "DOWNSAMPLED_ECG" and args.device_type != "ECG":
                            multiplier = 1.0

                        current_interval = max(MIN_INTERVAL, profile["interval"] * multiplier)
                        print(
                            f"[MODE CHANGE] dev={args.device_id} type={args.device_type} "
                            f"cmd={current_command} mode={tx_mode} interval={current_interval:.4f}s"
                        )
            except BlockingIOError:
                pass
            except Exception:
                pass

            # Safety fallback: if no control command arrives for a while, return to FULL_ECG.
            if now - last_command_time > command_timeout_s and current_command != "FULL_ECG":
                current_command = "FULL_ECG"
                tx_mode, multiplier = COMMAND_SEMANTIC_MAP.get(current_command, ("RAW", 1.0))
                current_interval = max(MIN_INTERVAL, profile["interval"] * multiplier)
                last_command_time = now
                print(
                    f"[WATCHDOG] dev={args.device_id} no command for {command_timeout_s:.0f}s; "
                    f"reverting to {current_command}"
                )

            # --- Burst logic: check end BEFORE assigning value ---
            if burst_mode and now > burst_end_time:
                burst_mode = False

            # Correlated burst: if the sentinel file exists and is fresh (<5s),
            # force burst on this device regardless of its individual probability.
            _corr_file = getattr(args, "correlated_burst_file", "")
            if not burst_mode and _corr_file:
                try:
                    _mtime = os.path.getmtime(_corr_file)
                    if now - _mtime < 5.0:
                        burst_mode     = True
                        burst_end_time = now + profile["burst_dur"]
                except OSError:
                    pass  # file not present — no correlated burst

            if not burst_mode and random.random() < profile["burst_prob"]:
                burst_mode     = True
                burst_end_time = now + profile["burst_dur"]

            # --- Physiological value — smooth OU / neurokit2 (not random.randint) ---
            # Burst flag is passed in so the model's OU attractor shifts
            # gradually toward the burst range rather than jumping instantly.
            value = _next_value(burst_mode)

            # --- Label via clinical importance (replaces fragile high * 0.95) ---
            if sem_encoder is not None:
                importance = sem_encoder.clinical_importance_learned(value)
            else:
                importance = sem_buffer.clinical_importance(value)
            if burst_mode:
                label = "CRITICAL"
            elif importance >= 0.7:
                label = "ALERT"
            else:
                label = "NORMAL"

            sample_seq += 1
            sem_buffer.push(value)

            # --- Determine whether to attempt transmission this tick ---
            # SUMMARY is time-gated (not sample-gated) to avoid broken rates
            # for slow devices like Temperature.
            if tx_mode == "SUMMARY":
                should_attempt      = (now - last_summary_emit) >= summary_interval
                is_intentional_skip = not should_attempt
            else:
                should_attempt      = True
                is_intentional_skip = False

            # --- Semantic encoder fast-path (ECG + BloodPressure only) ---
            # push every tick so the buffer fills up regardless of transmission gating
            used_sem_encoder = False
            if sem_encoder is not None and args.device_type in SEMANTIC_CAPABLE_TYPES:
                sem_encoder.push(float(value))
                if should_attempt and current_command.startswith("SEMANTIC_") and sem_encoder.ready:
                    # SEMANTIC_CRITICAL mirrors the CRITICAL_ONLY raw logic:
                    # only transmit (even as a compressed z) when the reading is
                    # clinically significant.  Sending z for a perfectly normal
                    # patient under a CRITICAL-severity command wastes bandwidth
                    # and contradicts the suppression intent of the mode.
                    _importance = sem_buffer.clinical_importance(value)
                    _suppress_critical = (
                        current_command == "SEMANTIC_CRITICAL" and _importance < 0.7
                    )
                    if _suppress_critical:
                        mode_counters[tx_mode]["suppressed"] += 1
                        used_sem_encoder = True   # block legacy path from double-counting
                    else:
                        z = sem_encoder.encode()
                        if z is not None:
                            sem_payload = encode_payload(
                                device_id   = args.device_id,
                                seq         = tx_seq + 1,
                                ts          = now,
                                device_type = args.device_type,
                                z           = z,
                                command     = current_command,
                                label       = label,
                            )
                            if sem_payload is not None:
                                try:
                                    tx_sock.sendto(sem_payload, receiver)
                                    tx_seq += 1
                                    mode_counters[tx_mode]["sent"] += 1
                                    if tx_mode == "SUMMARY":
                                        last_summary_emit = now
                                except Exception:
                                    pass
                                used_sem_encoder = True

            # --- Build and send payload (legacy path when sem encoder not used) ---
            payload: Optional[bytes] = None
            if not used_sem_encoder and should_attempt:
                next_tx_seq = tx_seq + 1
                payload = build_payload(
                    device_id   = args.device_id,
                    seq         = next_tx_seq,
                    now         = now,
                    device_type = args.device_type,
                    value       = value,
                    label       = label,
                    mode        = tx_mode,
                    sem_buffer  = sem_buffer,
                )
                # Tag raw payloads from semantic-capable types so the receiver
                # can skip decode_payload() during the codec buffer fill window
                # (first ~2s).  Without this marker the receiver attempts JSON
                # decode on CSV bytes and logs spurious warnings.
                if (payload is not None
                        and args.device_type in SEMANTIC_CAPABLE_TYPES
                        and sem_encoder is not None
                        and not sem_encoder.ready):
                    payload = b"RAW_LEGACY:" + payload

            if payload is not None:
                try:
                    tx_sock.sendto(payload, receiver)
                    tx_seq += 1
                    mode_counters[tx_mode]["sent"] += 1
                    if tx_mode == "SUMMARY":
                        last_summary_emit = now
                except Exception:
                    pass
            elif not is_intentional_skip and not used_sem_encoder:
                # Only count as semantically suppressed when we tried and were filtered.
                # Intentional SUMMARY skips are not counted — they are by design.
                mode_counters[tx_mode]["suppressed"] += 1

            # --- Log every sample (file stays open; flushed on interval) ---
            log_writer.writerow([
                sample_seq, now,
                args.device_id, args.device_type,
                value, profile["unit"], label,
                used_sem_encoder,
            ])

            # --- Periodic flushes ---
            if now - last_log_flush >= LOG_FLUSH_INTERVAL:
                log_f.flush()
                last_log_flush = now

            if now - last_stats_flush >= STATS_FLUSH_INTERVAL:
                flush_semantic_stats(now)
                last_stats_flush = now

            time.sleep(current_interval)

    except KeyboardInterrupt:
        pass
    finally:
        # Final flush and close on any exit path
        flush_semantic_stats(time.time())
        log_f.flush()
        log_f.close()
        stat_f.close()
        tx_sock.close()
        ack_sock.close()
        ctrl_sock.close()


if __name__ == "__main__":
    main()
