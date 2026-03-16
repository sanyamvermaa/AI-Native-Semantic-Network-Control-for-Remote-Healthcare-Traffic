"""
health_sender.py — Simulates a single wireless patient monitoring device.
Launched multiple times by dynamic_traffic_generator.py, once per device.

Usage:
    python health_sender.py --device-id <id> --device-type <type>

Device Types: ECG | SpO2 | BloodPressure | Temperature | Respiration
"""
import socket
import time
import random
import csv
import os
import argparse
import json
import math
from collections import deque

# ── Device Profiles ───────────────────────────────────────────────────────────
# Each device type has its own send rate and realistic vital-sign ranges.
# "burst" represents a clinically abnormal episode (alarm condition).
DEVICE_PROFILES = {
    "ECG": {
        "interval":    0.010,           # 100 pps  — high-freq waveform data
        "normal":      (60,  100),       # BPM
        "burst":       (110, 150),
        "unit":        "bpm",
        "burst_prob":  0.002,
        "burst_dur":   6,
    },
    "SpO2": {
        "interval":    0.020,           # 50 pps
        "normal":      (95,  100),       # % saturation
        "burst":       (80,   94),
        "unit":        "%",
        "burst_prob":  0.002,
        "burst_dur":   5,
    },
    "BloodPressure": {
        "interval":    0.050,           # 20 pps  — slower update rate
        "normal":      (110, 130),       # systolic mmHg
        "burst":       (160, 200),
        "unit":        "mmHg",
        "burst_prob":  0.001,
        "burst_dur":   8,
    },
    "Temperature": {
        "interval":    0.100,           # 10 pps
        "normal":      (360,  373),      # °C × 10  (36.0 – 37.3)
        "burst":       (380,  402),
        "unit":        "°Cx10",
        "burst_prob":  0.001,
        "burst_dur":   10,
    },
    "Respiration": {
        "interval":    0.020,           # 50 pps
        "normal":      (12,   20),       # breaths/min
        "burst":       (25,   40),
        "unit":        "br/min",
        "burst_prob":  0.002,
        "burst_dur":   5,
    },
}

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device-id",   type=int, required=True,  help="Unique device number (0-N)")
parser.add_argument("--device-type", type=str, default="ECG",  choices=DEVICE_PROFILES.keys())
parser.add_argument("--receiver-ip", type=str, default="10.0.0.2")
parser.add_argument("--receiver-port",type=int, default=9000)
parser.add_argument("--base-dir",    type=str, default=os.environ.get("HEALTH_DATA_BASE_DIR", "/home/ayhm23/health_data/csv"))
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────
profile   = DEVICE_PROFILES[args.device_type]
BASE_DIR  = args.base_dir
try:
    os.makedirs(BASE_DIR, exist_ok=True)
except Exception as exc:
    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    # Fallback for environments where /home/... is not writable.
    fallback_base = os.path.join(os.getcwd(), "csv")
    os.makedirs(fallback_base, exist_ok=True)
    print(f"[Sender][WARN] BASE_DIR '{BASE_DIR}' unavailable ({exc}). Using '{fallback_base}'.")
    BASE_DIR = fallback_base
LOG_FILE  = os.path.join(BASE_DIR, f"sender_log_dev{args.device_id}_{args.device_type}.csv")

sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server = (args.receiver_ip, args.receiver_port)

# ── CLOSED LOOP ADDITION ─────────────────────────────────────
current_mode = "FULL_ECG"
last_command_meta = {"network_state": "N/A", "health_state": "N/A", "timestamp": time.time()}
last_state_mtime = None
last_state_check_ts = 0.0
STATE_POLL_INTERVAL = 0.10
WARD_STATE_FILE = os.path.join(BASE_DIR, "ward_mode_state.json")

MODE_INTERVALS = {
    "FULL_ECG": 0.10,
    "FULL_ECG_PRIORITY": 0.05,
    "DOWNSAMPLED_ECG": 0.20,
    "SEMANTIC_ALERT": 0.50,
    "SEMANTIC_CRITICAL": 0.25,
    "SEMANTIC_SUMMARY": 1.00,
}

def refresh_mode_from_ward_controller(now_ts):
    global current_mode, last_command_meta, last_state_mtime, last_state_check_ts

    if (now_ts - last_state_check_ts) < STATE_POLL_INTERVAL:
        return

    last_state_check_ts = now_ts

    try:
        stat = os.stat(WARD_STATE_FILE)
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"[Sender][WARN] Could not stat ward state file: {exc}")
        return

    if last_state_mtime is not None and stat.st_mtime <= last_state_mtime:
        return

    try:
        with open(WARD_STATE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)

        new_mode = str(payload.get("command", "FULL_ECG"))
        net_state = str(payload.get("network_state", "N/A"))
        health_state = str(payload.get("health_state", "N/A"))
        cmd_time = float(payload.get("timestamp", time.time()))

        if new_mode not in MODE_INTERVALS:
            return

        old_mode = current_mode
        current_mode = new_mode
        last_command_meta = {
            "network_state": net_state,
            "health_state": health_state,
            "timestamp": cmd_time,
        }
        last_state_mtime = stat.st_mtime

        print(f"[CMD RECEIVED] {new_mode}  (net={net_state}, health={health_state})")
        if old_mode != current_mode:
            print(
                f"[MODE CHANGE] {old_mode} → {current_mode}  "
                f"(net={net_state} health={health_state} t={cmd_time:.2f})"
            )
    except Exception as exc:
        print(f"[Sender][WARN] Failed to read ward state: {exc}")

# Write log header
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seq", "timestamp", "device_id", "device_type", "value", "unit", "label"])

print(f"[Device {args.device_id} | {args.device_type}] Sender started → {server}")

# ── Main Loop ─────────────────────────────────────────────────────────────────
burst_mode     = False
burst_end_time = 0.0
seq_num        = 0

# ── CLOSED LOOP ADDITION ─────────────────────────────────────
recent_hr = deque()

VITAL_MODEL = {
    "ECG": {
        "baseline": 74.0,
        "hard_bounds": (45.0, 185.0),
        "roc_per_sec": 4.2,
        "noise_sigma": 0.20,
        "display": lambda x: int(round(x)),
    },
    "SpO2": {
        "baseline": 98.0,
        "hard_bounds": (82.0, 100.0),
        "roc_per_sec": 1.0,
        "noise_sigma": 0.05,
        "display": lambda x: int(round(x)),
    },
    "BloodPressure": {
        "baseline": 118.0,
        "hard_bounds": (85.0, 210.0),
        "roc_per_sec": 3.0,
        "noise_sigma": 0.25,
        "display": lambda x: int(round(x)),
    },
    "Temperature": {
        "baseline": 367.0,
        "hard_bounds": (350.0, 405.0),
        "roc_per_sec": 0.5,
        "noise_sigma": 0.04,
        "display": lambda x: int(round(x)),
    },
    "Respiration": {
        "baseline": 15.0,
        "hard_bounds": (8.0, 45.0),
        "roc_per_sec": 1.2,
        "noise_sigma": 0.06,
        "display": lambda x: int(round(x)),
    },
}


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _stress_from_state(mode, meta, now_ts):
    net_state = str(meta.get("network_state", "Stable"))
    health_state = str(meta.get("health_state", "NORMAL"))

    net_map = {"Stable": 0.20, "Unstable": 0.55, "Critical": 0.90}
    health_map = {"NORMAL": 0.0, "ALERT": 0.15, "CRITICAL": 0.30}
    mode_map = {
        "FULL_ECG": 0.00,
        "FULL_ECG_PRIORITY": 0.08,
        "DOWNSAMPLED_ECG": 0.10,
        "SEMANTIC_ALERT": 0.18,
        "SEMANTIC_CRITICAL": 0.25,
        "SEMANTIC_SUMMARY": 0.12,
    }

    base = net_map.get(net_state, 0.35)
    health_boost = health_map.get(health_state.upper(), 0.10)
    mode_boost = mode_map.get(mode, 0.0)
    drift = 0.08 * math.sin(now_ts / 28.0)

    return _clamp(base + health_boost + mode_boost + drift, 0.0, 1.20)


def _target_from_stress(device_type, stress, now_ts, device_bias):
    hr_proxy = 72.0 + 42.0 * stress + 2.2 * math.sin(now_ts / 9.0 + 0.7)
    rr_proxy = 14.0 + 10.0 * stress + 1.2 * math.sin(now_ts / 11.0 + 0.4)

    if device_type == "ECG":
        return hr_proxy + device_bias
    if device_type == "SpO2":
        return 98.5 - 4.6 * stress - 0.22 * max(0.0, rr_proxy - 20.0) + 0.15 * device_bias
    if device_type == "BloodPressure":
        return 116.0 + 45.0 * stress + 0.20 * (hr_proxy - 70.0) + device_bias
    if device_type == "Temperature":
        return 366.5 + 8.0 * stress + 0.03 * (hr_proxy - 70.0) + 0.3 * device_bias
    if device_type == "Respiration":
        return rr_proxy + 0.035 * (hr_proxy - 70.0) + 0.2 * device_bias
    return hr_proxy


def _label_from_value(device_type, value):
    if device_type == "ECG":
        if value >= 135 or value <= 48:
            return "CRITICAL"
        if value >= 110 or value <= 56:
            return "ALERT"
        return "NORMAL"
    if device_type == "SpO2":
        if value <= 89:
            return "CRITICAL"
        if value <= 94:
            return "ALERT"
        return "NORMAL"
    if device_type == "BloodPressure":
        if value >= 185 or value <= 88:
            return "CRITICAL"
        if value >= 150 or value <= 100:
            return "ALERT"
        return "NORMAL"
    if device_type == "Temperature":
        if value >= 390 or value <= 355:
            return "CRITICAL"
        if value >= 378 or value <= 360:
            return "ALERT"
        return "NORMAL"
    if device_type == "Respiration":
        if value >= 33 or value <= 9:
            return "CRITICAL"
        if value >= 24 or value <= 11:
            return "ALERT"
        return "NORMAL"
    return "NORMAL"

def update_recent_hr(ts, hr_value):
    recent_hr.append((ts, hr_value))
    cutoff = ts - 10.0
    while recent_hr and recent_hr[0][0] < cutoff:
        recent_hr.popleft()

def get_current_mode():
    return current_mode, dict(last_command_meta)


model_cfg = VITAL_MODEL[args.device_type]
device_rng = random.Random(1009 + args.device_id * 37)
device_bias = device_rng.uniform(-1.2, 1.2)
phys_value = model_cfg["baseline"] + device_bias
last_loop_ts = time.time()
burst_strength = 0.0

while True:
    now = time.time()

    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    # Pull latest fleet mode from centralized ward controller state.
    refresh_mode_from_ward_controller(now)

    mode, mode_meta = get_current_mode()

    dt = _clamp(now - last_loop_ts, 0.01, 1.5)
    last_loop_ts = now

    stress = _stress_from_state(mode, mode_meta, now)

    # Bursts now raise stress smoothly instead of forcing abrupt random spikes.
    burst_trigger_prob = profile["burst_prob"] * (1.0 + 0.8 * stress)
    if not burst_mode and random.random() < burst_trigger_prob:
        burst_mode = True
        burst_end_time = now + profile["burst_dur"]
        burst_strength = _clamp(0.25 + 0.35 * stress, 0.20, 0.65)

    if burst_mode and now >= burst_end_time:
        burst_mode = False

    if burst_mode:
        stress = _clamp(stress + burst_strength, 0.0, 1.30)

    target = _target_from_stress(args.device_type, stress, now, device_bias)

    lo, hi = model_cfg["hard_bounds"]
    max_step = model_cfg["roc_per_sec"] * dt
    delta = _clamp(target - phys_value, -max_step, max_step)
    noise = random.gauss(0.0, model_cfg["noise_sigma"] * math.sqrt(dt))
    phys_value = _clamp(phys_value + delta + noise, lo, hi)

    value = model_cfg["display"](phys_value)
    label = _label_from_value(args.device_type, value)

    seq_num += 1

    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    # Keep backward-compatible key names expected by downstream receiver logic.
    hr_value = int(value)
    update_recent_hr(now, hr_value)

    packet = {
        "device_id": args.device_id,
        "device_type": args.device_type,
        "seq": seq_num,
        "timestamp": round(now, 6),
        "health_label": label,
        "mode": mode,
    }

    if mode in ("FULL_ECG", "FULL_ECG_PRIORITY"):
        packet["heart_rate"] = hr_value
        packet["raw_ecg_snippet"] = [
            max(0, int(round(hr_value + 2.2 * math.sin((now * 9.0) + i * 0.45) + random.gauss(0, 0.6))))
            for i in range(10)
        ]
    elif mode == "DOWNSAMPLED_ECG":
        packet["heart_rate"] = hr_value
    elif mode == "SEMANTIC_ALERT":
        packet["alert_message"] = f"ALERT: HR={hr_value} STATUS={label}"
    elif mode == "SEMANTIC_CRITICAL":
        packet["health_label"] = "CRITICAL"
        packet["alert_message"] = f"CRITICAL: HR={hr_value} IMMEDIATE ATTENTION"
    elif mode == "SEMANTIC_SUMMARY":
        hr_samples = [v for _, v in recent_hr]
        if hr_samples:
            hr_avg = int(sum(hr_samples) / len(hr_samples))
            hr_min = int(min(hr_samples))
            hr_max = int(max(hr_samples))
        else:
            hr_avg = hr_min = hr_max = hr_value
        packet["summary"] = f"HR_AVG={hr_avg} HR_MIN={hr_min} HR_MAX={hr_max} over 10s"
    else:
        packet["heart_rate"] = hr_value
        mode = "FULL_ECG"
        packet["mode"] = mode
        packet["raw_ecg_snippet"] = [
            max(0, int(round(hr_value + 2.2 * math.sin((now * 9.0) + i * 0.45) + random.gauss(0, 0.6))))
            for i in range(10)
        ]

    payload = json.dumps(packet, separators=(",", ":"))
    payload = payload.ljust(64)
    sock.sendto(payload.encode("utf-8"), server)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seq_num, now, args.device_id, args.device_type, value, profile["unit"], label])

    send_interval = MODE_INTERVALS.get(mode, profile["interval"])
    time.sleep(send_interval)
