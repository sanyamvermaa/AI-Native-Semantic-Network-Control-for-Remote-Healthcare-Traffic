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

def update_recent_hr(ts, hr_value):
    recent_hr.append((ts, hr_value))
    cutoff = ts - 10.0
    while recent_hr and recent_hr[0][0] < cutoff:
        recent_hr.popleft()

def get_current_mode():
    return current_mode, dict(last_command_meta)

while True:
    now = time.time()

    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    # Pull latest fleet mode from centralized ward controller state.
    refresh_mode_from_ward_controller(now)

    # Randomly trigger a clinical burst/alarm episode
    if not burst_mode and random.random() < profile["burst_prob"]:
        burst_mode     = True
        burst_end_time = now + profile["burst_dur"]

    if burst_mode:
        value = random.randint(*profile["burst"])
        if now > burst_end_time:
            burst_mode = False
    else:
        value = random.randint(*profile["normal"])

    # Clinical label (patient health, not network health)
    low, high = profile["normal"]
    if burst_mode:
        label = "CRITICAL"
    elif value > high * 0.95:
        label = "ALERT"
    else:
        label = "NORMAL"

    seq_num += 1

    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    hr_value = int(value)
    update_recent_hr(now, hr_value)

    mode, _ = get_current_mode()

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
            max(0, hr_value + random.randint(-3, 3)) for _ in range(10)
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
            max(0, hr_value + random.randint(-3, 3)) for _ in range(10)
        ]

    payload = json.dumps(packet, separators=(",", ":"))
    payload = payload.ljust(64)
    sock.sendto(payload.encode("utf-8"), server)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seq_num, now, args.device_id, args.device_type, value, profile["unit"], label])

    send_interval = MODE_INTERVALS.get(mode, profile["interval"])
    time.sleep(send_interval)
