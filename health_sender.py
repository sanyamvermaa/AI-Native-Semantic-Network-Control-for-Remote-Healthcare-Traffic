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
parser.add_argument("--base-dir",    type=str, default="/home/ayhm23/health_data/csv")
args = parser.parse_args()

# ── Setup ─────────────────────────────────────────────────────────────────────
profile   = DEVICE_PROFILES[args.device_type]
BASE_DIR  = args.base_dir
os.makedirs(BASE_DIR, exist_ok=True)
LOG_FILE  = os.path.join(BASE_DIR, f"sender_log_dev{args.device_id}_{args.device_type}.csv")

sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server = (args.receiver_ip, args.receiver_port)

# Write log header
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seq", "timestamp", "device_id", "device_type", "value", "unit", "label"])

print(f"[Device {args.device_id} | {args.device_type}] Sender started → {server}")

# ── Main Loop ─────────────────────────────────────────────────────────────────
burst_mode     = False
burst_end_time = 0.0
seq_num        = 0

while True:
    now = time.time()

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

    # Message format: device_id,seq,timestamp,device_type,value,label
    # Padded to ~64 bytes to make throughput_bps a meaningful feature
    payload = f"{args.device_id},{seq_num},{now:.6f},{args.device_type},{value},{label}"
    payload = payload.ljust(64)           # pad to fixed size
    sock.sendto(payload.encode(), server)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seq_num, now, args.device_id, args.device_type, value, profile["unit"], label])

    time.sleep(profile["interval"])
