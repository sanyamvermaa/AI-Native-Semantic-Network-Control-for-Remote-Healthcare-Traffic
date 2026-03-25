"""
Adaptive closed-loop sender.

Each sender transmits vital signs to the receiver and listens for control
commands from ward_controller to adapt its transmission mode in real time.
"""

import argparse
import csv
import json
import os
import random
import socket
import time

from common import default_base_dir

DEVICE_PROFILES = {
    "ECG": {
        "interval": 0.010,
        "normal": (60, 100),
        "burst": (110, 150),
        "unit": "bpm",
        "burst_prob": 0.002,
        "burst_dur": 6,
    },
    "SpO2": {
        "interval": 0.020,
        "normal": (95, 100),
        "burst": (80, 94),
        "unit": "%",
        "burst_prob": 0.002,
        "burst_dur": 5,
    },
    "BloodPressure": {
        "interval": 0.050,
        "normal": (110, 130),
        "burst": (160, 200),
        "unit": "mmHg",
        "burst_prob": 0.001,
        "burst_dur": 8,
    },
    "Temperature": {
        "interval": 0.100,
        "normal": (360, 373),
        "burst": (380, 402),
        "unit": "Cx10",
        "burst_prob": 0.001,
        "burst_dur": 10,
    },
    "Respiration": {
        "interval": 0.020,
        "normal": (12, 20),
        "burst": (25, 40),
        "unit": "br/min",
        "burst_prob": 0.002,
        "burst_dur": 5,
    },
}


def command_to_interval(base_interval: float, command: str, device_type: str) -> float:
    c = (command or "").upper()
    if c == "FULL_ECG" or c == "FULL_ECG_PRIORITY":
        return base_interval
    if c == "DOWNSAMPLED_ECG":
        if device_type == "ECG":
            return base_interval * 3.5
        return base_interval * 1.3
    if c == "SEMANTIC_ALERT":
        if device_type == "ECG":
            return base_interval * 2.0
        return base_interval * 1.2
    if c == "SEMANTIC_CRITICAL":
        if device_type in ("ECG", "SpO2"):
            return base_interval * 0.8
        return base_interval * 1.1
    if c == "SEMANTIC_SUMMARY":
        return base_interval * 4.0
    return base_interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--device-type", type=str, default="ECG", choices=DEVICE_PROFILES.keys())
    parser.add_argument("--receiver-ip", type=str, default="10.0.0.2")
    parser.add_argument("--receiver-port", type=int, default=9000)
    parser.add_argument("--base-dir", type=str, default=default_base_dir())
    parser.add_argument("--control-ip", type=str, default="0.0.0.0")
    parser.add_argument("--control-base-port", type=int, default=6000)
    args = parser.parse_args()

    profile = DEVICE_PROFILES[args.device_type]
    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    log_file = os.path.join(base_dir, f"sender_log_dev{args.device_id}_{args.device_type}.csv")

    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver = (args.receiver_ip, args.receiver_port)

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_port = args.control_base_port + args.device_id
    ctrl_sock.bind((args.control_ip, ctrl_port))
    ctrl_sock.setblocking(False)

    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "timestamp", "device_id", "device_type", "value", "unit", "label"])

    current_command = "FULL_ECG"
    current_interval = profile["interval"]
    print(
        f"[Device {args.device_id}|{args.device_type}] Sender started -> {receiver}, "
        f"control_port={ctrl_port}"
    )

    burst_mode = False
    burst_end_time = 0.0
    seq_num = 0

    while True:
        now = time.time()

        try:
            while True:
                data, _ = ctrl_sock.recvfrom(2048)
                msg = json.loads(data.decode("utf-8", errors="replace"))
                if not isinstance(msg, dict):
                    continue
                new_command = str(msg.get("command") or current_command)
                if new_command != current_command:
                    current_command = new_command
                    current_interval = max(0.002, command_to_interval(profile["interval"], current_command, args.device_type))
                    print(
                        f"[MODE CHANGE] dev={args.device_id} type={args.device_type} "
                        f"command={current_command} interval={current_interval:.4f}s"
                    )
        except BlockingIOError:
            pass
        except Exception:
            pass

        if (not burst_mode) and random.random() < profile["burst_prob"]:
            burst_mode = True
            burst_end_time = now + profile["burst_dur"]

        if burst_mode:
            value = random.randint(*profile["burst"])
            if now > burst_end_time:
                burst_mode = False
        else:
            value = random.randint(*profile["normal"])

        low, high = profile["normal"]
        if burst_mode:
            label = "CRITICAL"
        elif value > high * 0.95:
            label = "ALERT"
        else:
            label = "NORMAL"

        seq_num += 1
        payload = f"{args.device_id},{seq_num},{now:.6f},{args.device_type},{value},{label}".ljust(64)

        try:
            tx_sock.sendto(payload.encode("utf-8"), receiver)
        except Exception:
            pass

        try:
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([seq_num, now, args.device_id, args.device_type, value, profile["unit"], label])
        except Exception:
            pass

        time.sleep(current_interval)


if __name__ == "__main__":
    main()
