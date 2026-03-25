"""
Closed-loop receiver that computes per-window telemetry, infers network/health
states, and publishes updates to ward_controller over UDP.
"""

import csv
import json
import os
import select
import socket
import time
from collections import defaultdict

from common import choose_health_state, choose_network_state, default_base_dir

BASE_DIR = default_base_dir()
TELEMETRY_INTERVAL = 0.25
DRAIN_BUDGET = 0.15
STATUS_INTERVAL = 15.0

WARD_IP = os.getenv("WARD_CONTROLLER_IP", "10.0.0.1")
WARD_PORT = int(os.getenv("WARD_CONTROLLER_PORT", "5006"))

os.makedirs(BASE_DIR, exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(False)
sock.bind(("0.0.0.0", 9000))

ward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("[Receiver] Closed-loop gateway listening on :9000")
print(f"[Receiver] Ward controller target: {WARD_IP}:{WARD_PORT}")

device_expected_seq = defaultdict(lambda: 1)
device_prev_delay = defaultdict(lambda: None)
device_last_seen = {}

bytes_received = 0
bytes_attempted = 0
received_packets = 0
lost_packets = 0
jitter_sum = 0.0
jitter_count = 0
delay_sum = 0.0
delay_count = 0
active_devices_window = set()
label_counts = defaultdict(int)

telemetry_path = os.path.join(BASE_DIR, "network_telemetry.csv")
telemetry_file = open(telemetry_path, "w", newline="", encoding="utf-8")
telemetry_writer = csv.writer(telemetry_file)
telemetry_writer.writerow(
    [
        "timestamp",
        "bandwidth_usage_bps",
        "throughput_bps",
        "packet_loss_rate",
        "jitter",
        "avg_delay",
        "active_devices",
        "packets_per_window",
        "network_condition",
    ]
)
telemetry_file.flush()

telemetry_rows = 0


def flush_telemetry(now: float) -> None:
    global bytes_received, bytes_attempted
    global received_packets, lost_packets
    global jitter_sum, jitter_count
    global delay_sum, delay_count
    global active_devices_window
    global telemetry_rows
    global label_counts

    total_attempted = received_packets + lost_packets
    packet_loss_rate = (lost_packets / total_attempted) if total_attempted > 0 else 0.0
    avg_jitter = (jitter_sum / jitter_count) if jitter_count > 0 else 0.0
    avg_delay = (delay_sum / delay_count) if delay_count > 0 else 0.0

    throughput_bps = bytes_received / TELEMETRY_INTERVAL
    bandwidth_bps = bytes_attempted / TELEMETRY_INTERVAL

    network_state = choose_network_state(packet_loss_rate, avg_delay * 1000.0, avg_jitter * 1000.0)
    health_state = choose_health_state(label_counts)

    telemetry_writer.writerow(
        [
            round(now, 3),
            round(bandwidth_bps, 2),
            round(throughput_bps, 2),
            round(packet_loss_rate, 6),
            round(avg_jitter * 1000.0, 6),
            round(avg_delay * 1000.0, 6),
            len(active_devices_window),
            total_attempted,
            network_state,
        ]
    )
    telemetry_file.flush()

    payload = {
        "timestamp": now,
        "network_state": network_state,
        "health_state": health_state,
        "packet_loss_rate": packet_loss_rate,
        "avg_delay_ms": avg_delay * 1000.0,
        "jitter_ms": avg_jitter * 1000.0,
        "active_devices": len(active_devices_window),
        "packets_per_window": total_attempted,
    }

    try:
        ward_sock.sendto(json.dumps(payload).encode("utf-8"), (WARD_IP, WARD_PORT))
    except Exception:
        pass

    telemetry_rows += 1

    bytes_received = 0
    bytes_attempted = 0
    received_packets = 0
    lost_packets = 0
    jitter_sum = 0.0
    jitter_count = 0
    delay_sum = 0.0
    delay_count = 0
    active_devices_window = set()
    label_counts = defaultdict(int)


start_time = time.time()
next_flush_time = time.time() + TELEMETRY_INTERVAL
last_status_time = time.time()

try:
    while True:
        now = time.time()

        if now >= next_flush_time:
            flush_telemetry(now)
            next_flush_time += TELEMETRY_INTERVAL

        if now - last_status_time >= STATUS_INTERVAL:
            elapsed_sec = now - start_time
            elapsed_min = int(elapsed_sec // 60)
            elapsed_s = int(elapsed_sec % 60)
            print(f"[Receiver] Elapsed: {elapsed_min}m {elapsed_s}s | Rows: {telemetry_rows:5d} | Status: Running")
            last_status_time = now

        ready, _, _ = select.select([sock], [], [], 0.01)
        if not ready:
            continue

        drain_start = time.time()
        while time.time() - drain_start < DRAIN_BUDGET:
            try:
                data, _ = sock.recvfrom(1024)
            except BlockingIOError:
                break

            recv_time = time.time()
            try:
                parts = data.decode("utf-8", errors="replace").strip().split(",")
                device_id = int(parts[0])
                seq = int(parts[1])
                send_time = float(parts[2])
                device_type = parts[3]
                _value = parts[4]
                label = parts[5].strip().upper()
                _ = device_type
            except Exception:
                continue

            device_last_seen[device_id] = recv_time
            delay = recv_time - send_time
            delay_sum += delay
            delay_count += 1

            prev = device_prev_delay[device_id]
            if prev is not None:
                jitter_sum += abs(delay - prev)
                jitter_count += 1
            device_prev_delay[device_id] = delay

            expected = device_expected_seq[device_id]
            if seq > expected:
                gap = seq - expected
                lost_packets += gap
                bytes_attempted += gap * len(data)
            device_expected_seq[device_id] = seq + 1

            received_packets += 1
            active_devices_window.add(device_id)

            packet_size = len(data)
            bytes_received += packet_size
            bytes_attempted += packet_size

            if label in ("NORMAL", "ALERT", "CRITICAL"):
                label_counts[label] += 1

except KeyboardInterrupt:
    print("\n[Receiver] Flushing final window...")
    flush_telemetry(time.time())
finally:
    telemetry_file.flush()
    telemetry_file.close()
    sock.close()
    ward_sock.close()

    total_elapsed = time.time() - start_time
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print(
        f"[Receiver] Shutdown. Total runtime: {total_min}m {total_sec}s | "
        f"Telemetry rows: {telemetry_rows}"
    )
