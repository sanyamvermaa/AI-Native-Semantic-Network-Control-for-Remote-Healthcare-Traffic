"""
health_receiver.py — Central receiver / hospital gateway

Fixes telemetry starvation caused by heavy UDP packet drain.

Key design:
1. Telemetry flush runs on a strict schedule (next_flush_time).
2. Packet drain has a strict time budget (DRAIN_BUDGET).
3. select() timeout keeps CPU usage low.
"""

import socket
import time
import csv
import os
import select
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────
BASE_DIR           = "/home/ayhm23/health_data/csv"
TELEMETRY_INTERVAL = 0.25        # flush every 0.25s
DRAIN_BUDGET       = 0.15        # max time spent draining packets
STATUS_INTERVAL    = 15.0        # print progress every 30 seconds
DEBUG_LOG          = False

os.makedirs(BASE_DIR, exist_ok=True)

# ── Socket Setup ───────────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(False)
sock.bind(("0.0.0.0", 9000))

print("[Receiver] Healthcare gateway listening on :9000 ...")

# ── Per-device tracking ─────────────────────────────────────────────
device_expected_seq = defaultdict(lambda: 1)
device_prev_delay   = defaultdict(lambda: None)
device_last_seen    = {}

# ── Window accumulators ─────────────────────────────────────────────
bytes_received        = 0
bytes_attempted       = 0
received_packets      = 0
lost_packets          = 0
jitter_sum            = 0.0
jitter_count          = 0
delay_sum             = 0.0
delay_count           = 0
queue_length          = 0
active_devices_window = set()

# ── Telemetry output file ───────────────────────────────────────────
telemetry_path = os.path.join(BASE_DIR, "network_telemetry.csv")

telemetry_file = open(telemetry_path, "w", newline="")
telemetry_writer = csv.writer(telemetry_file)

telemetry_writer.writerow([
    "timestamp",
    "active_devices",
    "bandwidth_usage_bps",
    "throughput_bps",
    "packet_loss_rate",
    "jitter",
    "avg_delay",
    "queue_length",
    "packets_per_window",
])

telemetry_file.flush()

# ── Optional packet log ─────────────────────────────────────────────
if DEBUG_LOG:
    receiver_log_path = os.path.join(BASE_DIR, "receiver_log.csv")
    receiver_log_file = open(receiver_log_path, "w", newline="")
    receiver_log_writer = csv.writer(receiver_log_file)

    receiver_log_writer.writerow([
        "seq","device_id","device_type",
        "send_time","recv_time","value","label","delay"
    ])

# ── Telemetry flush ─────────────────────────────────────────────────
telemetry_rows = 0

def flush_telemetry(now):
    global bytes_received, bytes_attempted
    global received_packets, lost_packets
    global jitter_sum, jitter_count
    global delay_sum, delay_count
    global active_devices_window, queue_length
    global telemetry_rows

    total_attempted = received_packets + lost_packets

    packet_loss_rate = (
        lost_packets / total_attempted if total_attempted > 0 else 0.0
    )

    avg_jitter = (
        jitter_sum / jitter_count if jitter_count > 0 else 0.0
    )

    avg_delay = (
        delay_sum / delay_count if delay_count > 0 else 0.0
    )

    throughput_bps = bytes_received / TELEMETRY_INTERVAL
    bandwidth_bps  = bytes_attempted / TELEMETRY_INTERVAL

    queue_length = lost_packets

    telemetry_writer.writerow([
        round(now,3),
        len(active_devices_window),
        round(bandwidth_bps,2),
        round(throughput_bps,2),
        round(packet_loss_rate,6),
        round(avg_jitter,6),
        round(avg_delay,6),
        queue_length,
        total_attempted
    ])

    telemetry_file.flush()
    telemetry_rows += 1

    # reset window
    bytes_received        = 0
    bytes_attempted       = 0
    received_packets      = 0
    lost_packets          = 0
    jitter_sum            = 0.0
    jitter_count          = 0
    delay_sum             = 0.0
    delay_count           = 0
    queue_length          = 0
    active_devices_window = set()

# ── Main loop timers ────────────────────────────────────────────────
start_time = time.time()
next_flush_time = time.time() + TELEMETRY_INTERVAL
last_status_time = time.time()

packets_since_status = 0

try:
    while True:

        now = time.time()

        # ── 1. Telemetry flush ──────────────────────────────────────
        if now >= next_flush_time:
            flush_telemetry(now)
            next_flush_time += TELEMETRY_INTERVAL

        # ── 2. Status print (elapsed time) ──────────────────────────
        if now - last_status_time >= STATUS_INTERVAL:

            elapsed_sec = now - start_time
            elapsed_min = int(elapsed_sec // 60)
            elapsed_s = int(elapsed_sec % 60)

            packets_since_status = 0
            last_status_time = now

            print(f"[Receiver] Elapsed: {elapsed_min}m {elapsed_s}s | Rows: {telemetry_rows:5d} | Status: Running")

        # ── 3. Wait briefly for packets ─────────────────────────────
        ready,_,_ = select.select([sock],[],[],0.01)

        if not ready:
            continue

        # ── 4. Drain socket with time budget ────────────────────────
        drain_start = time.time()

        while time.time() - drain_start < DRAIN_BUDGET:

            try:
                data,addr = sock.recvfrom(1024)
            except BlockingIOError:
                break

            recv_time = time.time()

            packets_since_status += 1

            try:
                parts       = data.decode().strip().split(",")
                device_id   = int(parts[0])
                seq         = int(parts[1])
                send_time   = float(parts[2])
                device_type = parts[3]
                value       = parts[4]
                label       = parts[5]
            except:
                continue

            device_last_seen[device_id] = recv_time

            delay = recv_time - send_time

            delay_sum   += delay
            delay_count += 1

            # jitter
            prev = device_prev_delay[device_id]

            if prev is not None:
                jitter_sum += abs(delay - prev)
                jitter_count += 1

            device_prev_delay[device_id] = delay

            # packet loss detection
            expected = device_expected_seq[device_id]

            if seq > expected:
                gap = seq - expected
                lost_packets += gap
                bytes_attempted += gap * len(data)

            device_expected_seq[device_id] = seq + 1

            # counters
            received_packets += 1
            active_devices_window.add(device_id)

            pkt_size = len(data)

            bytes_received += pkt_size
            bytes_attempted += pkt_size

            if DEBUG_LOG:
                receiver_log_writer.writerow([
                    seq,device_id,device_type,
                    send_time,recv_time,value,label,
                    round(delay,6)
                ])

except KeyboardInterrupt:
    print("\n[Receiver] Flushing final window...")
    flush_telemetry(time.time())

finally:

    telemetry_file.flush()
    telemetry_file.close()

    if DEBUG_LOG:
        receiver_log_file.flush()
        receiver_log_file.close()

    sock.close()

    total_elapsed = time.time() - start_time
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print(
        f"[Receiver] Shutdown. Total runtime: {total_min}m {total_sec}s | "
        f"Telemetry rows: {telemetry_rows}"
    )