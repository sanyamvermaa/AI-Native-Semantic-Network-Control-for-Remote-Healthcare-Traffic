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
import json
from collections import Counter, defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Config ─────────────────────────────────────────────────────────
BASE_DIR           = os.environ.get("HEALTH_DATA_BASE_DIR", "/home/ayhm23/health_data/csv")
TELEMETRY_INTERVAL = 0.25        # flush every 0.25s
DRAIN_BUDGET       = 0.15        # max time spent draining packets
STATUS_INTERVAL    = 15.0        # print progress every 30 seconds
DEBUG_LOG          = False

# ── CLOSED LOOP ADDITION ─────────────────────────────────────
COMMAND_PORT       = 5006
MODEL_PATH         = "best_network_model.pkl"
DEBOUNCE_WINDOWS   = 5
LATENCY_REPORT_SEC = 60.0

FEATURE_COLS = [
    "bandwidth_usage_bps",
    "throughput_bps",
    "packet_loss_rate",
    "jitter",
    "avg_delay",
    "rolling_loss_mean",
    "rolling_loss_std",
    "rolling_jitter_mean",
    "rolling_delay_mean",
    "rolling_throughput_mean",
    "rolling_throughput_std",
    "loss_delta",
    "jitter_delta",
    "delay_delta",
    "loss_accel",
    "loss_trend_3",
    "throughput_trend_3",
    "delay_trend_3",
    "delay_slope",
    "loss_slope",
    "active_devices",
    "packets_per_window",
    "loss_x_delay",
    "loss_x_jitter",
]

DECISION_TABLE = {
    ("Stable", "NORMAL"): "FULL_ECG",
    ("Stable", "ALERT"): "FULL_ECG",
    ("Stable", "CRITICAL"): "FULL_ECG_PRIORITY",
    ("Unstable", "NORMAL"): "DOWNSAMPLED_ECG",
    ("Unstable", "ALERT"): "DOWNSAMPLED_ECG",
    ("Unstable", "CRITICAL"): "SEMANTIC_ALERT",
    ("Critical", "NORMAL"): "SEMANTIC_SUMMARY",
    ("Critical", "ALERT"): "SEMANTIC_ALERT",
    ("Critical", "CRITICAL"): "SEMANTIC_CRITICAL",
}

try:
    os.makedirs(BASE_DIR, exist_ok=True)
except Exception as exc:
    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    # Fallback for environments where /home/... is not writable.
    fallback_base = os.path.join(os.getcwd(), "csv")
    os.makedirs(fallback_base, exist_ok=True)
    print(f"[Receiver][WARN] BASE_DIR '{BASE_DIR}' unavailable ({exc}). Using '{fallback_base}'.")
    BASE_DIR = fallback_base

# ── Socket Setup ───────────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(False)
sock.bind(("0.0.0.0", 9000))

# ── CLOSED LOOP ADDITION ─────────────────────────────────────
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

# ── CLOSED LOOP ADDITION ─────────────────────────────────────
last_health_state = "NORMAL"
window_health_state = None
last_sender_ip = None

window_history = deque(maxlen=10)

model = None
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Critical", "Stable", "Unstable"])

try:
    model = joblib.load(MODEL_PATH)
    print(f"[Receiver] Loaded model: {MODEL_PATH}")
except FileNotFoundError:
    print(f"[Receiver][WARN] Model not found: {MODEL_PATH}. Running without inference.")
except Exception as exc:
    print(f"[Receiver][WARN] Failed to load model: {exc}. Running without inference.")

command_log_file = None
command_log_writer = None
command_log_path = os.path.join(BASE_DIR, "command_log.csv")
try:
    command_log_file = open(command_log_path, "w", newline="")
    command_log_writer = csv.writer(command_log_file)
    command_log_writer.writerow([
        "timestamp",
        "network_state",
        "health_state",
        "command",
        "latency_ms",
        "consecutive_windows_before_send",
    ])
    command_log_file.flush()
except Exception as exc:
    print(f"[Receiver][WARN] Could not open command log at '{command_log_path}': {exc}")

pending_command = None
pending_count = 0
last_sent_command = None
last_network_state = None

commands_sent = 0
command_counter = Counter()
mode_change_count = 0
network_transition_counter = Counter()

latency_samples_ms = []
last_latency_report = time.time()

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

def _rolling_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


# ── CLOSED LOOP ADDITION ─────────────────────────────────────
def build_feature_vector(current_row, history_rows):
    if len(history_rows) < 10:
        return None

    losses = [r["packet_loss_rate"] for r in history_rows]
    jitters = [r["jitter"] for r in history_rows]
    delays = [r["avg_delay"] for r in history_rows]
    throughputs = [r["throughput_bps"] for r in history_rows]

    latest_loss = losses[-1]
    latest_jitter = jitters[-1]
    latest_delay = delays[-1]
    latest_throughput = throughputs[-1]

    prev_loss = losses[-2]
    prev_jitter = jitters[-2]
    prev_delay = delays[-2]
    prev_loss_delta = losses[-2] - losses[-3]

    xs = np.arange(8)

    features = {
        "bandwidth_usage_bps": current_row["bandwidth_usage_bps"],
        "throughput_bps": current_row["throughput_bps"],
        "packet_loss_rate": latest_loss,
        "jitter": latest_jitter,
        "avg_delay": latest_delay,
        "rolling_loss_mean": float(np.mean(losses)),
        "rolling_loss_std": _rolling_std(losses),
        "rolling_jitter_mean": float(np.mean(jitters)),
        "rolling_delay_mean": float(np.mean(delays)),
        "rolling_throughput_mean": float(np.mean(throughputs)),
        "rolling_throughput_std": _rolling_std(throughputs),
        "loss_delta": latest_loss - prev_loss,
        "jitter_delta": latest_jitter - prev_jitter,
        "delay_delta": latest_delay - prev_delay,
        "loss_accel": (latest_loss - prev_loss) - prev_loss_delta,
        "loss_trend_3": float(np.mean(losses[-3:]) - np.mean(losses[-8:])),
        "throughput_trend_3": float(np.mean(throughputs[-3:]) - np.mean(throughputs[-8:])),
        "delay_trend_3": float(np.mean(delays[-3:]) - np.mean(delays[-8:])),
        "delay_slope": float(np.polyfit(xs, np.array(delays[-8:]), 1)[0]),
        "loss_slope": float(np.polyfit(xs, np.array(losses[-8:]), 1)[0]),
        "active_devices": current_row["active_devices"],
        "packets_per_window": current_row["packets_per_window"],
        "loss_x_delay": latest_loss * latest_delay,
        "loss_x_jitter": latest_loss * latest_jitter,
    }

    return pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)


def flush_telemetry(now, window_start):
    global bytes_received, bytes_attempted
    global received_packets, lost_packets
    global jitter_sum, jitter_count
    global delay_sum, delay_count
    global active_devices_window, queue_length
    global telemetry_rows
    global window_health_state, last_health_state
    global pending_command, pending_count, last_sent_command
    global commands_sent, mode_change_count
    global last_sender_ip, last_network_state
    global last_latency_report

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

    telemetry_row = {
        "timestamp": round(now, 3),
        "active_devices": len(active_devices_window),
        "bandwidth_usage_bps": round(bandwidth_bps, 2),
        "throughput_bps": round(throughput_bps, 2),
        "packet_loss_rate": round(packet_loss_rate, 6),
        "jitter": round(avg_jitter, 6),
        "avg_delay": round(avg_delay, 6),
        "queue_length": queue_length,
        "packets_per_window": total_attempted,
    }

    telemetry_writer.writerow([
        telemetry_row["timestamp"],
        telemetry_row["active_devices"],
        telemetry_row["bandwidth_usage_bps"],
        telemetry_row["throughput_bps"],
        telemetry_row["packet_loss_rate"],
        telemetry_row["jitter"],
        telemetry_row["avg_delay"],
        telemetry_row["queue_length"],
        telemetry_row["packets_per_window"],
    ])

    telemetry_file.flush()
    telemetry_rows += 1

    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    health_state = window_health_state or last_health_state
    last_health_state = health_state

    window_history.append({
        "packet_loss_rate": telemetry_row["packet_loss_rate"],
        "jitter": telemetry_row["jitter"],
        "avg_delay": telemetry_row["avg_delay"],
        "throughput_bps": telemetry_row["throughput_bps"],
    })

    if model is not None:
        features_df = build_feature_vector(telemetry_row, list(window_history))

        if features_df is not None:
            inf_start = time.time()

            try:
                pred_encoded = model.predict(features_df)
                pred_idx = int(pred_encoded[0])
                network_state = label_encoder.inverse_transform([pred_idx])[0]
                inference_ms = (time.time() - inf_start) * 1000.0

                if last_network_state is not None and network_state != last_network_state:
                    transition = f"{last_network_state}→{network_state}"
                    network_transition_counter[transition] += 1
                last_network_state = network_state

                command = DECISION_TABLE.get((network_state, health_state), "FULL_ECG")

                if command == pending_command:
                    pending_count += 1
                else:
                    pending_command = command
                    pending_count = 1

                required_windows = 2 if network_state == "Critical" else DEBOUNCE_WINDOWS

                should_send = False
                if last_sent_command is None:
                    should_send = True
                elif command != last_sent_command and pending_count >= required_windows:
                    should_send = True

                if should_send and last_sender_ip is not None:
                    send_ts = time.time()
                    total_response_ms = (send_ts - window_start) * 1000.0
                    payload = {
                        "command": command,
                        "network_state": network_state,
                        "health_state": health_state,
                        "timestamp": send_ts,
                        "consecutive_windows": pending_count,
                    }

                    try:
                        command_sock.sendto(
                            json.dumps(payload).encode("utf-8"),
                            (last_sender_ip, COMMAND_PORT),
                        )
                        if command_log_writer is not None and command_log_file is not None:
                            command_log_writer.writerow([
                                round(send_ts, 6),
                                network_state,
                                health_state,
                                command,
                                round(total_response_ms, 3),
                                pending_count,
                            ])
                            command_log_file.flush()

                        if last_sent_command is not None and last_sent_command != command:
                            mode_change_count += 1

                        last_sent_command = command
                        commands_sent += 1
                        command_counter[command] += 1
                        latency_samples_ms.append(total_response_ms)

                        print(
                            f"[PREDICT] t={now:.2f}  net={network_state}  "
                            f"health={health_state}  → {command}"
                        )
                        print(
                            f"[LATENCY] inference={inference_ms:.1f}ms  "
                            f"total_response={total_response_ms:.1f}ms"
                        )
                    except Exception as cmd_exc:
                        print(f"[Receiver][WARN] Failed to send command: {cmd_exc}")
                else:
                    print(
                        f"[PREDICT] t={now:.2f}  net={network_state}  "
                        f"health={health_state}  → {command}"
                    )

            except Exception as inf_exc:
                print(f"[Receiver][WARN] Inference failed: {inf_exc}")

    if time.time() - last_latency_report >= LATENCY_REPORT_SEC:
        avg_latency = sum(latency_samples_ms) / len(latency_samples_ms) if latency_samples_ms else 0.0
        print(
            f"[LATENCY] 60s avg total_response={avg_latency:.1f}ms "
            f"(samples={len(latency_samples_ms)})"
        )
        last_latency_report = time.time()

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
    window_health_state   = None

# ── Main loop timers ────────────────────────────────────────────────
start_time = time.time()
next_flush_time = time.time() + TELEMETRY_INTERVAL
last_status_time = time.time()
current_window_start = time.time()

packets_since_status = 0

try:
    while True:

        now = time.time()

        # ── 1. Telemetry flush ──────────────────────────────────────
        if now >= next_flush_time:
            flush_telemetry(now, current_window_start)
            current_window_start = now
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
                label       = parts[5].strip()
            except Exception:
                # ── CLOSED LOOP ADDITION ─────────────────────────────────────
                # Backward-compatible fallback for JSON packets from adaptive sender.
                try:
                    pkt = json.loads(data.decode().strip())
                    device_id = int(pkt.get("device_id", -1))
                    seq = int(pkt["seq"])
                    send_time = float(pkt["timestamp"])
                    device_type = str(pkt.get("device_type", "Unknown"))
                    value = pkt.get("heart_rate", pkt.get("value", -1))
                    label = str(pkt.get("health_label", "NORMAL")).strip()
                except Exception:
                    continue

            device_last_seen[device_id] = recv_time
            window_health_state = label
            last_sender_ip = addr[0]

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
    flush_telemetry(time.time(), current_window_start)

finally:

    telemetry_file.flush()
    telemetry_file.close()
    if command_log_file is not None:
        command_log_file.flush()
        command_log_file.close()

    if DEBUG_LOG:
        receiver_log_file.flush()
        receiver_log_file.close()

    sock.close()
    command_sock.close()

    total_elapsed = time.time() - start_time
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print(
        f"[Receiver] Shutdown. Total runtime: {total_min}m {total_sec}s | "
        f"Telemetry rows: {telemetry_rows}"
    )

    # ── CLOSED LOOP ADDITION ─────────────────────────────────────
    total_cmd = commands_sent
    full_ecg_count = command_counter["FULL_ECG"] + command_counter["FULL_ECG_PRIORITY"]
    downsampled_count = command_counter["DOWNSAMPLED_ECG"]
    semantic_count = (
        command_counter["SEMANTIC_ALERT"]
        + command_counter["SEMANTIC_CRITICAL"]
        + command_counter["SEMANTIC_SUMMARY"]
    )

    def pct(v):
        return (100.0 * v / total_cmd) if total_cmd else 0.0

    avg_latency = sum(latency_samples_ms) / len(latency_samples_ms) if latency_samples_ms else 0.0
    most_common_transition = "N/A"
    if network_transition_counter:
        most_common_transition = network_transition_counter.most_common(1)[0][0]

    print("── Closed Loop Summary ──────────────────────────────")
    print(f"Total commands sent     : {total_cmd}")
    print(
        "Mode distribution       : "
        f"FULL_ECG={pct(full_ecg_count):.0f}% "
        f"DOWNSAMPLED={pct(downsampled_count):.0f}% "
        f"SEMANTIC={pct(semantic_count):.0f}%"
    )
    print(f"Avg response latency    : {avg_latency:.0f}ms")
    print(f"Mode changes            : {mode_change_count}")
    print(f"Most common transition  : {most_common_transition}")