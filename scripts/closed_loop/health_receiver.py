"""
Closed-loop receiver that computes per-window telemetry, infers network/health
states, and publishes updates to ward_controller over UDP.
"""

import csv
import importlib
import json
import os
import select
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common import choose_health_state, choose_network_state, default_base_dir

BASE_DIR = default_base_dir()
TELEMETRY_INTERVAL = 0.25
DRAIN_BUDGET = 0.15
STATUS_INTERVAL = 15.0

WARD_IP = os.getenv("WARD_CONTROLLER_IP", "10.0.0.1")
WARD_PORT = int(os.getenv("WARD_CONTROLLER_PORT", "5006"))
MODEL_PATH_ENV = os.getenv("HEALTH_NETWORK_MODEL_PATH", "")

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
window_history: List[Dict[str, float]] = []


def mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std_or_zero(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean_or_zero(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def simple_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = mean_or_zero(values)
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = i - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    return num / den if den > 0 else 0.0


def decode_model_prediction(raw_pred) -> str:
    if isinstance(raw_pred, str):
        if raw_pred in ("Stable", "Unstable", "Critical"):
            return raw_pred
        return choose_network_state(0.0, 0.0, 0.0)

    try:
        idx = int(raw_pred)
    except Exception:
        return choose_network_state(0.0, 0.0, 0.0)

    classes = ["Critical", "Stable", "Unstable"]
    if 0 <= idx < len(classes):
        return classes[idx]
    return "Critical"


def load_model():
    joblib = None
    try:
        joblib = importlib.import_module("joblib")
    except Exception:
        pass

    if joblib is None:
        print("[WARN] joblib unavailable; using heuristic predictor.")
        return None

    model_candidates = []
    if MODEL_PATH_ENV:
        model_candidates.append(Path(MODEL_PATH_ENV))

    repo_root = Path(__file__).resolve().parents[2]
    model_candidates.extend(
        [
            repo_root / "models" / "best_network_model.pkl",
            repo_root / "models" / "xgboost_network_model.pkl",
            repo_root / "models" / "robust_network_model.pkl",
        ]
    )

    for path in model_candidates:
        try:
            if path.exists():
                model = joblib.load(path)
                print(f"[MODEL] Loaded model: {path}")
                return model
        except Exception as exc:
            print(f"[WARN] Failed to load model {path}: {exc}")

    print("[WARN] No model file loaded; using heuristic predictor.")
    return None


def build_feature_row(curr: Dict[str, float], history: List[Dict[str, float]]) -> Dict[str, float]:
    W = 10
    S = 8

    loss_vals = [h["packet_loss_rate"] for h in history]
    jitter_vals = [h["jitter"] for h in history]
    delay_vals = [h["avg_delay"] for h in history]
    thr_vals = [h["throughput_bps"] for h in history]

    def rolling(vals, n):
        chunk = vals[-n:] if len(vals) >= n else vals
        return chunk

    loss_delta = loss_vals[-1] - loss_vals[-2] if len(loss_vals) >= 2 else 0.0
    jitter_delta = jitter_vals[-1] - jitter_vals[-2] if len(jitter_vals) >= 2 else 0.0
    delay_delta = delay_vals[-1] - delay_vals[-2] if len(delay_vals) >= 2 else 0.0
    prev_loss_delta = loss_vals[-2] - loss_vals[-3] if len(loss_vals) >= 3 else 0.0
    loss_accel = loss_delta - prev_loss_delta

    loss3 = mean_or_zero(rolling(loss_vals, 3))
    loss8 = mean_or_zero(rolling(loss_vals, 8))
    thr3 = mean_or_zero(rolling(thr_vals, 3))
    thr8 = mean_or_zero(rolling(thr_vals, 8))
    delay3 = mean_or_zero(rolling(delay_vals, 3))
    delay8 = mean_or_zero(rolling(delay_vals, 8))

    row = {
        "bandwidth_usage_bps": curr["bandwidth_usage_bps"],
        "throughput_bps": curr["throughput_bps"],
        "packet_loss_rate": curr["packet_loss_rate"],
        "jitter": curr["jitter"],
        "avg_delay": curr["avg_delay"],
        "rolling_loss_mean": mean_or_zero(rolling(loss_vals, W)),
        "rolling_loss_std": std_or_zero(rolling(loss_vals, W)),
        "rolling_jitter_mean": mean_or_zero(rolling(jitter_vals, W)),
        "rolling_delay_mean": mean_or_zero(rolling(delay_vals, W)),
        "rolling_throughput_mean": mean_or_zero(rolling(thr_vals, W)),
        "rolling_throughput_std": std_or_zero(rolling(thr_vals, W)),
        "loss_delta": loss_delta,
        "jitter_delta": jitter_delta,
        "delay_delta": delay_delta,
        "loss_accel": loss_accel,
        "loss_trend_3": loss3 - loss8,
        "throughput_trend_3": thr3 - thr8,
        "delay_trend_3": delay3 - delay8,
        "delay_slope": simple_slope(rolling(delay_vals, S)),
        "loss_slope": simple_slope(rolling(loss_vals, S)),
        "active_devices": curr["active_devices"],
        "packets_per_window": curr["packets_per_window"],
        "loss_x_delay": curr["packet_loss_rate"] * curr["avg_delay"],
        "loss_x_jitter": curr["packet_loss_rate"] * curr["jitter"],
    }
    return row


model = load_model()


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

    heuristic_state = choose_network_state(packet_loss_rate, avg_delay * 1000.0, avg_jitter * 1000.0)
    health_state = choose_health_state(label_counts)

    feature_snapshot = {
        "bandwidth_usage_bps": bandwidth_bps,
        "throughput_bps": throughput_bps,
        "packet_loss_rate": packet_loss_rate,
        "jitter": avg_jitter * 1000.0,
        "avg_delay": avg_delay * 1000.0,
        "active_devices": float(len(active_devices_window)),
        "packets_per_window": float(total_attempted),
    }
    window_history.append(feature_snapshot)
    if len(window_history) > 16:
        window_history.pop(0)

    network_state = heuristic_state
    predictor_source = "heuristic"
    if model is not None:
        try:
            row = build_feature_row(feature_snapshot, window_history)
            feat_names = list(getattr(model, "feature_names_in_", []))
            if feat_names:
                vector = [[row.get(name, 0.0) for name in feat_names]]
            else:
                vector = [[
                    row["bandwidth_usage_bps"],
                    row["throughput_bps"],
                    row["packet_loss_rate"],
                    row["jitter"],
                    row["avg_delay"],
                    row["rolling_loss_mean"],
                    row["rolling_loss_std"],
                    row["rolling_jitter_mean"],
                    row["rolling_delay_mean"],
                    row["rolling_throughput_mean"],
                    row["rolling_throughput_std"],
                    row["loss_delta"],
                    row["jitter_delta"],
                    row["delay_delta"],
                    row["loss_accel"],
                    row["loss_trend_3"],
                    row["throughput_trend_3"],
                    row["delay_trend_3"],
                    row["delay_slope"],
                    row["loss_slope"],
                    row["active_devices"],
                    row["packets_per_window"],
                    row["loss_x_delay"],
                    row["loss_x_jitter"],
                ]]

            pred_raw = model.predict(vector)[0]
            network_state = decode_model_prediction(pred_raw)
            predictor_source = "model"
        except Exception as exc:
            print(f"[WARN] model inference failed: {exc}")

    print(
        f"[PREDICT] source={predictor_source} state={network_state} "
        f"loss={packet_loss_rate*100.0:.2f}% delay={avg_delay*1000.0:.2f}ms "
        f"jitter={avg_jitter*1000.0:.2f}ms devices={len(active_devices_window)}"
    )

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
        "window_sec": TELEMETRY_INTERVAL,
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
                raw = data.decode("utf-8", errors="replace").strip()
                if raw.startswith("{"):
                    packet = json.loads(raw)
                    if not isinstance(packet, dict):
                        continue
                    device_id = int(packet.get("device_id"))
                    seq = int(packet.get("seq"))
                    send_time = float(packet.get("ts") or recv_time)
                    device_type = str(packet.get("device_type") or "UNKNOWN")
                    _value = packet.get("value", packet.get("mean", 0))
                    label = str(packet.get("label") or "NORMAL").strip().upper()
                    _ = device_type, _value
                else:
                    parts = raw.split(",")
                    device_id = int(parts[0])
                    seq = int(parts[1])
                    send_time = float(parts[2])
                    device_type = parts[3]
                    _value = parts[4]
                    label = parts[5].strip().upper()
                    _ = device_type, _value
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
