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
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from common import choose_network_state, default_base_dir, DEVICE_LAYOUT

# ---------------------------------------------------------------------------
# Optional semantic codec imports
# ---------------------------------------------------------------------------
_SEMANTIC_DIR = str(Path(__file__).resolve().parents[1] / "semantic")
if _SEMANTIC_DIR not in sys.path:
    sys.path.insert(0, _SEMANTIC_DIR)

try:
    from semantic_encoder import SemanticEncoder          # type: ignore
    from channel_quantizer import decode_payload          # type: ignore
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

BASE_DIR = default_base_dir()
TELEMETRY_INTERVAL = 0.25 #reciever computes telemetry and makes predictions every 250msq
DRAIN_BUDGET = 0.15 #when a packet is received, spend up to 150ms draining the socket to get the most recent data for the window, then compute telemetry and predict
STATUS_INTERVAL = 15.0 #periodically print status updates every 15 seconds
MIN_HISTORY_WINDOWS = 8
REORDER_TRACK_LIMIT = 2048
PACKET_SIZE_EMA_ALPHA = 0.2
CONF_THRESHOLDS = {"Stable": 0.60, "Unstable": 0.55, "Critical": 0.60}
OVERLOAD_DRAIN_UTILIZATION = 0.90
# Model drift: warn after this many consecutive windows where ML != heuristic
DRIFT_WINDOW = 20
DRIFT_ALERT_EVERY = 5   # re-emit warning every N windows after first alert

WARD_IP = os.getenv("WARD_CONTROLLER_IP", "10.0.0.1")
WARD_PORT = int(os.getenv("WARD_CONTROLLER_PORT", "5006"))
MODEL_PATH_ENV = os.getenv("HEALTH_NETWORK_MODEL_PATH", "")

os.makedirs(BASE_DIR, exist_ok=True)

LISTEN_PORT    = int(os.getenv("RECEIVER_PORT", "9000"))
DRAIN_LOG_PATH = os.getenv("SEMANTIC_DRAIN_PATH", os.path.join(BASE_DIR, "semantic_drain.csv"))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #for receiving clinical data from devices
sock.setblocking(False)
sock.bind(("0.0.0.0", LISTEN_PORT))
try:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
except OSError:
    pass

ward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #for sending telemetry and predictions to the ward controller

print(f"\n[Receiver] Closed-loop gateway listening on :{LISTEN_PORT}")
print(f"[Receiver] Ward controller target: {WARD_IP}:{WARD_PORT}\n")

device_expected_seq = defaultdict(lambda: 1)
device_prev_delay = defaultdict(lambda: None)
device_last_seen = {}
device_missing_seqs = defaultdict(set)
device_avg_packet_size = defaultdict(lambda: 64.0)

bytes_received = 0
bytes_attempted = 0
received_packets = 0
lost_packets = 0
estimated_lost_bytes = 0
reordered_recovered_packets = 0
drain_cycle_count = 0
drain_cycle_max_ms = 0.0
drain_cycle_sum_ms = 0.0
drain_budget_hits = 0
drain_backlog_hits = 0
jitter_sum = 0.0
jitter_count = 0
delay_sum = 0.0
delay_count = 0
active_devices_window = set()
label_counts = defaultdict(int)
_HEALTH_EMA_ALPHA = 0.3
_health_ema: Dict[str, float] = {"NORMAL": 0.0, "ALERT": 0.0, "CRITICAL": 0.0}
decode_confidence_sum = 0.0
decode_confidence_count = 0
# Latest latent vector per device_id — forwarded to ward_controller in each flush
_last_z: Dict[int, Dict] = {}  # {device_id: {"device_type": str, "z": List[float]}}

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
        "predictor_source",
        "prediction_confidence",
        "estimated_lost_bytes",
        "reordered_recovered_packets",
        "receiver_overloaded",
        "drain_max_ms",
        "drain_avg_ms",
        "drain_budget_hits",
        "drain_backlog_hits",
    ]
)
telemetry_file.flush() #ensure header is written to disk immediately

# Semantic drain log — one row per decoded semantic packet
drain_log_file   = open(DRAIN_LOG_PATH, "w", newline="", encoding="utf-8")
drain_log_writer = csv.writer(drain_log_file)
drain_log_writer.writerow(["timestamp", "device_id", "device_type", "seq",
                            "n_dims", "decoded_label", "confidence"])
drain_log_file.flush()
print(f"[Receiver] Semantic drain log : {DRAIN_LOG_PATH}")

telemetry_rows = 0
window_history: List[Dict[str, float]] = []
_drift_disagree_streak: int = 0


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


_LABEL_CLASSES: Optional[List[str]] = None

def _load_label_classes() -> List[str]:
    global _LABEL_CLASSES
    if _LABEL_CLASSES is not None:
        return _LABEL_CLASSES
    map_path = Path(__file__).resolve().parents[2] / "models" / "label_classes.json"
    default = ["Critical", "Stable", "Unstable"]
    if map_path.exists():
        import json as _json
        try:
            with open(map_path, encoding="utf-8") as _f:
                loaded = _json.load(_f)
            if isinstance(loaded, list) and len(loaded) >= 3:
                _LABEL_CLASSES = [str(x) for x in loaded]
            else:
                print(f"[WARN] Invalid label class mapping in {map_path}; using default mapping")
                _LABEL_CLASSES = default
        except Exception as exc:
            print(f"[WARN] Failed to read label class mapping ({exc}); using default mapping")
            _LABEL_CLASSES = default
    else:
        _LABEL_CLASSES = default
    return _LABEL_CLASSES

def decode_model_prediction(raw_pred) -> str:
    if isinstance(raw_pred, str):
        if raw_pred in ("Stable", "Unstable", "Critical"):
            return raw_pred
        return choose_network_state(0.0, 0.0, 0.0)

    try:
        idx = int(raw_pred)
    except Exception:
        return choose_network_state(0.0, 0.0, 0.0)

    classes = _load_label_classes()
    if 0 <= idx < len(classes):
        return classes[idx]
    return "Critical"


def _check_model_scale(path: Path) -> None:
    """Verify the model's JSON metadata sidecar declares ms-scale delay/jitter.

    The original realistic_network_dataset.csv stored avg_delay and jitter in
    *seconds*.  health_receiver.py converts its raw timing values to milliseconds
    before inference (avg_delay * 1000.0).  A model trained on the un-converted
    seconds-scale CSV will therefore receive features that are 1000× too large
    and will misclassify nearly every window.  The metadata sidecar written by
    train_model.py and retrain_on_live_data.py records the scale used at training
    time so we can detect this mismatch here.
    """
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        print(
            f"[WARN][SCALE] No metadata sidecar found for {path.name}. "
            "Cannot verify delay/jitter scale. If this model was trained on "
            "realistic_network_dataset.csv without the *1000 conversion, "
            "predictions will be wrong. Re-run train_model.py to generate a "
            "validated ms-scale model."
        )
        return
    try:
        with open(meta_path, encoding="utf-8") as _mf:
            meta = json.load(_mf)
    except Exception as exc:
        print(f"[WARN][SCALE] Could not read {meta_path.name}: {exc}")
        return
    delay_scale = meta.get("delay_scale", "unknown")
    if delay_scale != "ms":
        print(
            f"[ERROR][SCALE] Model {path.name} metadata reports "
            f"delay_scale='{delay_scale}' — expected 'ms'. "
            "The receiver feeds milliseconds; features will be mis-scaled "
            "by 1000×. Falling back to heuristic predictor is strongly advised. "
            "Re-run train_model.py to regenerate a correct ms-scale model."
        )
    else:
        trained_on = meta.get("trained_on", "unknown")
        trained_at = meta.get("trained_at", "unknown")
        print(
            f"[MODEL] Scale OK — delay_scale=ms  "
            f"trained_on={trained_on}  trained_at={trained_at}"
        )


def load_model():
    joblib = None
    try:
        joblib = importlib.import_module("joblib")
    except Exception:
        pass

    if joblib is None:
        print("[WARN] joblib unavailable; using heuristic predictor.")
        return None

    if MODEL_PATH_ENV == "DISABLE":
        print("[MODEL] Model loading disabled by environment variable; using heuristic predictor.")
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
                print(f"\n[MODEL] Loaded model: {path}")
                _check_model_scale(path)
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

    #short windows to capture recent trends without too much smoothing
    loss3 = mean_or_zero(rolling(loss_vals, 3))
    loss8 = mean_or_zero(rolling(loss_vals, 8))
    thr3 = mean_or_zero(rolling(thr_vals, 3))
    thr8 = mean_or_zero(rolling(thr_vals, 8))
    delay3 = mean_or_zero(rolling(delay_vals, 3))
    delay8 = mean_or_zero(rolling(delay_vals, 8))

    #24 features total - 
    # original 7 plus engineered features based on recent history and trends
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

# ---------------------------------------------------------------------------
# Per-device-type semantic decoder instances
# ---------------------------------------------------------------------------
# Only ECG and BloodPressure have trained encoder/decoder models.
# SpO2, Temperature, Respiration do not have models and stay on the raw path.
SEMANTIC_CAPABLE_TYPES: frozenset = frozenset({"ECG", "BloodPressure"})

device_decoders: Dict[str, "SemanticEncoder"] = {}
if _SEMANTIC_AVAILABLE:
    _models_dir = Path(__file__).resolve().parents[2] / "models" / "semantic"
    for _, _dev_type in DEVICE_LAYOUT:
        if _dev_type not in device_decoders and _dev_type in SEMANTIC_CAPABLE_TYPES:
            try:
                device_decoders[_dev_type] = SemanticEncoder(_dev_type, str(_models_dir))
                print(f"[Receiver] Loaded semantic decoder for {_dev_type}")
            except Exception as _e:
                print(f"[WARN] Could not load semantic decoder for {_dev_type}: {_e}")
    # Explicitly skip non-capable device types
    for _, _dev_type in DEVICE_LAYOUT:
        if _dev_type not in SEMANTIC_CAPABLE_TYPES:
            print(f"[Receiver] {_dev_type}: no semantic model — raw-only path")

#flush current telemetry and send initial state to ward controller 
#predict initial network state based on empty history

def flush_telemetry(now: float) -> None:
    global bytes_received, bytes_attempted
    global received_packets, lost_packets
    global estimated_lost_bytes, reordered_recovered_packets
    global drain_cycle_count, drain_cycle_max_ms, drain_cycle_sum_ms
    global drain_budget_hits, drain_backlog_hits
    global jitter_sum, jitter_count
    global delay_sum, delay_count
    global active_devices_window
    global telemetry_rows
    global label_counts
    global decode_confidence_sum, decode_confidence_count
    global _last_z
    global _health_ema
    global device_missing_seqs
    global _drift_disagree_streak

    total_attempted = received_packets + lost_packets
    packet_loss_rate = (lost_packets / total_attempted) if total_attempted > 0 else 0.0
    avg_jitter = (jitter_sum / jitter_count) if jitter_count > 0 else 0.0
    avg_delay = (delay_sum / delay_count) if delay_count > 0 else 0.0

    throughput_bps = bytes_received / TELEMETRY_INTERVAL
    bandwidth_bps = bytes_attempted / TELEMETRY_INTERVAL
    drain_avg_ms = (drain_cycle_sum_ms / drain_cycle_count) if drain_cycle_count > 0 else 0.0
    drain_util_pct = (drain_cycle_max_ms / (DRAIN_BUDGET * 1000.0)) if DRAIN_BUDGET > 0 else 0.0
    receiver_overloaded = (
        drain_backlog_hits > 0
        or drain_budget_hits > 0
        or drain_util_pct >= OVERLOAD_DRAIN_UTILIZATION
    )

    avg_decode_confidence = (
        decode_confidence_sum / decode_confidence_count
        if decode_confidence_count > 0 else None
    )

    heuristic_state = choose_network_state(packet_loss_rate, avg_delay * 1000.0, avg_jitter * 1000.0)
    # EMA-smoothed clinical state (avoids single-packet flap-triggers)
    _total = sum(label_counts.values()) or 1
    for _cls in ("NORMAL", "ALERT", "CRITICAL"):
        _obs = label_counts.get(_cls, 0) / _total
        _health_ema[_cls] = _HEALTH_EMA_ALPHA * _obs + (1 - _HEALTH_EMA_ALPHA) * _health_ema[_cls]
    health_state = max(_health_ema, key=_health_ema.__getitem__)

    if avg_decode_confidence is not None and avg_decode_confidence < 0.55:
        if health_state != "CRITICAL":
            health_state = "NORMAL"
            print(
                f"[SEMANTIC] Low decode confidence {avg_decode_confidence:.2f} "
                "\u2014 health state held at NORMAL"
            )

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
    if len(window_history) > 64:
        window_history.pop(0)

    network_state = heuristic_state
    predictor_source = "heuristic"
    prediction_confidence = None
    if model is not None and len(window_history) >= MIN_HISTORY_WINDOWS:
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

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vector)[0]
                best_idx = max(range(len(probs)), key=lambda i: probs[i])
                prediction_confidence = float(probs[best_idx])
                classes = list(getattr(model, "classes_", []))
                pred_raw = classes[best_idx] if classes else best_idx
                pred_state = decode_model_prediction(pred_raw)

                conf_floor = CONF_THRESHOLDS.get(pred_state, 0.55)
                if prediction_confidence < conf_floor:
                    network_state = "Unstable" if pred_state == "Stable" else "Critical"
                    predictor_source = "model_low_conf"
                else:
                    network_state = pred_state
                    predictor_source = "model"
            else:
                pred_raw = model.predict(vector)[0]
                network_state = decode_model_prediction(pred_raw)
                predictor_source = "model_no_proba"
        except Exception as exc:
            print(f"[WARN] model inference failed: {exc}")
    elif model is not None:
        predictor_source = "insufficient_history"

    conf_text = f" conf={prediction_confidence:.2f}" if prediction_confidence is not None else ""
    print(
        f"[PREDICT] source={predictor_source} state={network_state} "
        f"loss={packet_loss_rate*100.0:.2f}% delay={avg_delay*1000.0:.2f}ms "
        f"jitter={avg_jitter*1000.0:.2f}ms devices={len(active_devices_window)}{conf_text}"
    )

    # Model drift detection: track consecutive windows where ML disagrees with heuristic.
    # Persistent divergence means the training distribution may no longer match live traffic.
    if predictor_source in ("model", "model_low_conf", "model_no_proba") \
            and network_state != heuristic_state:
        _drift_disagree_streak += 1
        over_threshold = _drift_disagree_streak >= DRIFT_WINDOW
        repeat_alert = over_threshold and (_drift_disagree_streak - DRIFT_WINDOW) % DRIFT_ALERT_EVERY == 0
        if _drift_disagree_streak == DRIFT_WINDOW or repeat_alert:
            print(
                f"[WARN][DRIFT] Model disagrees with heuristic for "
                f"{_drift_disagree_streak} consecutive windows "
                f"(model={network_state}, heuristic={heuristic_state}). "
                "Model may be stale — consider retraining."
            )
    else:
        _drift_disagree_streak = 0

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
            predictor_source,
            round(prediction_confidence, 4) if prediction_confidence is not None else "",
            int(estimated_lost_bytes),
            int(reordered_recovered_packets),
            int(receiver_overloaded),
            round(drain_cycle_max_ms, 3),
            round(drain_avg_ms, 3),
            int(drain_budget_hits),
            int(drain_backlog_hits),
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
        "predictor_source": predictor_source,
        "prediction_confidence": prediction_confidence,
        "active_devices": len(active_devices_window),
        "packets_per_window": total_attempted,
        "estimated_lost_bytes": int(estimated_lost_bytes),
        "reordered_recovered_packets": int(reordered_recovered_packets),
        "receiver_overloaded": receiver_overloaded,
        "drain_max_ms": drain_cycle_max_ms,
        "drain_avg_ms": drain_avg_ms,
        "drain_budget_hits": drain_budget_hits,
        "drain_backlog_hits": drain_backlog_hits,
        "decode_confidence": avg_decode_confidence,
        "semantic_packets_in_window": decode_confidence_count,
        "z_per_device": [
            {"device_id": did, "device_type": entry["device_type"], "z": entry["z"]}
            for did, entry in _last_z.items()
        ],
    }

    try:
        ward_sock.sendto(json.dumps(payload).encode("utf-8"), (WARD_IP, WARD_PORT))
    except Exception as exc:
        print(f"[WARN] Failed to forward telemetry to ward controller: {exc}")

    telemetry_rows += 1

    bytes_received = 0
    bytes_attempted = 0
    received_packets = 0
    lost_packets = 0
    estimated_lost_bytes = 0
    reordered_recovered_packets = 0
    drain_cycle_count = 0
    drain_cycle_max_ms = 0.0
    drain_cycle_sum_ms = 0.0
    drain_budget_hits = 0
    drain_backlog_hits = 0
    jitter_sum = 0.0
    jitter_count = 0
    delay_sum = 0.0
    delay_count = 0
    active_devices_window = set()
    label_counts = defaultdict(int)
    decode_confidence_sum = 0.0
    decode_confidence_count = 0

    # Keep missing-sequence ledger across telemetry windows for reorder recovery,
    # but cap memory per device.
    for dev_id, missing in list(device_missing_seqs.items()):
        if len(missing) > REORDER_TRACK_LIMIT:
            expected = device_expected_seq[dev_id]
            keep_from = max(1, expected - REORDER_TRACK_LIMIT)
            device_missing_seqs[dev_id] = {s for s in missing if s >= keep_from}


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
                # Strip RAW_LEGACY prefix emitted by semantic-capable senders
                # whose encoder buffer hasn't filled yet.  After stripping,
                # the remainder is a normal CSV line.
                _is_raw_legacy = raw.startswith("RAW_LEGACY:")
                if _is_raw_legacy:
                    raw = raw[len("RAW_LEGACY:"):].strip()
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

            # --- Semantic decoder path (overrides label if packet was encoded) ---
            # Only attempt decode for device types that have trained models.
            # Fast-path: skip decode on raw CSV packets (not JSON) — these come
            # from legacy fallback or RAW_LEGACY-tagged packets during the
            # encoder buffer fill window.  decode_payload() expects JSON and
            # would return None anyway, but skipping avoids the overhead and
            # spurious log warnings.
            decoded_result = None
            if (_SEMANTIC_AVAILABLE
                    and device_type in SEMANTIC_CAPABLE_TYPES
                    and not _is_raw_legacy
                    and raw.startswith("{")):
                decoded_result = decode_payload(data)

            if decoded_result is not None:
                z_full   = decoded_result["z_full"]
                dev_type = decoded_result.get("device_type", "UNKNOWN")
                _n_dims  = int(decoded_result.get("n_dims", 0))
                decoder  = device_decoders.get(dev_type)
                if decoder is not None:
                    result = decoder.decode(z_full)
                    label  = result["clinical_state"]
                    _conf  = result["confidence"]
                    decode_confidence_sum   += _conf
                    decode_confidence_count += 1
                    drain_log_writer.writerow([
                        round(recv_time, 6), device_id, dev_type,
                        seq, _n_dims, label, round(_conf, 4),
                    ])
                else:
                    label = str(decoded_result.get("label", "NORMAL")).strip().upper()
                    drain_log_writer.writerow([
                        round(recv_time, 6), device_id, dev_type,
                        seq, _n_dims, label, "",
                    ])
                drain_log_file.flush()
                # Store latest z for this device for ward_controller forwarding
                _last_z[device_id] = {"device_type": dev_type, "z": list(z_full)}
            # else: label already set by legacy parsing path above

            device_last_seen[device_id] = recv_time
            delay = recv_time - send_time
            delay_sum += delay
            delay_count += 1

            prev = device_prev_delay[device_id]
            if prev is not None:
                jitter_sum += abs(delay - prev)
                jitter_count += 1
            device_prev_delay[device_id] = delay

            packet_size = len(data)
            avg_packet_size = device_avg_packet_size[device_id]
            device_avg_packet_size[device_id] = (
                (1.0 - PACKET_SIZE_EMA_ALPHA) * avg_packet_size
                + PACKET_SIZE_EMA_ALPHA * packet_size
            )

            expected = device_expected_seq[device_id]
            missing_for_dev = device_missing_seqs[device_id]
            if seq > expected:
                gap = seq - expected
                lost_packets += gap
                est_lost = int(round(gap * device_avg_packet_size[device_id]))
                bytes_attempted += est_lost
                estimated_lost_bytes += est_lost
                if gap <= REORDER_TRACK_LIMIT:
                    missing_for_dev.update(range(expected, seq))
                else:
                    missing_for_dev.clear()
                device_expected_seq[device_id] = seq + 1
            elif seq == expected:
                device_expected_seq[device_id] = seq + 1
            else:
                # Reordered packet that was previously considered lost in this window.
                if seq in missing_for_dev:
                    missing_for_dev.remove(seq)
                    if lost_packets > 0:
                        lost_packets -= 1
                    est_recover = int(round(device_avg_packet_size[device_id]))
                    if estimated_lost_bytes >= est_recover:
                        estimated_lost_bytes -= est_recover
                    if bytes_attempted >= est_recover:
                        bytes_attempted -= est_recover
                    reordered_recovered_packets += 1

            received_packets += 1
            active_devices_window.add(device_id)

            bytes_received += packet_size
            bytes_attempted += packet_size

            if label in ("NORMAL", "ALERT", "CRITICAL"):
                label_counts[label] += 1

        drain_elapsed_ms = (time.time() - drain_start) * 1000.0
        drain_cycle_count += 1
        drain_cycle_sum_ms += drain_elapsed_ms
        if drain_elapsed_ms > drain_cycle_max_ms:
            drain_cycle_max_ms = drain_elapsed_ms
        if drain_elapsed_ms >= (DRAIN_BUDGET * 1000.0 * 0.99):
            drain_budget_hits += 1
        still_ready, _, _ = select.select([sock], [], [], 0)
        if still_ready:
            drain_backlog_hits += 1

except KeyboardInterrupt:
    print("\n[Receiver] Flushing final window...")
    flush_telemetry(time.time())
finally:
    telemetry_file.flush()
    telemetry_file.close()
    drain_log_file.flush()
    drain_log_file.close()
    sock.close()
    ward_sock.close()

    total_elapsed = time.time() - start_time
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print(
        f"[Receiver] Shutdown. Total runtime: {total_min}m {total_sec}s | "
        f"Telemetry rows: {telemetry_rows}"
    )
