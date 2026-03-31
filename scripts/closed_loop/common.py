import csv
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


def default_base_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get(
        "HEALTH_DATA_BASE_DIR",
        os.path.join(os.path.dirname(os.path.dirname(here)), "data", "logs"),
    )


DEVICE_LAYOUT: List[Tuple[int, str]] = [
    (0, "ECG"),
    (1, "ECG"),
    (2, "SpO2"),
    (3, "SpO2"),
    (4, "BloodPressure"),
    (5, "BloodPressure"),
    (6, "Temperature"),
    (7, "Respiration"),
]


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    direct = safe_float(value)
    if direct is not None:
        return direct
    try:
        return time.mktime(time.strptime(str(value), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return None


def normalize_network_state(raw: Any) -> str:
    if raw is None:
        return "UNKNOWN"
    value = str(raw).strip().lower()
    if value == "stable":
        return "Stable"
    if value == "unstable":
        return "Unstable"
    if value == "critical":
        return "Critical"
    return "UNKNOWN"


def normalize_health_state(raw: Any) -> str:
    if raw is None:
        return "UNKNOWN"
    value = str(raw).strip().lower()
    if value == "normal":
        return "NORMAL"
    if value == "alert":
        return "ALERT"
    if value == "critical":
        return "CRITICAL"
    return "UNKNOWN"

#rule-based network classifier based on thresholds for packet loss, delay, and jitter
def choose_network_state(packet_loss_rate: float, avg_delay_ms: float, jitter_ms: float) -> str:
    if packet_loss_rate >= 0.08 or avg_delay_ms >= 130.0 or jitter_ms >= 35.0:
        return "Critical"
    if packet_loss_rate >= 0.03 or avg_delay_ms >= 35.0 or jitter_ms >= 8.0:
        return "Unstable"
    return "Stable"

#classifier based on counts of NORMAL, ALERT, and CRITICAL labels in the receiver window
def choose_health_state(label_counts: Dict[str, int]) -> str:
    if label_counts.get("CRITICAL", 0) > 0:
        return "CRITICAL"
    if label_counts.get("ALERT", 0) > 0:
        return "ALERT"
    if label_counts.get("NORMAL", 0) > 0:
        return "NORMAL"
    return "UNKNOWN"

#Brain of the closed-loop system 
def policy_command(network_state: str, health_state: str) -> str:
    net = normalize_network_state(network_state)
    health = normalize_health_state(health_state)

    # UNKNOWN health means the receiver window had zero labelled packets
    # (startup race or a brief loss spike).  Treat it as NORMAL — benefit
    # of the doubt — so a momentary empty window on a stable network does
    # NOT collapse the fleet to SEMANTIC_SUMMARY.
    # Critical network + UNKNOWN stays mapped via the Critical/NORMAL row
    # (SEMANTIC_SUMMARY), which is intentional: bad channel AND no signal.
    if health == "UNKNOWN":
        health = "NORMAL"

    policy = {
        ("Stable",   "NORMAL"):   "FULL_ECG",
        ("Stable",   "ALERT"):    "FULL_ECG_PRIORITY",
        ("Stable",   "CRITICAL"): "SEMANTIC_CRITICAL",
        ("Unstable", "NORMAL"):   "DOWNSAMPLED_ECG",
        ("Unstable", "ALERT"):    "SEMANTIC_ALERT",
        ("Unstable", "CRITICAL"): "SEMANTIC_CRITICAL",
        ("Critical", "NORMAL"):   "SEMANTIC_SUMMARY",
        ("Critical", "ALERT"):    "SEMANTIC_ALERT",
        ("Critical", "CRITICAL"): "SEMANTIC_CRITICAL",
    }
    # Fallback: DOWNSAMPLED_ECG for any unexpected combination — conservative
    # but not as disruptive as suppressing everything with SEMANTIC_SUMMARY.
    return policy.get((net, health), "DOWNSAMPLED_ECG")


def tail_csv_lines(filepath: str, n_lines: int) -> List[str]:
    try:
        if n_lines <= 0 or not os.path.exists(filepath):
            return []

        with open(filepath, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return []

            block = min(size, max(4096, n_lines * 220))
            if block < size:
                f.seek(-block, os.SEEK_END)
                chunk = f.read().decode("utf-8", errors="replace")
                lines = chunk.splitlines()
                if lines:
                    lines = lines[1:]
            else:
                f.seek(0)
                chunk = f.read().decode("utf-8", errors="replace")
                lines = chunk.splitlines()

        return [line for line in lines if line.strip()][-n_lines:]
    except Exception:
        return []


def read_csv_tail_rows(filepath: str, n_rows: int) -> List[Dict[str, str]]:
    try:
        if n_rows <= 0 or not os.path.exists(filepath):
            return []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip()
        if not header:
            return []

        tail_data = tail_csv_lines(filepath, n_rows + 5)
        data_lines = [line for line in tail_data if line.strip() and line.strip() != header]
        if not data_lines:
            return []

        payload = header + "\n" + "\n".join(data_lines[-n_rows:])
        reader = csv.DictReader(io.StringIO(payload))
        return [dict(row) for row in reader][-n_rows:]
    except Exception:
        return []


def safe_read_json(filepath: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if fallback is None:
        fallback = {}
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else fallback
    except Exception:
        return fallback
