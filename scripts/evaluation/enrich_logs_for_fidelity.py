#!/usr/bin/env python3
"""
enrich_logs_for_fidelity.py — Run the semantic encoding pipeline over sender logs
to produce enriched CSVs that semantic_fidelity.py can evaluate meaningfully.

The raw sender logs contain only sensor values and ground-truth labels, with no
encoded/decoded information.  This script:

    1. Reads dev10/11 (ECG) and dev12/13 (BloodPressure) sender logs.
    2. Feeds data through the trained SemanticEncoder windows (size 200).
    3. Applies channel_quantizer at different command levels.
    4. Rotates across 12 (network_condition × command) combinations so every
       cell of the semantic_fidelity heatmap is populated.
    5. Writes enriched sender_log_devN_*.csv, network_telemetry.csv, and
       command_log.csv to --output-dir (default: data/eval_logs).

Usage (project root, WSL):
    python3 scripts/evaluation/enrich_logs_for_fidelity.py \\
        [--logs-dir   data/logs] \\
        [--models-dir models/semantic] \\
        [--output-dir data/eval_logs]

Then run semantic_fidelity.py on the enriched data:
    python3 scripts/evaluation/semantic_fidelity.py --logs-dir data/eval_logs
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: make scripts/semantic importable
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(os.path.dirname(_HERE))
_SEMANTIC = os.path.join(_ROOT, "scripts", "semantic")
if _SEMANTIC not in sys.path:
    sys.path.insert(0, _SEMANTIC)

from semantic_encoder import SemanticEncoder        # noqa: E402
from channel_quantizer import quantize, dequantize  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 200      # rows per window — must match metadata.json window_size
LATENT_DIM  = 16

# Devices to process: (device_id, device_type)
DEVICE_CONFIG: List[Tuple[int, str]] = [
    (10, "ECG"),
    (11, "ECG"),
    (12, "BloodPressure"),
    (13, "BloodPressure"),
]

# 12 (network_condition × command) combos — fills every heatmap cell
COMBOS: List[Tuple[str, str]] = [
    ("Stable",   "FULL_ECG"),
    ("Stable",   "SEMANTIC_ALERT"),
    ("Stable",   "SEMANTIC_CRITICAL"),
    ("Stable",   "SEMANTIC_SUMMARY"),
    ("Unstable", "FULL_ECG"),
    ("Unstable", "SEMANTIC_ALERT"),
    ("Unstable", "SEMANTIC_CRITICAL"),
    ("Unstable", "SEMANTIC_SUMMARY"),
    ("Critical", "FULL_ECG"),
    ("Critical", "SEMANTIC_ALERT"),
    ("Critical", "SEMANTIC_CRITICAL"),
    ("Critical", "SEMANTIC_SUMMARY"),
]

# Simulated command latency (ms)
_LATENCY_BASE: Dict[str, float] = {
    "FULL_ECG":           8.0,
    "FULL_ECG_PRIORITY":  7.0,
    "DOWNSAMPLED_ECG":    10.0,
    "SEMANTIC_ALERT":     12.0,
    "SEMANTIC_CRITICAL":  15.0,
    "SEMANTIC_SUMMARY":   10.0,
}
# Extra latency per network condition
_LATENCY_COND_EXTRA: Dict[str, float] = {
    "Stable":   0.0,
    "Unstable": 18.0,
    "Critical": 45.0,
}

# Telemetry stride: write one telemetry row every N packets within a window.
# At 100 Hz (dt=0.01 s), stride=25 → 0.25 s between telemetry rows.
# merge_asof tolerance=0.3 s → every packet is within 0.125 s of a tele row.
TELE_STRIDE = 25

LABEL_CLASSES = ["NORMAL", "ALERT", "CRITICAL"]
# Severity rank: higher = worse
_SEVERITY = {"NORMAL": 0, "ALERT": 1, "CRITICAL": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worst_case_label(labels: List[str]) -> str:
    """Return the highest-severity label from a list."""
    best = "NORMAL"
    for lbl in labels:
        if _SEVERITY.get(lbl.upper(), 0) > _SEVERITY[best]:
            best = lbl.upper()
    return best


def _sim_latency(cmd: str, net_cond: str, rng: random.Random) -> float:
    base  = _LATENCY_BASE.get(cmd, 10.0)
    extra = _LATENCY_COND_EXTRA.get(net_cond, 0.0)
    noise = rng.gauss(0.0, 3.0)
    return max(1.0, base + extra + noise)


def load_device_log(logs_dir: str, dev_id: int, dtype: str) -> Optional[pd.DataFrame]:
    fname = f"sender_log_dev{dev_id}_{dtype}.csv"
    path  = os.path.join(logs_dir, fname)
    if not os.path.isfile(path):
        print(f"[WARN] Not found: {path}")
        return None
    df = pd.read_csv(path)
    for col in ("timestamp", "value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich sender logs with semantic encoding for fidelity evaluation."
    )
    parser.add_argument(
        "--logs-dir",
        default=os.path.join(_ROOT, "data", "logs"),
        help="Directory containing orignal sender_log_devN_*.csv and network_telemetry.csv",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join(_ROOT, "models", "semantic"),
        help="Directory with enc_*.pt, dec_*.pt, metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_ROOT, "data", "eval_logs"),
        help="Destination for enriched CSVs (will be created if absent)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(42)
    np.random.seed(42)

    print(f"[INFO] Logs dir   : {args.logs_dir}")
    print(f"[INFO] Models dir : {args.models_dir}")
    print(f"[INFO] Output dir : {args.output_dir}")

    # ------------------------------------------------------------------
    # 1. Load device logs
    # ------------------------------------------------------------------
    devices: Dict[int, Tuple[str, pd.DataFrame]] = {}
    for dev_id, dtype in DEVICE_CONFIG:
        df = load_device_log(args.logs_dir, dev_id, dtype)
        if df is not None:
            devices[dev_id] = (dtype, df)

    if not devices:
        sys.exit("[ERROR] No device log files found.")

    dev_ids = sorted(devices.keys())
    n_rows  = min(len(df) for _, df in devices.values())
    n_windows = n_rows // WINDOW_SIZE

    print(f"[INFO] Devices    : {dev_ids}")
    print(f"[INFO] Rows/device: {n_rows}  →  {n_windows} windows of {WINDOW_SIZE} rows")
    print(f"[INFO] Combos     : {len(COMBOS)} → ~{n_windows // len(COMBOS)} windows/cell")

    # ------------------------------------------------------------------
    # 2. Create one SemanticEncoder per device_id (not per device_type)
    # ------------------------------------------------------------------
    encoders: Dict[int, SemanticEncoder] = {}
    for dev_id, (dtype, _) in devices.items():
        enc = SemanticEncoder(dtype, args.models_dir)
        if enc._enc is None:
            print(f"[WARN] No model loaded for dev{dev_id} ({dtype}) — skipping.")
            continue
        encoders[dev_id] = enc

    if not encoders:
        sys.exit("[ERROR] No trained encoder models found.")

    print(f"[INFO] Encoders ready: dev{sorted(encoders.keys())}")

    # ------------------------------------------------------------------
    # 3. Process non-overlapping windows
    # ------------------------------------------------------------------
    telemetry_rows: List[dict]            = []
    command_rows:   List[dict]            = []
    enriched_pkts:  Dict[int, List[dict]] = {dev_id: [] for dev_id in devices}

    prev_combo: Optional[Tuple[str, str]] = None
    consecutive = 0

    for wi in range(n_windows):
        net_cond, cmd = COMBOS[wi % len(COMBOS)]
        if (net_cond, cmd) == prev_combo:
            consecutive += 1
        else:
            consecutive  = 0
            prev_combo   = (net_cond, cmd)

        row_start = wi * WINDOW_SIZE
        row_end   = row_start + WINDOW_SIZE

        # --- Push rows into encoders ---------------------------------------
        for dev_id in dev_ids:
            if dev_id not in encoders:
                continue
            _, df = devices[dev_id]
            for idx in range(row_start, row_end):
                encoders[dev_id].push(float(df.at[idx, "value"]))

        # --- Encode, quantize, decode each device -------------------------
        decoded_states: List[str]  = []
        confidences:    List[float] = []

        per_dev_state: Dict[int, str]   = {}
        per_dev_conf:  Dict[int, float] = {}

        for dev_id in dev_ids:
            if dev_id not in encoders or not encoders[dev_id].ready:
                per_dev_state[dev_id] = "NORMAL"
                per_dev_conf[dev_id]  = 0.5
                continue

            z_full   = encoders[dev_id].encode()
            if z_full is None:
                per_dev_state[dev_id] = "NORMAL"
                per_dev_conf[dev_id]  = 0.5
                continue

            indexed  = quantize(z_full, cmd)
            z_rec    = dequantize(indexed, LATENT_DIM)
            result   = encoders[dev_id].decode(z_rec)

            per_dev_state[dev_id] = result["clinical_state"]
            per_dev_conf[dev_id]  = float(result["confidence"])
            decoded_states.append(result["clinical_state"])
            confidences.append(float(result["confidence"]))

        # Patient-level health state: worst-case across all active devices
        health_state = _worst_case_label(decoded_states) if decoded_states else "NORMAL"
        mean_conf    = float(np.mean(confidences)) if confidences else 0.5

        # Window end timestamp (use last row from first device as reference)
        ts_ref_df = devices[dev_ids[0]][1]
        ts_end    = float(ts_ref_df.at[row_end - 1, "timestamp"])

        # --- Latency -------------------------------------------------------
        lat_ms = _sim_latency(cmd, net_cond, rng)

        # --- Command log (one per window) ----------------------------------
        command_rows.append({
            "timestamp":           ts_end,
            "network_state":       net_cond,
            "health_state":        health_state,
            "command":             cmd,
            "latency_ms":          round(lat_ms, 2),
            "consecutive_windows": consecutive,
        })

        # --- Telemetry (every TELE_STRIDE rows within the window) ----------
        for off in range(0, WINDOW_SIZE, TELE_STRIDE):
            row_idx = row_start + off
            ts_tele = float(ts_ref_df.at[row_idx, "timestamp"])
            telemetry_rows.append({
                "timestamp":                  ts_tele,
                "network_condition":          net_cond,
                "decode_confidence":          round(mean_conf, 4),
                "health_state":               health_state,
                "semantic_packets_in_window": WINDOW_SIZE * len(encoders),
            })

        # --- Enriched sender packets (encoded = True) ----------------------
        for dev_id, (dtype, df) in devices.items():
            for idx in range(row_start, row_end):
                row_dict            = df.iloc[idx].to_dict()
                row_dict["encoded"] = True
                enriched_pkts[dev_id].append(row_dict)

    # Tail rows (< one full window) — keep as non-encoded
    n_processed = n_windows * WINDOW_SIZE
    for dev_id, (dtype, df) in devices.items():
        if n_processed < len(df):
            for idx in range(n_processed, len(df)):
                row_dict            = df.iloc[idx].to_dict()
                row_dict["encoded"] = False
                enriched_pkts[dev_id].append(row_dict)

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------

    # Enriched sender logs
    for dev_id, (dtype, _) in devices.items():
        fname   = f"sender_log_dev{dev_id}_{dtype}.csv"
        out_path = os.path.join(args.output_dir, fname)
        out_df  = pd.DataFrame(enriched_pkts[dev_id])
        out_df.to_csv(out_path, index=False)
        print(f"[OUT] {fname}  ({len(out_df):,} rows)")

    # network_telemetry.csv
    tele_df = (
        pd.DataFrame(telemetry_rows)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )
    tele_path = os.path.join(args.output_dir, "network_telemetry.csv")
    tele_df.to_csv(tele_path, index=False)
    print(f"[OUT] network_telemetry.csv  ({len(tele_df):,} rows)")

    # command_log.csv
    cmd_df = (
        pd.DataFrame(command_rows)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    cmd_path = os.path.join(args.output_dir, "command_log.csv")
    cmd_df.to_csv(cmd_path, index=False)
    print(f"[OUT] command_log.csv  ({len(cmd_df):,} command events)")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n[SUMMARY]")
    print(f"  Windows processed : {n_windows}")
    print(f"  Combos covered    : {len(COMBOS)}  ({n_windows // len(COMBOS)} windows/cell minimum)")
    state_counts = pd.Series([r["health_state"] for r in command_rows]).value_counts()
    for lbl in LABEL_CLASSES:
        cnt = state_counts.get(lbl, 0)
        pct = 100.0 * cnt / max(1, len(command_rows))
        print(f"  health_state  {lbl:8s}: {cnt:4d}  ({pct:.1f}%)")

    print(f"\n[DONE] Run semantic_fidelity.py with:")
    print(f"  python3 scripts/evaluation/semantic_fidelity.py --logs-dir {args.output_dir}")


if __name__ == "__main__":
    main()
