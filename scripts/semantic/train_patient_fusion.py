#!/usr/bin/env python3
"""
Train a cross-modal patient-level fusion model.

Reads per-device sender logs from data/logs/, groups rows into 250ms time
buckets, encodes each device window into a 16-dim latent z via the trained
SemanticEncoder, and trains a Transformer-based PatientFusionEncoder that
predicts a joint patient state (NORMAL / ALERT / CRITICAL) and a scalar
deterioration probability across the whole cohort of devices.

Outputs:
  models/semantic/patient_fusion.pt  — TorchScript fusion model

Usage (WSL, project root):
  python3 scripts/semantic/train_patient_fusion.py
"""

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CLOSED_LOOP  = os.path.join(_PROJECT_ROOT, "scripts", "closed_loop")

if _CLOSED_LOOP not in sys.path:
    sys.path.insert(0, _CLOSED_LOOP)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from health_sender import DEVICE_PROFILES  # type: ignore
except ImportError as _e:
    sys.exit(f"[ERROR] Cannot import DEVICE_PROFILES: {_e}")

try:
    from semantic_encoder import SemanticEncoder  # type: ignore
except ImportError as _e:
    sys.exit(f"[ERROR] Cannot import SemanticEncoder: {_e}")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split
except ImportError:
    sys.exit(
        "[ERROR] PyTorch not installed.\n"
        "  WSL: pip3 install torch --break-system-packages"
    )

try:
    from sklearn.metrics import f1_score
except ImportError:
    sys.exit(
        "[ERROR] scikit-learn not installed.\n"
        "  WSL: pip3 install scikit-learn --break-system-packages"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_CLASSES: List[str] = ["NORMAL", "ALERT", "CRITICAL"]
LABEL_INDEX:   Dict[str, int] = {c: i for i, c in enumerate(LABEL_CLASSES)}

DEVICE_TYPE_INDEX: Dict[str, int] = {
    "ECG":           0,
    "SpO2":          1,
    "BloodPressure": 2,
    "Temperature":   3,
    "Respiration":   4,
}

DEVICE_TYPES   = ["ECG", "BloodPressure"]   # only trained device types
N_DEVICE_SLOTS = 8   # always pad to 8 — matches DEVICE_LAYOUT in common.py
LATENT_DIM     = 16
BUCKET_SEC     = 0.25
MIN_DEVICES    = 2   # minimum distinct device types per bucket

CORRELATED_BURST_PROB    = 0.20   # 20% of NORMAL buckets get synthetic burst
CORRELATED_BURST_DEVICES = {"ECG", "BloodPressure"}
CORRELATED_BURST_LABEL   = "ALERT"   # inject at ALERT level

EPOCHS    = 60
BATCH     = 32
LR        = 1e-3
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class PatientFusionEncoder(nn.Module):
    """
    Transformer-based cross-modal patient fusion encoder.

    Input:  [batch, N_DEVICE_SLOTS, 17]  (16 latent dims + device_type_index)
    Output: (joint_logits [batch, 3], deterioration_logit [batch, 1])
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable device-type embedding (8 types, 16 dims)
        self.type_embed = nn.Embedding(8, LATENT_DIM)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model       = LATENT_DIM,
            nhead         = 4,
            dim_feedforward = 32,
            dropout       = 0.1,
            batch_first   = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        # Deterioration head
        self.deterioration = nn.Linear(LATENT_DIM, 1)

    def forward(
        self,
        x: torch.Tensor,      # [batch, 8, 17]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z          = x[:, :, :LATENT_DIM]   # [batch, 8, 16]
        type_idx   = x[:, :, LATENT_DIM].long()  # [batch, 8]

        # Add device-type embedding to latent vectors
        z = z + self.type_embed(type_idx)        # [batch, 8, 16]

        # Transformer — produces [batch, 8, 16]
        h = self.transformer(z)

        # Mean pool across device dimension → [batch, 16]
        pooled = h.mean(dim=1)

        joint_logits   = self.classifier(pooled)    # [batch, 3]
        det_logit      = self.deterioration(pooled) # [batch, 1]

        return joint_logits, det_logit


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _bucket_key(ts: float) -> float:
    """Floor timestamp to nearest BUCKET_SEC boundary."""
    return math.floor(ts / BUCKET_SEC) * BUCKET_SEC


def load_sender_logs(project_root: str) -> Dict[float, Dict[int, Dict]]:
    """
    Read all sender_log_dev*.csv files and group into time buckets.

    Returns dict:  bucket_ts → {device_id → latest_row_dict}
    where each row_dict has keys: device_id, device_type, value, label, timestamp
    """
    logs_dir = os.path.join(project_root, "data", "logs")
    buckets: Dict[float, Dict[int, Dict]] = defaultdict(dict)

    if not os.path.isdir(logs_dir):
        print(f"[WARN] logs dir not found: {logs_dir}")
        return buckets

    for fname in sorted(os.listdir(logs_dir)):
        if not (fname.startswith("sender_log_dev") and fname.endswith(".csv")):
            continue
        fpath = os.path.join(logs_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        ts         = float(row["timestamp"])
                        device_id  = int(row["device_id"])
                        device_type = str(row["device_type"]).strip()
                        value      = float(row["value"])
                        label      = str(row["label"]).strip().upper()
                    except (KeyError, ValueError):
                        continue

                    bk = _bucket_key(ts)
                    existing = buckets[bk].get(device_id)
                    if existing is None or ts > existing["timestamp"]:
                        buckets[bk][device_id] = {
                            "device_id":   device_id,
                            "device_type": device_type,
                            "value":       value,
                            "label":       label,
                            "timestamp":   ts,
                        }
        except OSError:
            pass

    return buckets


def build_device_history(
    project_root: str,
) -> Dict[int, Tuple[str, List[Tuple[float, str]]]]:
    """
    Build per-device ordered (value, label) history for SemanticEncoder.push().
    Returns dict: device_id → (device_type, [(value, label), ...])
    Only includes device types in DEVICE_TYPES.
    """
    logs_dir = os.path.join(project_root, "data", "logs")
    history: Dict[int, Tuple[str, List[Tuple[float, str]]]] = {}

    if not os.path.isdir(logs_dir):
        return history

    for fname in sorted(os.listdir(logs_dir)):
        if not (fname.startswith("sender_log_dev") and fname.endswith(".csv")):
            continue
        fpath = os.path.join(logs_dir, fname)
        rows: List[tuple] = []   # (ts, device_id, device_type, value, label)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        ts         = float(row["timestamp"])
                        did        = int(row["device_id"])
                        dtype      = str(row["device_type"]).strip()
                        val        = float(row["value"])
                        lbl        = str(row["label"]).strip().upper()
                        rows.append((ts, did, dtype, val, lbl))
                    except (KeyError, ValueError):
                        continue
        except OSError:
            pass

        for ts, did, dtype, val, lbl in rows:
            if dtype not in DEVICE_TYPES:
                continue
            if did not in history:
                history[did] = (dtype, [])
            history[did][1].append((val, lbl))

    return history


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    bucket: Dict[int, Dict],
    dev_encoders: Dict[int, SemanticEncoder],
    slot_map: Dict[int, int],
) -> Optional[np.ndarray]:
    """
    Build the [N_DEVICE_SLOTS, 17] feature matrix for one bucket.

    Each device_id has its own pre-filled SemanticEncoder in dev_encoders.
    slot_map maps device_id → slot index (0-based, continuous).
    Returns None if no device in this bucket could be encoded.
    """
    matrix = np.zeros((N_DEVICE_SLOTS, LATENT_DIM + 1), dtype=np.float32)
    any_encoded = False

    for dev_id, row in bucket.items():
        enc = dev_encoders.get(dev_id)
        if enc is None:
            continue

        slot     = slot_map.get(dev_id, dev_id % N_DEVICE_SLOTS)
        dev_type = row["device_type"]

        # Push the bucket's current value into the device's rolling buffer
        enc.push(float(row["value"]))

        z = enc.encode()
        if z is None:
            type_idx = DEVICE_TYPE_INDEX.get(dev_type, 0)
            matrix[slot, LATENT_DIM] = float(type_idx)
            continue

        type_idx                  = DEVICE_TYPE_INDEX.get(dev_type, 0)
        matrix[slot, :LATENT_DIM] = np.array(z, dtype=np.float32)
        matrix[slot, LATENT_DIM]  = float(type_idx)
        any_encoded = True

    return matrix if any_encoded else None


# ---------------------------------------------------------------------------
# Dataset builder (chronological bucket-by-bucket replay)
# ---------------------------------------------------------------------------

def build_dataset(
    project_root: str,
    models_dir:   str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological replay approach:
      - Load per-device histories (value, label) in row order for DEVICE_TYPES only.
      - Push rows bucket-by-bucket (25 rows = 0.25 s at 100 Hz) into one encoder
        per device — no pre-push of the entire history.
      - Collect one feature matrix per bucket once all individual-device encoders
        reach window_size (first ~8 buckets are warm-up only).

    Returns:
      X       — [n_samples, N_DEVICE_SLOTS, 17]  float32
      y_class — [n_samples]                       int64
      y_det   — [n_samples]                       float32
    """
    from collections import Counter

    device_histories = build_device_history(project_root)
    # device_histories: {dev_id: (dtype, [(val, lbl), ...])}

    # Build empty encoder per device (nothing pre-pushed)
    dev_encoders: Dict[int, SemanticEncoder] = {}
    dev_types:    Dict[int, str]              = {}
    for dev_id, (dtype, _) in device_histories.items():
        try:
            enc = SemanticEncoder(dtype, models_dir)
        except Exception as exc:
            print(f"  [WARN] SemanticEncoder({dtype}) dev {dev_id}: {exc}")
            continue
        if enc._enc is None:
            continue
        dev_encoders[dev_id] = enc
        dev_types[dev_id]    = dtype

    if not dev_encoders:
        print("[DATA] No valid device encoders.")
        return (np.zeros((0, N_DEVICE_SLOTS, LATENT_DIM + 1), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32))

    # Each bucket = ROWS_PER_BUCKET rows of 100 Hz data = 0.25 s
    ROWS_PER_BUCKET = max(1, round(BUCKET_SEC / 0.010))  # = 25

    # Slot mapping: sorted device_ids → consecutive slots
    sorted_ids = sorted(dev_encoders.keys())
    slot_map   = {dev_id: min(i, N_DEVICE_SLOTS - 1)
                  for i, dev_id in enumerate(sorted_ids)}
    print(f"[DATA] Devices: {sorted_ids}   slot_map: {slot_map}")

    # Maximum history length across devices
    max_rows    = max(len(hist) for _, (_, hist) in device_histories.items()
                      if _ in dev_encoders.values() or True)  # capture all
    max_rows    = max(len(device_histories[d][1]) for d in dev_encoders)
    n_buckets   = max_rows // ROWS_PER_BUCKET
    print(f"[DATA] Rows per device: {max_rows}  buckets to scan: {n_buckets}")

    X_list:  List[np.ndarray] = []
    y_class: List[int]        = []
    y_det:   List[float]      = []

    for b in range(n_buckets):
        start = b * ROWS_PER_BUCKET
        end   = start + ROWS_PER_BUCKET

        matrix        = np.zeros((N_DEVICE_SLOTS, LATENT_DIM + 1), dtype=np.float32)
        labels:        List[str] = []
        present_types: set       = set()

        for dev_id, enc in dev_encoders.items():
            _, history = device_histories[dev_id]
            chunk = history[start:end]
            if not chunk:
                continue

            # Push this bucket's rows into the rolling buffer
            for val, _ in chunk:
                enc.push(val)

            if not enc.ready:
                continue

            z = enc.encode()
            if z is None:
                continue

            # Worst-case label for these rows
            bucket_lbls = [lbl for _, lbl in chunk]
            if "CRITICAL" in bucket_lbls:
                dev_label = "CRITICAL"
            elif "ALERT" in bucket_lbls:
                dev_label = "ALERT"
            else:
                dev_label = "NORMAL"

            dtype    = dev_types[dev_id]
            slot     = slot_map[dev_id]
            type_idx = DEVICE_TYPE_INDEX.get(dtype, 0)
            matrix[slot, :LATENT_DIM] = np.array(z, dtype=np.float32)
            matrix[slot, LATENT_DIM]  = float(type_idx)
            labels.append(dev_label)
            present_types.add(dtype)

        if len(present_types) < MIN_DEVICES:
            continue

        # Joint worst-case label
        if "CRITICAL" in labels:
            joint_label = "CRITICAL"
        elif "ALERT" in labels:
            joint_label = "ALERT"
        else:
            joint_label = "NORMAL"

        # Correlated burst injection for NORMAL buckets
        if joint_label == "NORMAL" and random.random() < CORRELATED_BURST_PROB:
            for dev_id, enc in dev_encoders.items():
                if dev_types[dev_id] in CORRELATED_BURST_DEVICES:
                    slot = slot_map[dev_id]
                    matrix[slot, :LATENT_DIM] += np.random.uniform(
                        0.3, 0.7, size=LATENT_DIM
                    ).astype(np.float32)
            joint_label = CORRELATED_BURST_LABEL

        X_list.append(matrix)
        y_class.append(LABEL_INDEX.get(joint_label, 0))
        y_det.append(1.0 if joint_label in ("ALERT", "CRITICAL") else 0.0)

    n = len(X_list)
    print(f"[DATA] Usable buckets: {n}")
    if n == 0:
        return (np.zeros((0, N_DEVICE_SLOTS, LATENT_DIM + 1), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32))

    X   = np.stack(X_list, axis=0)
    y_c = np.array(y_class, dtype=np.int64)
    y_d = np.array(y_det,   dtype=np.float32)

    dist = Counter(y_class)
    for i, cls in enumerate(LABEL_CLASSES):
        print(f"  {cls}: {dist[i]:,} ({100*dist[i]/n:.1f}%)")

    return X, y_c, y_d


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_sampler(
    ds: "torch.utils.data.Subset",
    y_all: "torch.Tensor",
) -> "WeightedRandomSampler":
    """Balanced sampler: equal expected class frequency per batch."""
    idx         = np.array(ds.indices)
    y_np        = y_all[idx].numpy()
    counts      = np.bincount(y_np, minlength=len(LABEL_CLASSES)).astype(np.float64)
    cls_w       = (len(y_np) / (len(LABEL_CLASSES) * np.maximum(counts, 1))).astype(np.float32)
    sample_w    = cls_w[y_np]
    return WeightedRandomSampler(sample_w.tolist(), num_samples=len(ds), replacement=True)

def train_fusion(project_root: str, models_dir: str, pt_device: torch.device) -> None:
    X, y_class, y_det = build_dataset(project_root, models_dir)

    if len(X) < 10:
        print("[WARN] Not enough buckets to train PatientFusionEncoder. "
              "Run data collection first (run_closed_loop_stress_auto.sh).")
        return

    # Reshape X for PyTorch: [n, 8, 17]
    X_t   = torch.tensor(X,       dtype=torch.float32)
    y_c_t = torch.tensor(y_class, dtype=torch.long)
    y_d_t = torch.tensor(y_det,   dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_t, y_c_t, y_d_t)

    n        = len(dataset)
    n_test   = max(1, int(n * TEST_SPLIT))
    n_val    = max(1, int(n * VAL_SPLIT))
    n_train  = n - n_val - n_test

    if n_train < 1:
        print(f"[WARN] Too few samples ({n}) for train/val/test split. Skipping.")
        return

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=_make_sampler(train_ds, y_c_t), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, drop_last=False)

    model = PatientFusionEncoder().to(pt_device)
    opt   = optim.Adam(model.parameters(), lr=LR)
    ce    = nn.CrossEntropyLoss()
    bce   = nn.BCEWithLogitsLoss()

    print(f"[TRAIN] PatientFusionEncoder: {n_train} train / {n_val} val / {n_test} test")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for Xb, yc, yd in train_loader:
            Xb = Xb.to(pt_device)
            yc = yc.to(pt_device)
            yd = yd.to(pt_device)

            logits, det = model(Xb)
            loss = ce(logits, yc) + bce(det, yd)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yc, yd in val_loader:
                    Xb = Xb.to(pt_device)
                    yc = yc.to(pt_device)
                    yd = yd.to(pt_device)
                    logits, det = model(Xb)
                    val_loss += (ce(logits, yc) + bce(det, yd)).item()
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={total_loss/len(train_loader):.4f}  "
                f"val_loss={val_loss/max(1,len(val_loader)):.4f}"
            )

    # --- Test evaluation ---
    model.eval()
    all_preds: List[int] = []
    all_true:  List[int] = []
    with torch.no_grad():
        for Xb, yc, _yd in test_loader:
            Xb = Xb.to(pt_device)
            logits, _ = model(Xb)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(yc.tolist())

    accuracy = sum(p == t for p, t in zip(all_preds, all_true)) / max(1, len(all_true))
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    f1_per = f1_score(all_true, all_preds, labels=list(range(len(LABEL_CLASSES))),
                      average=None, zero_division=0)
    print(f"\n[EVAL] PatientFusionEncoder — test_accuracy={accuracy:.3f}  f1_macro={f1:.3f}")
    for i, cls in enumerate(LABEL_CLASSES):
        print(f"  F1[{cls}] = {f1_per[i]:.4f}")

    if f1 >= 0.55:
        print("[PASS] f1_macro >= 0.55")
    else:
        print("[FAIL] f1_macro < 0.55 — consider retraining or extending data")

    # --- Save as TorchScript ---
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "patient_fusion.pt")
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, N_DEVICE_SLOTS, LATENT_DIM + 1, device=pt_device)
        traced = torch.jit.trace(model, dummy)
        traced.save(out_path)

    print(f"[SAVE] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    models_dir = os.path.join(_PROJECT_ROOT, "models", "semantic")
    pt_device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[TRAIN] PatientFusion training")
    print(f"[TRAIN] Project root : {_PROJECT_ROOT}")
    print(f"[TRAIN] Models dir   : {models_dir}")
    print(f"[TRAIN] Device       : {pt_device}")

    train_fusion(_PROJECT_ROOT, models_dir, pt_device)


if __name__ == "__main__":
    main()
