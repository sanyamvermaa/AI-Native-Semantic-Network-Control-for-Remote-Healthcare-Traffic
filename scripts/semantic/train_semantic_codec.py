#!/usr/bin/env python3
"""
Train per-device-type semantic encoder-decoder pairs using PyTorch.

Reads OU-generated CSV sender logs from data/logs/
  (columns: seq, timestamp, device_id, device_type, value, unit, label)
and trains a SemanticEncoder + SemanticDecoder for each device type.

Outputs (models/semantic/):
  enc_{device_type}.pt  — TorchScript encoder
  dec_{device_type}.pt  — TorchScript decoder
  metadata.json         — window sizes, latent dim, F1 scores, normal ranges

Usage (from project root, WSL):
  python3 scripts/semantic/train_semantic_codec.py

Requirements (WSL system Python):
  pip3 install torch scikit-learn --break-system-packages
  (numpy, pandas already installed)
"""

import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: add scripts/closed_loop so we can import DEVICE_PROFILES and
# CLINICAL_THRESHOLDS from health_sender.py (the authoritative source).
# Also needs common.py on the path (health_sender imports from common).
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CLOSED_LOOP  = os.path.join(_PROJECT_ROOT, "scripts", "closed_loop")

if _CLOSED_LOOP not in sys.path:
    sys.path.insert(0, _CLOSED_LOOP)

try:
    from health_sender import CLINICAL_THRESHOLDS, DEVICE_PROFILES  # type: ignore
except ImportError as _e:
    sys.exit(
        f"[ERROR] Cannot import DEVICE_PROFILES / CLINICAL_THRESHOLDS from "
        f"{_CLOSED_LOOP}/health_sender.py\n  Cause: {_e}"
    )

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split
except ImportError:
    sys.exit(
        "[ERROR] PyTorch not installed.\n"
        "  WSL:  pip3 install torch --break-system-packages"
    )

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    sys.exit(
        "[ERROR] scikit-learn not installed.\n"
        "  WSL:  pip3 install scikit-learn --break-system-packages"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Authoritative class→index mapping. index 0=NORMAL, 1=ALERT, 2=CRITICAL.
# All downstream code must use LABEL_INDEX / LABEL_CLASSES; never hardcode ints.
LABEL_CLASSES: List[str] = ["NORMAL", "ALERT", "CRITICAL"]
LABEL_INDEX:   Dict[str, int] = {c: i for i, c in enumerate(LABEL_CLASSES)}

LATENT_DIM: int = 16

WINDOW_SIZES: Dict[str, int] = {
    "ECG":           200,
    "BloodPressure": 200,
    "SpO2":          10,
    "Respiration":   10,
    "Temperature":   4,
}

DEVICE_TYPES: List[str] = ["ECG", "BloodPressure"]  # SpO2/Temp/Resp logs too sparse

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_device_logs(
    project_root: str, device_type: str
) -> List[Tuple[float, str]]:
    """
    Load all sender_log_dev*_{device_type}.csv files from data/logs/.
    Returns (value, label) pairs in filename-then-row order.
    """
    logs_dir = os.path.join(project_root, "data", "logs")
    rows: List[Tuple[float, str]] = []

    if not os.path.isdir(logs_dir):
        print(f"  [WARN] data/logs/ not found at {logs_dir}")
        return rows

    for fname in sorted(os.listdir(logs_dir)):
        if not (fname.startswith("sender_log_") and
                fname.endswith(f"_{device_type}.csv")):
            continue
        fpath = os.path.join(logs_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        v   = float(row["value"])
                        lbl = row["label"].strip().upper()
                        if lbl in LABEL_INDEX:
                            rows.append((v, lbl))
                    except (KeyError, ValueError):
                        continue
        except Exception as exc:
            print(f"  [WARN] Could not read {fpath}: {exc}")

    return rows

# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def build_windows(
    samples: List[Tuple[float, str]], window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of `window_size` over the sample list.
    Window label = worst clinical state (CRITICAL > ALERT > NORMAL).
    Returns:
        X  shape [N, window_size]  float32
        y  shape [N]               int64
    """
    n = len(samples)
    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    values = np.array([s[0] for s in samples], dtype=np.float32)
    labels = [s[1] for s in samples]

    n_windows = n - window_size + 1
    X = np.lib.stride_tricks.sliding_window_view(values, window_size).copy()  # [N, W]
    y = np.empty(n_windows, dtype=np.int64)

    for i in range(n_windows):
        win_labels = labels[i: i + window_size]
        if "CRITICAL" in win_labels:
            y[i] = LABEL_INDEX["CRITICAL"]
        elif "ALERT" in win_labels:
            y[i] = LABEL_INDEX["ALERT"]
        else:
            y[i] = LABEL_INDEX["NORMAL"]

    return X, y


def normalize_windows(X: np.ndarray, device_type: str) -> np.ndarray:
    """
    Normalise raw values to [0, 1] using the device's normal range from
    DEVICE_PROFILES.  Values outside the range intentionally land outside [0,1],
    signalling a clinical out-of-range reading to the encoder.
    """
    lo, hi = map(float, DEVICE_PROFILES[device_type]["normal"])
    span = (hi - lo) if hi != lo else 1.0
    return ((X - lo) / span).astype(np.float32)


def compute_window_features(X_norm: np.ndarray) -> np.ndarray:
    """
    Per-window summary statistics used as head_b reconstruction target.
    Returns shape [N, 4]: [min, max, mean, std]  (normalised space).
    """
    mins  = X_norm.min(axis=1, keepdims=True)
    maxs  = X_norm.max(axis=1, keepdims=True)
    means = X_norm.mean(axis=1, keepdims=True)
    stds  = X_norm.std(axis=1, keepdims=True)
    return np.concatenate([mins, maxs, means, stds], axis=1).astype(np.float32)

# ---------------------------------------------------------------------------
# Correlated burst simulation (training-data augmentation #10)
# ---------------------------------------------------------------------------

def apply_correlated_burst(
    X: np.ndarray,
    y: np.ndarray,
    device_type: str,
    burst_fraction: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select burst_fraction of NORMAL windows at random and fill them with
    uniformly-sampled values from the device's burst range (DEVICE_PROFILES),
    then relabel as CRITICAL.

    This teaches the encoder to distinguish smooth OU-process normal values
    from a multi-device sepsis event where all vitals jump simultaneously.
    """
    normal_idx = np.where(y == LABEL_INDEX["NORMAL"])[0]
    n_corrupt  = int(len(normal_idx) * burst_fraction)
    if n_corrupt == 0:
        return X, y

    burst_lo = float(DEVICE_PROFILES[device_type]["burst"][0])
    burst_hi = float(DEVICE_PROFILES[device_type]["burst"][1])

    chosen = np.random.choice(normal_idx, size=n_corrupt, replace=False)
    X_out, y_out = X.copy(), y.copy()
    for idx in chosen:
        X_out[idx] = np.random.uniform(
            burst_lo, burst_hi, size=X.shape[1]
        ).astype(np.float32)
        y_out[idx] = LABEL_INDEX["CRITICAL"]

    return X_out, y_out

# ---------------------------------------------------------------------------
# Class-balance augmentation using synthetic OU samples
# ---------------------------------------------------------------------------

def _label_by_thresholds(value: float, device_type: str) -> str:
    """Assign NORMAL / ALERT / CRITICAL to a raw sensor value."""
    t = CLINICAL_THRESHOLDS.get(device_type, {})
    crit_hi = t.get("critical");  warn_hi = t.get("warn")
    crit_lo = t.get("low_critical"); warn_lo = t.get("low_warn")
    if crit_hi is not None and value >= crit_hi:
        return "CRITICAL"
    if crit_lo is not None and value <= crit_lo:
        return "CRITICAL"
    if warn_hi is not None and value >= warn_hi:
        return "ALERT"
    if warn_lo is not None and value <= warn_lo:
        return "ALERT"
    return "NORMAL"


def _ou_values(mu: float, sigma: float, theta: float, n: int) -> List[float]:
    """Generate n OU process values around mu."""
    x = mu
    out: List[float] = []
    for _ in range(n):
        x += theta * (mu - x) + sigma * random.gauss(0, 1)
        out.append(x)
    return out


def balance_raw_samples(
    samples: List[Tuple[float, str]],
    device_type: str,
    min_fraction: float = 0.25,
) -> List[Tuple[float, str]]:
    """
    Ensure each class has at least min_fraction of the dominant class count.
    Synthetic samples are generated via OU process within the correct value
    range for each class and validated against CLINICAL_THRESHOLDS.

    min_fraction = 0.25 means every class gets at least 25% of the dominant.
    """
    profile    = DEVICE_PROFILES[device_type]
    thresholds = CLINICAL_THRESHOLDS.get(device_type, {})

    by_class: Dict[str, int] = {c: 0 for c in LABEL_CLASSES}
    for _, lbl in samples:
        if lbl in by_class:
            by_class[lbl] += 1

    dominant = max(by_class.values()) if by_class else 1
    target   = max(int(dominant * min_fraction), 200)

    # --- value range helpers ---
    n_lo, n_hi = float(profile["normal"][0]), float(profile["normal"][1])
    b_lo, b_hi = float(profile["burst"][0]),  float(profile["burst"][1])
    warn_hi  = thresholds.get("warn")
    crit_hi  = thresholds.get("critical")
    warn_lo  = thresholds.get("low_warn")
    crit_lo  = thresholds.get("low_critical")

    def _alert_mu() -> float:
        if warn_hi is not None and crit_hi is not None:
            return (warn_hi + crit_hi) / 2.0
        if warn_lo is not None and crit_lo is not None:
            return (warn_lo + crit_lo) / 2.0
        return (n_hi + b_lo) / 2.0

    ranges = {
        "NORMAL":   (n_lo, n_hi),
        "ALERT":    None,          # computed via _alert_mu
        "CRITICAL": (b_lo, b_hi),
    }

    added: List[Tuple[float, str]] = []
    for cls in LABEL_CLASSES:
        deficit = target - by_class[cls]
        if deficit <= 0:
            continue

        if cls == "ALERT":
            mu    = _alert_mu()
            sigma = abs(mu) * 0.04
        else:
            lo, hi = ranges[cls]  # type: ignore[misc]
            mu     = (lo + hi) / 2.0
            sigma  = (hi - lo) * 0.08

        theta  = 0.12
        budget = deficit * 10   # generate extra to account for label mismatch
        stream = _ou_values(mu, sigma, theta, budget)

        collected = 0
        for v in stream:
            if _label_by_thresholds(v, device_type) == cls:
                added.append((v, cls))
                collected += 1
                if collected >= deficit:
                    break

        print(f"  [BALANCE] {cls:<10}: added {collected:,} synthetic samples "
              f"(was {by_class[cls]:,}, target {target:,})")

    if not added:
        return samples

    combined = list(samples) + added
    random.shuffle(combined)
    return combined

# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    """
    Maps a normalised vital-sign window → 16-dim latent code.

    Input : x  [B, W]  float32  (pre-normalised to ~[0,1])
    Output: z  [B, 16] float32  (tanh-activated)

    Architecture:
        unsqueeze(1) → [B, 1, W]
        Conv1d(1→16, k=5, pad=2) → ReLU
        Conv1d(16→32, k=3, pad=1) → ReLU
        global average pool → [B, 32]
        Linear(32 → 16) → tanh
    """

    def __init__(self, window_size: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1,  16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.fc    = nn.Linear(32, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)                  # [B, 1, W]
        h = self.relu(self.conv1(h))         # [B, 16, W]
        h = self.relu(self.conv2(h))         # [B, 32, W]
        h = h.mean(dim=2)                    # global avg pool → [B, 32]
        return torch.tanh(self.fc(h))        # [B, latent_dim]


class SemanticDecoder(nn.Module):
    """
    Maps a latent code → clinical state + reconstructed window features.

    Input : z     [B, 16]
    Output: (out_a [B, 3], out_b [B, 4])
        out_a — softmax over NORMAL / ALERT / CRITICAL
        out_b — linear [min, max, mean, std] in normalised space

    Architecture:
        Linear(16→32) → ReLU → Linear(32→32) → ReLU
        head_a: Linear(32→3) → softmax
        head_b: Linear(32→4)
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.fc1    = nn.Linear(latent_dim, 32)
        self.fc2    = nn.Linear(32, 32)
        self.relu   = nn.ReLU()
        self.head_a = nn.Linear(32, len(LABEL_CLASSES))   # classification
        self.head_b = nn.Linear(32, 4)                    # feature reconstruction

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h     = self.relu(self.fc1(z))
        h     = self.relu(self.fc2(h))
        out_a = torch.softmax(self.head_a(h), dim=-1)
        out_b = self.head_b(h)
        return out_a, out_b


def apply_channel_truncation(z: torch.Tensor, training: bool) -> torch.Tensor:
    """
    During training: simulate channel truncation by zeroing the last k
    dimensions of the latent vector.  k is drawn uniformly from {0, 4, 8, 12}
    (equal probability 0.25 each — the four values partition the 16 dims evenly).
    The decoder must learn to classify reliably even with partial codes.

    No-op at inference (training=False).
    """
    if not training:
        return z
    k = random.choice([0, 4, 8, 12])
    if k == 0:
        return z
    mask = torch.ones_like(z)
    mask[:, -k:] = 0.0
    return z * mask

# ---------------------------------------------------------------------------
# Per-device training
# ---------------------------------------------------------------------------

def train_device_type(
    device_type: str,
    project_root: str,
    output_dir: str,
    pt_device: torch.device,
) -> Optional[Dict]:
    """
    Full training pipeline for one device type.
    Returns a metadata dict on success; None if data is insufficient.
    """
    print(f"\n{'=' * 60}")
    print(f"  Device type : {device_type}  (window_size={WINDOW_SIZES[device_type]})")
    print(f"{'=' * 60}")

    window_size = WINDOW_SIZES[device_type]

    # 1 — Load raw samples
    samples = load_device_logs(project_root, device_type)
    if not samples:
        print(f"  [SKIP] No sender logs found for {device_type}")
        return None
    print(f"  Loaded {len(samples):,} raw samples")

    # 2 — Build rolling windows; label by worst state in window
    X_raw, y = build_windows(samples, window_size)
    if len(X_raw) == 0:
        print(f"  [SKIP] Fewer samples ({len(samples)}) than window_size={window_size}")
        return None
    print(f"  Windows built: {len(X_raw):,}")

    # 3 — Normalise to [0, 1] relative to device's normal range
    X_norm = normalize_windows(X_raw, device_type)

    # 4 — Compute [min, max, mean, std] per window for head_b target
    feat_b = compute_window_features(X_norm)

    # Print class distribution
    for i, cls in enumerate(LABEL_CLASSES):
        cnt = int((y == i).sum())
        print(f"  {cls:10s}: {cnt:6,} windows ({100.0 * cnt / len(y):.1f}%)")

    # 6 — Build TensorDataset and 80/10/10 split
    X_t    = torch.tensor(X_norm, dtype=torch.float32)
    y_t    = torch.tensor(y,      dtype=torch.long)
    feat_t = torch.tensor(feat_b, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t, feat_t)

    n       = len(dataset)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)
    n_test  = n - n_train - n_val

    if n_train < 2 or n_test < 1:
        print(f"  [SKIP] Dataset too small ({n} windows) for 80/10/10 split")
        return None

    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=g
    )
    print(f"  Split: train={n_train:,}  val={n_val:,}  test={n_test:,}")

    # === BALANCED TRAINING BATCHES ===
    train_indices  = np.array(train_ds.indices)
    train_y_np     = y_t[train_indices].numpy()
    counts         = np.bincount(train_y_np, minlength=len(LABEL_CLASSES)).astype(np.float64)
    class_weights  = (len(train_y_np) / (len(LABEL_CLASSES) * np.maximum(counts, 1))).astype(np.float32)
    sample_weights = class_weights[train_y_np]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # 7 — Initialise models and optimiser
    enc = SemanticEncoder(window_size, LATENT_DIM).to(pt_device)
    dec = SemanticDecoder(LATENT_DIM).to(pt_device)

    params    = list(enc.parameters()) + list(dec.parameters())
    optimizer = optim.Adam(params, lr=5e-4)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    # 8 — Training loop (50 epochs)
    EPOCHS = 50
    for epoch in range(1, EPOCHS + 1):
        enc.train(); dec.train()
        epoch_loss = 0.0
        n_batches  = 0

        for X_b, y_b, fb_b in train_loader:
            X_b  = X_b.to(pt_device)
            y_b  = y_b.to(pt_device)
            fb_b = fb_b.to(pt_device)

            optimizer.zero_grad()
            z              = enc(X_b)
            z_trunc        = apply_channel_truncation(z, training=True)
            pred_a, pred_b = dec(z_trunc)

            # L = 1.0 * CE(head_a, label) + 0.1 * MSE(head_b, stats)
            loss = ce_loss(pred_a, y_b) + 0.1 * mse_loss(pred_b, fb_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if epoch % 10 == 0:
            enc.eval(); dec.eval()
            val_loss   = 0.0
            n_val_b    = 0
            with torch.no_grad():
                for X_b, y_b, fb_b in val_loader:
                    X_b, y_b, fb_b = (
                        X_b.to(pt_device),
                        y_b.to(pt_device),
                        fb_b.to(pt_device),
                    )
                    pred_a, pred_b = dec(enc(X_b))
                    val_loss += (
                        ce_loss(pred_a, y_b) + 0.1 * mse_loss(pred_b, fb_b)
                    ).item()
                    n_val_b += 1
            avg_train = epoch_loss / max(n_batches, 1)
            avg_val   = val_loss   / max(n_val_b,   1)
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}"
                f"  train_loss={avg_train:.4f}"
                f"  val_loss={avg_val:.4f}"
            )

    # 9 — Test evaluation
    enc.eval(); dec.eval()
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for X_b, y_b, _ in test_loader:
            pred_a, _ = dec(enc(X_b.to(pt_device)))
            all_pred.extend(pred_a.argmax(dim=1).cpu().tolist())
            all_true.extend(y_b.tolist())

    accuracy = float(accuracy_score(all_true, all_pred))
    f1_arr   = f1_score(
        all_true, all_pred,
        labels=list(range(len(LABEL_CLASSES))),
        average=None,
        zero_division=0,
    )
    f1_dict = {LABEL_CLASSES[i]: float(f1_arr[i]) for i in range(len(f1_arr))}
    test_f1_critical = f1_dict.get("CRITICAL", 0.0)

    print(f"  test_accuracy    : {accuracy:.4f}")
    print(f"  test_f1_CRITICAL : {test_f1_critical:.4f}")
    for cls, val in f1_dict.items():
        print(f"    F1[{cls}] = {val:.4f}")

    # 10 — Save TorchScript models via torch.jit.trace
    os.makedirs(output_dir, exist_ok=True)
    enc_path = os.path.join(output_dir, f"enc_{device_type}.pt")
    dec_path = os.path.join(output_dir, f"dec_{device_type}.pt")

    enc.eval(); dec.eval()
    with torch.no_grad():
        dummy_x   = torch.zeros(1, window_size, device=pt_device)
        enc_trace = torch.jit.trace(enc, dummy_x)
        enc_trace.save(enc_path)

        dummy_z   = torch.zeros(1, LATENT_DIM, device=pt_device)
        dec_trace = torch.jit.trace(dec, dummy_z)
        dec_trace.save(dec_path)

    print(f"  Saved: {enc_path}")
    print(f"  Saved: {dec_path}")

    return {
        "window_size":       window_size,
        "latent_dim":        LATENT_DIM,
        # Authoritative class→index mapping for all downstream consumers:
        # index 0 = NORMAL, 1 = ALERT, 2 = CRITICAL
        "label_classes":     LABEL_CLASSES,
        "normal_range":      list(DEVICE_PROFILES[device_type]["normal"]),
        "test_f1_per_class": f1_dict,
        "test_accuracy":     accuracy,
        "trained_at":        datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    output_dir = os.path.join(_PROJECT_ROOT, "models", "semantic")
    pt_device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[TRAIN] Semantic codec training")
    print(f"[TRAIN] Project root : {_PROJECT_ROOT}")
    print(f"[TRAIN] Output dir   : {output_dir}")
    print(f"[TRAIN] Torch device : {pt_device}")

    metadata: Dict                      = {}
    results:  List[Tuple[str, float, float]] = []

    for dtype in DEVICE_TYPES:
        result = train_device_type(dtype, _PROJECT_ROOT, output_dir, pt_device)
        if result is None:
            continue
        metadata[dtype] = result
        results.append((
            dtype,
            result["test_accuracy"],
            result["test_f1_per_class"].get("CRITICAL", 0.0),
        ))

    # Write metadata.json
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[TRAIN] metadata.json written: {meta_path}")

    # Summary table
    COL = 18
    print("\n" + "=" * 58)
    print(f"  {'device_type':<{COL}} {'test_accuracy':>13} {'f1_critical':>13}")
    print("-" * 58)
    failed: List[str] = []
    for dtype, acc, f1c in results:
        flag = "  ← BELOW 0.80" if f1c < 0.80 else ""
        print(f"  {dtype:<{COL}} {acc:>13.4f} {f1c:>13.4f}{flag}")
        if f1c < 0.80:
            failed.append(dtype)
    print("=" * 58)

    if failed:
        print(f"\n[FAIL] test_f1_critical < 0.80 for: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n[PASS] All device types meet test_f1_critical >= 0.80")


if __name__ == "__main__":
    main()
