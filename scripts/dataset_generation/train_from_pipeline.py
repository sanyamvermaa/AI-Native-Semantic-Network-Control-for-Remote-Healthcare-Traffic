"""
train_from_pipeline.py
======================
Retrain the XGBoost network classifier using all available baseline telemetry,
with special emphasis on the freshly-generated pipeline run.

Key improvements over retrain_on_live_data.py:
  - Accepts --new-run-dir so it knows which run was JUST collected and prints
    per-class row counts specifically for that run.
  - Uses a SOFT heuristic labeler: the thresholds are jittered randomly per row
    to blur class boundaries and teach the model that real life is ambiguous.
  - Prints a rich final summary table: new run rows, total corpus rows, class
    distribution (this run vs all), and sanity-check predictions.
  - Saves label_classes.json so health_receiver.py can decode numeric predictions.
"""

import argparse
import datetime
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Soft/fuzzy heuristic labeler
#
# Instead of hard thresholds, each row gets labelled with thresholds that are
# perturbed by ±FUZZ_PCT% Gaussian noise.  This means samples near the boundary
# get a label drawn from a distribution rather than a hard cutoff, which teaches
# the model to be uncertain near the edge — much closer to real life.
# ---------------------------------------------------------------------------
FUZZ_PCT = 0.12   # ±12% Gaussian noise on each threshold

_RNG = random.Random(42)

def _gauss_threshold(base: float) -> float:
    """Return base ± FUZZ_PCT*base with Box-Muller noise."""
    u1 = _RNG.random() + 1e-12
    u2 = _RNG.random() + 1e-12
    z  = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return max(0.0, base * (1.0 + FUZZ_PCT * z))

def soft_heuristic(row) -> str:
    """
    Fuzzy version of common.py#choose_network_state.
    Each invocation draws slightly different thresholds so boundary samples
    receive labels that reflect real-world ambiguity.
    """
    loss   = row["packet_loss_rate"]
    delay  = row["avg_delay"]    # ms
    jitter = row["jitter"]       # ms

    # Critical thresholds: loss≥8%, delay≥130 ms, jitter≥35 ms
    t_loss_crit   = _gauss_threshold(0.08)
    t_delay_crit  = _gauss_threshold(130.0)
    t_jitter_crit = _gauss_threshold(35.0)

    if loss >= t_loss_crit or delay >= t_delay_crit or jitter >= t_jitter_crit:
        return "Critical"

    # Unstable thresholds: loss≥3%, delay≥35 ms, jitter≥8 ms
    t_loss_unst   = _gauss_threshold(0.03)
    t_delay_unst  = _gauss_threshold(35.0)
    t_jitter_unst = _gauss_threshold(8.0)

    if loss >= t_loss_unst or delay >= t_delay_unst or jitter >= t_jitter_unst:
        return "Unstable"

    return "Stable"


# ---------------------------------------------------------------------------
# Feature engineering — must be identical to retrain_on_live_data.py /
# health_receiver.py so the model sees consistent features at inference time.
# ---------------------------------------------------------------------------
FEATURES = [
    "bandwidth_usage_bps", "throughput_bps", "packet_loss_rate", "jitter", "avg_delay",
    "rolling_loss_mean", "rolling_loss_std", "rolling_jitter_mean", "rolling_delay_mean",
    "rolling_throughput_mean", "rolling_throughput_std",
    "loss_delta", "jitter_delta", "delay_delta", "loss_accel",
    "loss_trend_3", "throughput_trend_3", "delay_trend_3",
    "delay_slope", "loss_slope",
    "active_devices", "packets_per_window",
    "loss_x_delay", "loss_x_jitter",
]

REQUIRED_RAW = ["packet_loss_rate", "avg_delay", "jitter",
                "throughput_bps", "bandwidth_usage_bps",
                "active_devices", "packets_per_window", "timestamp"]

W, S = 10, 8

def _slope(v: np.ndarray) -> float:
    n = len(v)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2.0
    ym = v.mean()
    num = sum((i - xm) * (y - ym) for i, y in enumerate(v))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["rolling_loss_mean"]       = df["packet_loss_rate"].rolling(W).mean()
    df["rolling_loss_std"]        = df["packet_loss_rate"].rolling(W).std()
    df["rolling_jitter_mean"]     = df["jitter"].rolling(W).mean()
    df["rolling_delay_mean"]      = df["avg_delay"].rolling(W).mean()
    df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
    df["rolling_throughput_std"]  = df["throughput_bps"].rolling(W).std()
    df["loss_delta"]              = df["packet_loss_rate"].diff()
    df["jitter_delta"]            = df["jitter"].diff()
    df["delay_delta"]             = df["avg_delay"].diff()
    df["loss_accel"]              = df["loss_delta"].diff()
    df["loss_trend_3"]            = (df["packet_loss_rate"].rolling(3).mean()
                                     - df["packet_loss_rate"].rolling(8).mean())
    df["throughput_trend_3"]      = (df["throughput_bps"].rolling(3).mean()
                                     - df["throughput_bps"].rolling(8).mean())
    df["delay_trend_3"]           = (df["avg_delay"].rolling(3).mean()
                                     - df["avg_delay"].rolling(8).mean())
    df["delay_slope"]             = df["avg_delay"].rolling(S).apply(
                                        lambda v: _slope(v.values), raw=False)
    df["loss_slope"]              = df["packet_loss_rate"].rolling(S).apply(
                                        lambda v: _slope(v.values), raw=False)
    df["loss_x_delay"]            = df["packet_loss_rate"] * df["avg_delay"]
    df["loss_x_jitter"]           = df["packet_loss_rate"] * df["jitter"]

    return df.dropna(subset=FEATURES)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
HOLDOUT_STEMS = {
    "baseline_natural_20260401_183044",   # original holdout run 1
    "baseline_natural_20260415_051333",   # holdout eval
    "closedloop_natural_20260401_184214", # closed-loop (corrupts distribution)
    "closedloop_natural_20260415_054221",
}

def load_telemetry_file(path: str) -> pd.DataFrame | None:
    """Return a cleaned DataFrame or None if the file should be skipped."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  [err]  {path}: {exc}")
        return None

    if not all(c in df.columns for c in REQUIRED_RAW):
        print(f"  [skip] {Path(path).parent.name}  (missing required columns)")
        return None

    if df["avg_delay"].max() < 20:
        print(f"  [skip] {Path(path).parent.name}  "
              f"(delay max {df['avg_delay'].max():.1f}ms — synthetic/small-scale)")
        return None

    stem = Path(path).parent.name
    if stem in HOLDOUT_STEMS or "closedloop" in stem:
        print(f"  [hold] {stem}  (held out — closed-loop or designated eval set)")
        return None

    return df


def print_banner(title: str, char: str = "═", width: int = 64) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_class_table(counts: dict, title: str = "Class distribution") -> None:
    total = sum(counts.values()) or 1
    print(f"\n  {title}:")
    print(f"  {'Class':<12} {'Rows':>7}  {'%':>6}")
    print(f"  {'─'*12} {'─'*7}  {'─'*6}")
    for cls in ("Stable", "Unstable", "Critical"):
        n = counts.get(cls, 0)
        print(f"  {cls:<12} {n:>7,}  {100*n/total:>5.1f}%")
    print(f"  {'─'*12} {'─'*7}  {'─'*6}")
    print(f"  {'TOTAL':<12} {sum(counts.values()):>7,}  100.0%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train XGBoost from pipeline telemetry")
    ap.add_argument("--new-run-dir", type=str, default=None,
                    help="The RUN_DIR that was just generated (for per-run summary).")
    args = ap.parse_args()

    print_banner("train_from_pipeline.py — Fuzzy Pipeline Retrainer")

    # -----------------------------------------------------------------------
    # 1. Discover all telemetry files
    # -----------------------------------------------------------------------
    globs = [
        str(PROJECT_ROOT / "outputs" / "**" / "network_telemetry.csv"),
        str(PROJECT_ROOT / "data"    / "logs" / "network_telemetry.csv"),
    ]
    all_paths = []
    for g in globs:
        all_paths.extend(glob.glob(g, recursive=True))

    new_run_path = None
    if args.new_run_dir:
        candidate = Path(args.new_run_dir) / "network_telemetry.csv"
        if candidate.exists():
            new_run_path = str(candidate)
            if new_run_path not in all_paths:
                all_paths.append(new_run_path)

    print(f"\nDiscovered {len(all_paths)} telemetry file(s).")

    # -----------------------------------------------------------------------
    # 2. Load and label each file with the SOFT heuristic
    # -----------------------------------------------------------------------
    print("\nLoading files...")
    frames = []
    new_run_df = None

    for path in all_paths:
        df = load_telemetry_file(path)
        if df is None:
            continue

        # Apply soft/fuzzy label
        # Re-seed the jitter RNG per file so repeated runs give consistent
        # labels per file but different files may still differ slightly.
        _RNG.seed(hash(Path(path).parent.name) & 0xFFFF_FFFF)
        df["network_condition"] = df.apply(soft_heuristic, axis=1)
        frames.append(df)

        stem = Path(path).parent.name
        cls_dist = df["network_condition"].value_counts().to_dict()
        print(f"  [load] {stem:<45}  rows={len(df):5,}  "
              f"delay_max={df['avg_delay'].max():6.0f}ms  "
              f"classes={cls_dist}")

        if path == new_run_path:
            new_run_df = df

    if not frames:
        print("\n[FATAL] No valid training files found. Exiting.")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows merged: {len(all_data):,}")

    # -----------------------------------------------------------------------
    # 3. This-run summary
    # -----------------------------------------------------------------------
    print_banner("New-Run Telemetry Summary")
    if new_run_df is not None:
        new_run_counts = new_run_df["network_condition"].value_counts().to_dict()
        print(f"\n  Run directory : {args.new_run_dir}")
        print(f"  Raw rows      : {len(new_run_df):,}")
        print_class_table(new_run_counts, "Class breakdown — this run")
    else:
        print("  (no new run directory provided / file not found)")

    # -----------------------------------------------------------------------
    # 4. Feature engineering
    # -----------------------------------------------------------------------
    print_banner("Feature Engineering")
    engineered = engineer_features(all_data)
    print(f"  Rows after rolling features + dropna: {len(engineered):,}")

    X = engineered[FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(engineered["network_condition"].values)
    print(f"  Label encoding: { dict(zip(le.classes_, le.transform(le.classes_).tolist())) }")

    # Save label classes so health_receiver.py can decode
    label_classes_path = MODELS_DIR / "label_classes.json"
    with open(label_classes_path, "w") as lf:
        json.dump(list(le.classes_), lf)
    print(f"  Saved label_classes.json → {label_classes_path}")

    # -----------------------------------------------------------------------
    # 5. Train XGBoost
    # -----------------------------------------------------------------------
    print_banner("Training XGBoost Classifier")
    start_t = time.perf_counter()

    X_df = pd.DataFrame(X, columns=FEATURES)
    # Split to prevent overfitting on the noise and validate true generalization
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)

    clf = XGBClassifier(
        n_estimators=120,    # Reduced to prevent memorizing noise
        max_depth=4,         # Reduced to enforce generalization
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,  # Increased to prevent fitting isolated noisy samples
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    clf.fit(X_train, y_train)

    elapsed = time.perf_counter() - start_t
    print(f"  Training time (80% split): {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # 6. Evaluation
    # -----------------------------------------------------------------------
    print_banner("Model Evaluation")
    
    preds_train = clf.predict(X_train)
    train_acc = (preds_train == y_train).mean()
    
    preds_test = clf.predict(X_test)
    test_acc = (preds_test == y_test).mean()

    print(f"  Train accuracy : {train_acc:.4f}  (80% split)")
    print(f"  Test accuracy  : {test_acc:.4f}  (20% held-out)")
    print()
    
    preds_str_test = le.inverse_transform(preds_test)
    y_str_test     = le.inverse_transform(y_test)
    print("  Test Set Classification Report:")
    print(classification_report(y_str_test, preds_str_test, target_names=["Critical","Stable","Unstable"]))

    pred_counts = pd.Series(preds_str_test).value_counts().to_dict()
    print_class_table(pred_counts, "Prediction distribution on test set")

    # -----------------------------------------------------------------------
    # 7. Full corpus label summary
    # -----------------------------------------------------------------------
    print_banner("Full Training Corpus — Label Summary")
    corpus_counts = pd.Series(y_str).value_counts().to_dict()
    print_class_table(corpus_counts, "Ground-truth label distribution (all files)")

    # Per-file breakdown
    print(f"\n  {'File (stem)':<45}  {'Stable':>8}  {'Unstable':>9}  {'Critical':>9}  {'Total':>7}")
    print(f"  {'─'*45}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*7}")
    for i, (path, df) in enumerate(zip(all_paths, frames)):
        stem = Path(path).parent.name
        vc   = df["network_condition"].value_counts().to_dict()
        st   = vc.get("Stable",   0)
        un   = vc.get("Unstable", 0)
        cr   = vc.get("Critical", 0)
        tot  = st + un + cr
        marker = " ← NEW" if path == new_run_path else ""
        print(f"  {stem:<45}  {st:>8,}  {un:>9,}  {cr:>9,}  {tot:>7,}{marker}")

    # -----------------------------------------------------------------------
    # 8. Sanity checks
    # -----------------------------------------------------------------------
    print_banner("Sanity-Check Predictions")

    # fmt: bandwidth, throughput, loss, jitter, delay,
    #      roll_loss_mean, roll_loss_std, roll_jitter_mean, roll_delay_mean,
    #      roll_thr_mean, roll_thr_std,
    #      loss_delta, jitter_delta, delay_delta, loss_accel,
    #      loss_trend3, thr_trend3, delay_trend3, delay_slope, loss_slope,
    #      active_devices, pkts_win, loss_x_delay, loss_x_jitter
    probes = {
        "Deep-Stable   (loss=0.5%, delay=12ms, jitter=2ms)": [
            100_000, 100_000, 0.005, 2.0, 12.0,
            0.005, 0.001, 2.0, 12.0, 100_000, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            4, 50, 0.005*12.0, 0.005*2.0,
        ],
        "Edge-Stable   (loss=2.5%, delay=32ms, jitter=7ms)": [
            85_000, 85_000, 0.025, 7.0, 32.0,
            0.022, 0.003, 6.5, 30.0, 85_000, 500,
            0.001, 0.3, 1.0, 0.0, 0.002, 0.0, 0.5, 0.05, 0.0001,
            4, 44, 0.025*32.0, 0.025*7.0,
        ],
        "Mid-Unstable  (loss=5%, delay=60ms, jitter=13ms)": [
            75_000, 75_000, 0.05, 13.0, 60.0,
            0.045, 0.008, 11.0, 55.0, 75_000, 800,
            0.005, 0.8, 4.0, 0.001, 0.003, 0.0, 1.5, 0.3, 0.001,
            4, 40, 0.05*60.0, 0.05*13.0,
        ],
        "Edge-Critical (loss=8%, delay=135ms, jitter=36ms)": [
            55_000, 55_000, 0.08, 36.0, 135.0,
            0.075, 0.010, 32.0, 128.0, 55_000, 1200,
            0.010, 2.0, 8.0, 0.002, 0.005, 0.0, 3.0, 1.0, 0.002,
            4, 30, 0.08*135.0, 0.08*36.0,
        ],
        "Deep-Critical (loss=18%, delay=250ms, jitter=65ms)": [
            35_000, 35_000, 0.18, 65.0, 250.0,
            0.160, 0.020, 58.0, 230.0, 35_000, 0,
            0.015, 4.0, 18.0, 0.003, 0.010, 0.0, 8.0, 2.5, 0.003,
            4, 18, 0.18*250.0, 0.18*65.0,
        ],
    }

    all_pass = True
    expected = {
        "Deep-Stable": "Stable",
        "Mid-Unstable": "Unstable",
        "Deep-Critical": "Critical",
    }
    print(f"\n  {'Probe':<50}  {'Prediction':>10}  {'Conf%':>6}")
    print(f"  {'─'*50}  {'─'*10}  {'─'*6}")
    for label, row in probes.items():
        row_df = pd.DataFrame([row], columns=FEATURES)
        probs  = clf.predict_proba(row_df)[0]
        best_i = int(np.argmax(probs))
        pred   = le.inverse_transform([best_i])[0]
        conf   = 100.0 * probs[best_i]
        key    = label.split("(")[0].strip().split("-")[1].strip()
        exp    = expected.get(key[:10], None)
        flag   = "" if exp is None or pred == exp else " ← UNEXPECTED"
        if flag:
            all_pass = False
        print(f"  {label:<50}  {pred:>10}  {conf:>5.1f}%{flag}")

    if all_pass:
        print(f"\n  ✓ All key sanity checks passed.")
    else:
        print(f"\n  ✗ Some sanity checks flagged unexpected predictions — review.")

    # -----------------------------------------------------------------------
    # 9. Save model + metadata
    # -----------------------------------------------------------------------
    print_banner("Saving Model")
    out = MODELS_DIR / "xgboost_network_model.pkl"
    joblib.dump(clf, out)
    print(f"  Saved → {out}")

    meta = {
        "delay_scale":   "ms",
        "jitter_scale":  "ms",
        "trained_on":    "live_telemetry_fuzzy_pipeline",
        "trained_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_type":    "XGBoost",
        "train_rows":    len(engineered),
        "train_accuracy": round(float(train_acc), 6),
        "corpus_files":  len(frames),
        "label_classes": list(le.classes_),
        "fuzz_pct":      FUZZ_PCT,
    }
    with open(out.with_suffix(".json"), "w") as mf:
        json.dump(meta, mf, indent=2)
    print(f"  Scale metadata → {out.with_suffix('.json')}")

    print_banner("Complete", char="═")
    print(f"  Model ready at: {out}")
    print(f"  Trained on {len(engineered):,} feature rows from {len(frames)} file(s).")
    print()


if __name__ == "__main__":
    main()
