"""
retrain_on_live_data.py
Retrain xgboost_network_model.pkl using all available live evaluation telemetry.
Training data is heuristic-labeled (matching choose_network_state from common.py).
Fixes the scale mismatch: old training data had avg_delay max 8ms, live data has ~100-400ms.
"""
import sys
import glob
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Heuristic ground-truth labeler (identical to common.py choose_network_state)
# ---------------------------------------------------------------------------
def heuristic(row):
    loss   = row["packet_loss_rate"]
    delay  = row["avg_delay"]      # already in ms (live telemetry stores ms)
    jitter = row["jitter"]         # already in ms
    if loss >= 0.08 or delay >= 130.0 or jitter >= 35.0:
        return "Critical"
    if loss >= 0.03 or delay >= 35.0 or jitter >= 8.0:
        return "Unstable"
    return "Stable"

# ---------------------------------------------------------------------------
# Collect files
# ---------------------------------------------------------------------------
globs = [
    str(PROJECT_ROOT / "outputs" / "**" / "network_telemetry.csv"),
    str(PROJECT_ROOT / "data"    / "logs" / "network_telemetry.csv"),
]
files = []
for g in globs:
    files.extend(glob.glob(g, recursive=True))

REQUIRED = ["packet_loss_rate", "avg_delay", "jitter",
            "throughput_bps", "bandwidth_usage_bps",
            "active_devices", "packets_per_window", "timestamp"]

# Files to hold out as the evaluation set (not included in training).
# We also explicitly hold out ANY closed-loop run as a broad rule, because 
# adaptive suppression (closed-loop) alters the network statistics itself, 
# corrupting the raw network baseline the model is supposed to learn from.
HOLDOUT_STEMS = {
    "baseline_natural_20260401_183044",   # Run 1 baseline
    "baseline_natural_20260415_051333",   # Sample holdout eval
    "closedloop_natural_20260401_184214", # Run 1 closedloop
    "closedloop_natural_20260415_054221", # Recent closedloop eval
}

frames = []
for f in files:
    try:
        df = pd.read_csv(f)
        if not all(c in df.columns for c in REQUIRED):
            continue
        if df["avg_delay"].max() < 20:          # skip tiny-scale synthetic data
            print(f"  [skip] {f}  (delay max {df.avg_delay.max():.1f}ms — synthetic scale)")
            continue
        stem = Path(f).parent.name
        
        # Hold out closed-loop runs universally because closed-loop suppresses traffic, 
        # which changes bandwidth, altering the raw problem distribution. We want the 
        # model trained purely on unadapted interference (baselines).
        if stem in HOLDOUT_STEMS or "closedloop" in stem:
            print(f"  [hold] {stem}  (held out for evaluation / closedloop corruption avoidance)")
            continue
        df["network_condition"] = df.apply(heuristic, axis=1)
        frames.append(df)
        print(f"  [load] {Path(f).parent.name}  rows={len(df):5d}  "
              f"delay_max={df.avg_delay.max():.0f}ms  "
              f"classes={df.network_condition.value_counts().to_dict()}")
    except Exception as e:
        print(f"  [err]  {f}: {e}")

if not frames:
    print("[FATAL] No valid training files found. Exiting.")
    sys.exit(1)

all_data = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows after merge: {len(all_data)}")
print("Class distribution:", all_data.network_condition.value_counts().to_dict())

# ---------------------------------------------------------------------------
# Feature engineering (identical to train_model.py + health_receiver.py)
# ---------------------------------------------------------------------------
df = all_data.sort_values("timestamp").reset_index(drop=True)
W, S = 10, 8

df["rolling_loss_mean"]        = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"]         = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"]      = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"]       = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"]  = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"]   = df["throughput_bps"].rolling(W).std()
df["loss_delta"]               = df["packet_loss_rate"].diff()
df["jitter_delta"]             = df["jitter"].diff()
df["delay_delta"]              = df["avg_delay"].diff()
df["loss_accel"]               = df["loss_delta"].diff()
df["loss_trend_3"]             = (df["packet_loss_rate"].rolling(3).mean()
                                  - df["packet_loss_rate"].rolling(8).mean())
df["throughput_trend_3"]       = (df["throughput_bps"].rolling(3).mean()
                                  - df["throughput_bps"].rolling(8).mean())
df["delay_trend_3"]            = (df["avg_delay"].rolling(3).mean()
                                  - df["avg_delay"].rolling(8).mean())

def _slope(v):
    n = len(v); xm = (n - 1) / 2.0; ym = v.mean()
    num = sum((i - xm) * (y - ym) for i, y in enumerate(v))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0

df["delay_slope"]   = df["avg_delay"].rolling(S).apply(_slope, raw=False)
df["loss_slope"]    = df["packet_loss_rate"].rolling(S).apply(_slope, raw=False)
df["loss_x_delay"]  = df["packet_loss_rate"] * df["avg_delay"]
df["loss_x_jitter"] = df["packet_loss_rate"] * df["jitter"]

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

df = df.dropna(subset=FEATURES)
print(f"After feature engineering + dropna: {len(df)} rows")

X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df["network_condition"].values)
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_).tolist())))

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("\nTraining XGBoost ...")
clf = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    eval_metric="mlogloss",
)
X_df = pd.DataFrame(X, columns=FEATURES)
clf.fit(X_df, y)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
preds_enc = clf.predict(X_df)
preds_str = le.inverse_transform(preds_enc)
y_str     = le.inverse_transform(y)

print(f"Train accuracy: {(preds_enc == y).mean():.4f}")
print()
print(classification_report(y_str, preds_str))
print("Prediction distribution on training data:")
print(pd.Series(preds_str).value_counts().to_dict())

# Quick sanity check on a fresh window (24 features in correct order)
# FEATURES order: bw, thr, loss, jitter, delay, roll_loss_mean, roll_loss_std,
#   roll_jit_mean, roll_delay_mean, roll_thr_mean, roll_thr_std,
#   loss_delta, jitter_delta, delay_delta, loss_accel,
#   loss_trend3, thr_trend3, delay_trend3, delay_slope, loss_slope,
#   active_dev, pkts_win, loss_x_delay, loss_x_jitter  (= 24 total)
stable_row   = [100000, 100000, 0.00,  3.0,  15.0,  0.00, 0.00,  3.0,  15.0, 100000, 0, 0.00, 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 4, 50,  0.00*15.0,  0.00*3.0]
unstable_row = [80000,  80000,  0.05, 12.0,  60.0,  0.04, 0.01, 10.0,  55.0,  80000, 0, 0.01, 1.0,  5.0,  0.0,  0.0, 0.0, 0.0, 0.5, 0.001, 4, 40, 0.05*60.0, 0.05*12.0]
critical_row = [40000,  40000,  0.15, 50.0, 200.0,  0.12, 0.02, 40.0, 180.0,  40000, 0, 0.02, 5.0, 20.0,  0.0,  0.0, 0.0, 0.0, 2.0, 0.002, 4, 20, 0.15*200.0, 0.15*50.0]

for name, row in [("stable", stable_row), ("unstable", unstable_row), ("critical", critical_row)]:
    row_df = pd.DataFrame([row], columns=FEATURES)
    raw = clf.predict(row_df)[0]
    print(f"  Sanity check [{name:8s}] → {le.inverse_transform([raw])[0]}")

# ---------------------------------------------------------------------------
# Save (overwrite xgboost_network_model.pkl)
# ---------------------------------------------------------------------------
out = MODELS_DIR / "xgboost_network_model.pkl"
joblib.dump(clf, out)
print(f"\n[OK] Saved → {out}")

# Write a metadata sidecar so health_receiver.py can verify this model was
# trained on ms-scale data and will not be misused with a seconds-scale model.
_meta = {
    "delay_scale": "ms",
    "jitter_scale": "ms",
    "trained_on": "live_telemetry",
    "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
    "model_type": "XGBoost",
}
with open(out.with_suffix(".json"), "w") as _mf:
    json.dump(_meta, _mf, indent=2)
print(f"[OK] Scale metadata → {out.with_suffix('.json')}")
