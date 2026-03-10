"""
train_model.py — Network Health Classifier
─────────────────────────────────────────────────────────────────
Changes from v1:
  - queue_length removed (was a label proxy — caused data leakage)
  - Time-based split instead of random shuffle (no rolling-window leakage)
  - TimeSeriesSplit cross-validation for honest multi-fold evaluation
  - Final model trained on all data after CV confirms it's solid
  - zero_division=0 suppresses warnings when a class is absent in a fold
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, ConfusionMatrixDisplay)
import joblib
import warnings
warnings.filterwarnings("ignore")

print("── Network Health Classifier Training ──────────────────────────")

# ── 1. Load ───────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("realistic_network_dataset.csv")
    print(f"[+] Loaded {len(df)} rows.")
except FileNotFoundError:
    print("[!] 'realistic_network_dataset.csv' not found. Run the generator first.")
    raise SystemExit(1)

print(f"    Class distribution:\n{df['network_condition'].value_counts().to_string()}\n")

# Check for required columns
REQUIRED_COLS = ["active_devices", "packets_per_window", "avg_delay"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. "
                     f"Re-run dynamic_traffic_generator.py to regenerate the dataset.")

# ── 2. Sort by time (critical for time-series CV) ─────────────────────────────
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 3. Rolling features ───────────────────────────────────────────────────────
print("[*] Generating rolling features (window=10) ...")
W = 10
df["rolling_loss_mean"]       = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"]        = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"]     = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"]      = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"]  = df["throughput_bps"].rolling(W).std()
# Trend features — direction and rate of change
df["loss_delta"]      = df["packet_loss_rate"].diff()          # rising or falling?
df["jitter_delta"]    = df["jitter"].diff()
df["delay_delta"]     = df["avg_delay"].diff()
df["loss_accel"]      = df["loss_delta"].diff()                # accelerating degradation?
df["loss_trend_3"]    = df["packet_loss_rate"].rolling(3).mean() - \
                        df["packet_loss_rate"].rolling(8).mean()  # short vs long trend
df["throughput_trend_3"] = (
    df["throughput_bps"].rolling(3).mean() -
    df["throughput_bps"].rolling(8).mean()
)
df["delay_trend_3"] = (
    df["avg_delay"].rolling(3).mean() -
    df["avg_delay"].rolling(8).mean()
)

df = df.dropna().reset_index(drop=True)
print(f"    Rows after NaN drop: {len(df)}")

# ── 4. Features ───────────────────────────────────────────────────────────────
# queue_length intentionally excluded — it's cumulative and leaks label info
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
    "loss_delta",
    "jitter_delta",
    "delay_delta",
    "loss_accel",
    "loss_trend_3",
    "throughput_trend_3",
    "delay_trend_3",
    "rolling_throughput_std",
    "active_devices",
    "packets_per_window",
]
print(f"    Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}\n")

X = df[FEATURE_COLS]
y = df["network_condition"]

# ── 5. TimeSeriesSplit Cross-Validation ───────────────────────────────────────
# Each fold trains on past data and tests on a future window — no lookahead.
print("── TimeSeriesSplit Cross-Validation (5 folds) ───────────────────")
tscv   = TimeSeriesSplit(n_splits=5)
scores = []
all_classes = sorted(y.unique())

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    macro_f1  = f1_score(y_te, y_pred, average="macro",    zero_division=0)
    per_class = f1_score(y_te, y_pred, average=None,
                         labels=all_classes, zero_division=0)
    scores.append(macro_f1)

    class_str = "  ".join(
        f"{c[:3]}={v:.2f}" for c, v in zip(all_classes, per_class)
    )
    print(f"  Fold {fold+1}  macro F1={macro_f1:.3f}   [{class_str}]"
          f"   train={len(train_idx)}  test={len(test_idx)}")

mean_f1 = np.mean(scores)
std_f1  = np.std(scores)
print(f"\n  CV Result: {mean_f1:.3f} ± {std_f1:.3f}  macro F1")

if mean_f1 >= 0.85:
    print("  ✅ Strong generalisation")
elif mean_f1 >= 0.75:
    print("  ⚠️  Acceptable — consider more data or wider profile overlap")
else:
    print("  ❌ Weak — boundaries likely too soft or dataset too small")

# ── 6. Final held-out evaluation (last 20% of time) ──────────────────────────
print("\n── Final Held-Out Test (last 20% of timeline) ───────────────────")
split    = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

present_classes = sorted(y_test.unique())
missing = set(all_classes) - set(present_classes)
if missing:
    print(f"  ⚠️  Classes absent in test window: {missing}")
    print(f"      (expected with short runs — CV above is the reliable metric)\n")

final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(classification_report(y_test, y_pred,
                             labels=present_classes,
                             zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=present_classes))
print(f"Labels: {present_classes}")

# ── 7. Feature importances ────────────────────────────────────────────────────
print("\nFeature importances (trained on full dataset):")
full_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
full_model.fit(X, y)
imp = (pd.Series(full_model.feature_importances_, index=FEATURE_COLS)
         .sort_values(ascending=False))
for feat, val in imp.items():
    bar = "█" * int(val * 80)
    print(f"  {feat:<30} {val:.4f}  {bar}")

# ── 8. Save the model trained on ALL data ────────────────────────────────────
# For deployment we want the most data possible, so save full_model.
model_filename = "robust_network_model.pkl"
joblib.dump(full_model, model_filename)
print(f"\n[✓] Model saved → '{model_filename}'  (trained on all {len(X)} samples)")
print(f"    CV macro F1: {mean_f1:.3f} ± {std_f1:.3f}")
