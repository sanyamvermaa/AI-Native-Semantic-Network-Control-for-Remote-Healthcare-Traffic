"""
train_model.py — Network Health Classifier
─────────────────────────────────────────────────────────────────
Changes from v2:
  - Added delay_slope and loss_slope (linear gradient over 8 windows)
    — helps separate Unstable (rising) from Critical (already high)
  - Added threshold baseline comparison for paper
  - Baseline shows model improvement over naive rule-based approach
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)
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

REQUIRED_COLS = ["active_devices", "packets_per_window", "avg_delay"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. "
                     f"Re-run dynamic_traffic_generator.py to regenerate the dataset.")

# ── 2. Sort by time ───────────────────────────────────────────────────────────
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 3. Feature Engineering ────────────────────────────────────────────────────
print("[*] Generating features ...")
W = 10

# Rolling statistics
df["rolling_loss_mean"]       = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"]        = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"]     = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"]      = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"]  = df["throughput_bps"].rolling(W).std()

# One-step deltas — rising or falling?
df["loss_delta"]   = df["packet_loss_rate"].diff()
df["jitter_delta"] = df["jitter"].diff()
df["delay_delta"]  = df["avg_delay"].diff()

# Acceleration — is degradation speeding up?
df["loss_accel"]   = df["loss_delta"].diff()

# Short vs long trend — is recent worse than the baseline?
df["loss_trend_3"]       = (df["packet_loss_rate"].rolling(3).mean()
                            - df["packet_loss_rate"].rolling(8).mean())
df["throughput_trend_3"] = (df["throughput_bps"].rolling(3).mean()
                            - df["throughput_bps"].rolling(8).mean())
df["delay_trend_3"]      = (df["avg_delay"].rolling(3).mean()
                            - df["avg_delay"].rolling(8).mean())

# ── Slope features (NEW) ──────────────────────────────────────────────────────
# Linear gradient over 8 windows — captures sustained rising/falling trends.
# Critical has already-high loss (flat or variable slope).
# Unstable has rising loss (positive slope) — this separates them.
SLOPE_W = 8
xs = np.arange(SLOPE_W)

df["delay_slope"] = df["avg_delay"].rolling(SLOPE_W).apply(
    lambda y: np.polyfit(xs, y, 1)[0], raw=True
)
df["loss_slope"] = df["packet_loss_rate"].rolling(SLOPE_W).apply(
    lambda y: np.polyfit(xs, y, 1)[0], raw=True
)

df = df.dropna().reset_index(drop=True)
print(f"    Rows after NaN drop: {len(df)}")

# ── 4. Feature columns ────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Raw telemetry
    "bandwidth_usage_bps",
    "throughput_bps",
    "packet_loss_rate",
    "jitter",
    "avg_delay",
    # Rolling statistics
    "rolling_loss_mean",
    "rolling_loss_std",
    "rolling_jitter_mean",
    "rolling_delay_mean",
    "rolling_throughput_mean",
    "rolling_throughput_std",
    # One-step deltas
    "loss_delta",
    "jitter_delta",
    "delay_delta",
    # Acceleration
    "loss_accel",
    # Short vs long trends
    "loss_trend_3",
    "throughput_trend_3",
    "delay_trend_3",
    # Slope features (NEW)
    "delay_slope",
    "loss_slope",
    # Multi-device diagnostics
    "active_devices",
    "packets_per_window",
]
print(f"    Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}\n")

X = df[FEATURE_COLS]
y = df["network_condition"]
all_classes = sorted(y.unique())

# ── 5. Time split ─────────────────────────────────────────────────────────────
split    = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── 6. Threshold Baseline ─────────────────────────────────────────────────────
# Simple rule-based classifier using only packet_loss_rate.
# This is the "naive" approach our model must beat to justify ML.
print("── Threshold Baseline (rule-based) ─────────────────────────────")

def threshold_predict(loss_rate):
    if loss_rate > 0.08:    return "Critical"
    elif loss_rate > 0.03:  return "Unstable"
    else:                   return "Stable"

baseline_pred = X_test["packet_loss_rate"].apply(threshold_predict)
baseline_f1   = f1_score(y_test, baseline_pred,
                         average="macro", zero_division=0)
baseline_acc  = (baseline_pred == y_test).mean()

print(f"  Rule: loss>8% → Critical | loss>3% → Unstable | else → Stable")
print(classification_report(y_test, baseline_pred,
                             labels=all_classes, zero_division=0))
print(f"  Baseline macro F1 : {baseline_f1:.3f}")
print(f"  Baseline accuracy : {baseline_acc:.3f}")

# ── 7. TimeSeriesSplit Cross-Validation ───────────────────────────────────────
print("── TimeSeriesSplit Cross-Validation (5 folds) ───────────────────")
tscv   = TimeSeriesSplit(n_splits=5)
scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    clf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    macro_f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
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
print(f"\n  CV Result : {mean_f1:.3f} ± {std_f1:.3f}  macro F1")
print(f"  Baseline  : {baseline_f1:.3f}  macro F1")
print(f"  Improvement over baseline: +{mean_f1 - baseline_f1:.3f}")

if mean_f1 >= 0.85:
    print("  ✅ Strong generalisation")
elif mean_f1 >= 0.75:
    print("  ⚠️  Acceptable — consider more data or wider profile overlap")
else:
    print("  ❌ Weak — boundaries too soft or dataset too small")

# ── 8. Final held-out evaluation ──────────────────────────────────────────────
print("\n── Final Held-Out Test (last 20% of timeline) ───────────────────")

present_classes = sorted(y_test.unique())
missing_cls = set(all_classes) - set(present_classes)
if missing_cls:
    print(f"  ⚠️  Classes absent in test window: {missing_cls}")
    print(f"      CV above is the reliable metric\n")

final_model = RandomForestClassifier(
    n_estimators=100, random_state=42,
    n_jobs=-1, class_weight="balanced"
)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(classification_report(y_test, y_pred,
                             labels=present_classes, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=present_classes))
print(f"Labels: {present_classes}")

held_out_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(f"\n  Held-out macro F1 : {held_out_f1:.3f}")
print(f"  Baseline macro F1 : {baseline_f1:.3f}")
print(f"  Model improvement : +{held_out_f1 - baseline_f1:.3f}")

# ── 9. Feature importances ────────────────────────────────────────────────────
print("\n── Feature Importances (full dataset) ───────────────────────────")
full_model = RandomForestClassifier(
    n_estimators=100, random_state=42,
    n_jobs=-1, class_weight="balanced"
)
full_model.fit(X, y)
imp = (pd.Series(full_model.feature_importances_, index=FEATURE_COLS)
         .sort_values(ascending=False))
for feat, val in imp.items():
    bar = "█" * int(val * 80)
    print(f"  {feat:<30} {val:.4f}  {bar}")

# ── 10. Save ──────────────────────────────────────────────────────────────────
model_filename = "robust_network_model.pkl"
joblib.dump(full_model, model_filename)
print(f"\n[✓] Model saved → '{model_filename}'  (trained on all {len(X)} samples)")
print(f"    CV macro F1       : {mean_f1:.3f} ± {std_f1:.3f}")
print(f"    Held-out macro F1 : {held_out_f1:.3f}")
print(f"    Baseline macro F1 : {baseline_f1:.3f}")
print(f"    Improvement       : +{mean_f1 - baseline_f1:.3f} over threshold rule")