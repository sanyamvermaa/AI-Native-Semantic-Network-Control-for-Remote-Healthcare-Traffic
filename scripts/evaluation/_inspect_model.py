"""
Check what the model actually learned from the training data.
Sample 20 rows per class directly from the (fixed) training data and see
what the model predicts on those.  Also print actual per-class feature stats.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "scripts" / "closed_loop"))

import joblib

MODEL_PATH = P / "models" / "best_network_model.pkl"
CSV_PATH   = P / "data" / "datasets" / "realistic_network_dataset.csv"

model = joblib.load(MODEL_PATH)
feat_names = list(model.feature_names_in_)
classes    = list(model.classes_)

df = pd.read_csv(CSV_PATH)

# Apply the same fix as train_model.py
df["jitter"]    *= 1000.0
df["avg_delay"] *= 1000.0
df = df.sort_values("timestamp").reset_index(drop=True)

W = 10
df["rolling_loss_mean"]       = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"]        = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"]     = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"]      = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"]  = df["throughput_bps"].rolling(W).std()
df["loss_delta"]   = df["packet_loss_rate"].diff()
df["jitter_delta"] = df["jitter"].diff()
df["delay_delta"]  = df["avg_delay"].diff()
df["loss_accel"]   = df["loss_delta"].diff()
df["loss_trend_3"]       = df["packet_loss_rate"].rolling(3).mean() - df["packet_loss_rate"].rolling(8).mean()
df["throughput_trend_3"] = df["throughput_bps"].rolling(3).mean() - df["throughput_bps"].rolling(8).mean()
df["delay_trend_3"]      = df["avg_delay"].rolling(3).mean() - df["avg_delay"].rolling(8).mean()
xs = np.arange(8)
df["delay_slope"] = df["avg_delay"].rolling(8).apply(lambda y: np.polyfit(xs, y, 1)[0], raw=True)
df["loss_slope"]  = df["packet_loss_rate"].rolling(8).apply(lambda y: np.polyfit(xs, y, 1)[0], raw=True)
df["loss_x_delay"]  = df["packet_loss_rate"] * df["avg_delay"]
df["loss_x_jitter"] = df["packet_loss_rate"] * df["jitter"]
df = df.dropna().reset_index(drop=True)

print(f"Training data after fix: {len(df)} rows")
print()

# Per-class actual feature ranges (the key ones)
print("=== Per-class ranges (ms after *1000) ===")
for cls in ["Stable", "Unstable", "Critical"]:
    sub = df[df["network_condition"] == cls]
    print(f"\n  {cls} ({len(sub)} rows):")
    print(f"    packet_loss_rate : {sub['packet_loss_rate'].min():.4f} – {sub['packet_loss_rate'].max():.4f}  (mean={sub['packet_loss_rate'].mean():.4f})")
    print(f"    jitter           : {sub['jitter'].min():.1f} – {sub['jitter'].max():.1f} ms  (mean={sub['jitter'].mean():.1f})")
    print(f"    avg_delay        : {sub['avg_delay'].min():.1f} – {sub['avg_delay'].max():.1f} ms  (mean={sub['avg_delay'].mean():.1f})")

# Model accuracy on training data
X = df[feat_names]
y = df["network_condition"]
preds = model.predict(X)
acc = (preds == y).mean()

from collections import Counter
correct_per_class = {}
for cls in ["Stable", "Unstable", "Critical"]:
    mask = y == cls
    correct_per_class[cls] = (preds[mask] == y[mask]).mean()

print(f"\n=== Model accuracy on training data ===")
print(f"  Overall: {acc:.1%}")
for cls, acc_c in correct_per_class.items():
    print(f"  {cls:10}: {acc_c:.1%}")

# Check what the model predicts for my pre-flight test cases
print("\n=== Testing specific production scenarios ===")
# Build proper feature vectors from actual training-data-style rows
# Sample steady stable conditions (low loss, low delay, low jitter)
cases = [
    ("live_Stable",   0.00, 12.0,  2.0),
    ("live_Unstable", 0.04, 60.0, 12.0),
    ("live_Critical", 0.12, 150.0, 45.0),
    # Values that match the training Critical range
    ("train_Unstable",0.05, 50.0,  8.0),
    ("train_Critical",0.20, 200.0, 35.0),
    # Extreme Critical matching the large delay observations
    ("extreme_Crit",  0.20, 5000.0, 3000.0),
]

for name, loss, delay, jitter in cases:
    bw = 20000.0
    row = {
        "bandwidth_usage_bps":     bw,
        "throughput_bps":          bw,
        "packet_loss_rate":        loss,
        "jitter":                  jitter,
        "avg_delay":               delay,
        "rolling_loss_mean":       loss,
        "rolling_loss_std":        0.0,
        "rolling_jitter_mean":     jitter,
        "rolling_delay_mean":      delay,
        "rolling_throughput_mean": bw,
        "rolling_throughput_std":  0.0,
        "loss_delta":              0.0,
        "jitter_delta":            0.0,
        "delay_delta":             0.0,
        "loss_accel":              0.0,
        "loss_trend_3":            0.0,
        "throughput_trend_3":      0.0,
        "delay_trend_3":           0.0,
        "delay_slope":             0.0,
        "loss_slope":              0.0,
        "active_devices":          8.0,
        "packets_per_window":      40.0,
        "loss_x_delay":            loss * delay,
        "loss_x_jitter":           loss * jitter,
    }
    X_test = pd.DataFrame([[row[n] for n in feat_names]], columns=feat_names)
    probs = model.predict_proba(X_test)[0]
    best  = max(range(len(probs)), key=lambda i: probs[i])
    pred  = classes[best]
    conf  = probs[best]
    print(f"  {name:<20} loss={loss:.2f} delay={delay:6.0f}ms jitter={jitter:6.0f}ms → {pred:<10} {conf:.1%}")

print()
