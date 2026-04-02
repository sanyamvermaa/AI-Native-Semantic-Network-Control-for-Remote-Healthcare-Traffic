"""
Pre-flight sanity check — run this BEFORE starting the live experiment.

Verifies:
  1. best_network_model.pkl loads via joblib
  2. Model feature names match what health_receiver.py feeds
  3. Model predicts sensibly across Stable / Unstable / Critical conditions
     using REALISTIC ms-scale feature vectors (with rolling history)
  4. No class-collapse (all three classes should appear across 3 test cases)
"""

import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "scripts" / "closed_loop"))

import joblib  # noqa: E402 — will error clearly if not installed

MODEL_PATH = P / "models" / "best_network_model.pkl"

print("=" * 64)
print("  PRE-FLIGHT MODEL CHECK")
print("=" * 64)

# ── 1. Load model ─────────────────────────────────────────────────────────────
print(f"\n[1] Loading model from: {MODEL_PATH}")
assert MODEL_PATH.exists(), f"FAIL: model not found at {MODEL_PATH}"
model = joblib.load(MODEL_PATH)
print(f"    Model type : {type(model).__name__}")

feat_names = list(getattr(model, "feature_names_in_", []))
classes    = list(getattr(model, "classes_", []))
print(f"    Features   : {len(feat_names)}")
print(f"    Classes    : {classes}")

assert feat_names, "FAIL: model has no feature_names_in_ attribute"
assert set(classes) == {"Stable", "Unstable", "Critical"}, \
    f"FAIL: unexpected classes {classes}"
print("    [OK] Model loaded")

# ── 2. Build feature vectors that mirror health_receiver.py ──────────────────
# health_receiver stores jitter/avg_delay in ms (after *1000 in feature_snapshot)
# Rolling history is also in ms.

def make_vector(loss_pct, delay_ms, jitter_ms, bw_bps=20_000):
    """Build a 24-feature vector matching health_receiver.py's build_feature_row."""
    # Simulate 10 windows of rolling history
    hist_loss   = [loss_pct]   * 10
    hist_jitter = [jitter_ms]  * 10
    hist_delay  = [delay_ms]   * 10
    hist_thr    = [bw_bps]     * 10

    def mean(v): return sum(v) / len(v) if v else 0.0
    def std(v):
        if len(v) < 2: return 0.0
        m = mean(v)
        return (sum((x - m)**2 for x in v) / len(v)) ** 0.5

    row = {
        "bandwidth_usage_bps":     bw_bps,
        "throughput_bps":          bw_bps,
        "packet_loss_rate":        loss_pct,
        "jitter":                  jitter_ms,
        "avg_delay":               delay_ms,
        "rolling_loss_mean":       mean(hist_loss),
        "rolling_loss_std":        std(hist_loss),
        "rolling_jitter_mean":     mean(hist_jitter),
        "rolling_delay_mean":      mean(hist_delay),
        "rolling_throughput_mean": mean(hist_thr),
        "rolling_throughput_std":  std(hist_thr),
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
        "loss_x_delay":            loss_pct * delay_ms,
        "loss_x_jitter":           loss_pct * jitter_ms,
    }
    return [[row[name] for name in feat_names]]


# ── 3. Test three canonical network states ─────────────────────────────────
print("\n[2] Testing canonical network conditions:")
print(f"    {'Condition':<12}  {'loss':>6}  {'delay':>8}  {'jitter':>8}  {'→ Prediction':<14}  {'Confidence':>10}")
print(f"    {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*10}")

test_cases = [
    ("Stable",   0.0,   12.0,  2.0),    # idle ward, clean WiFi
    ("Unstable", 0.04,  60.0, 12.0),    # shift change, moderate congestion
    ("Critical", 0.12, 150.0, 45.0),    # peak hour, heavy interference
]

predictions = []
all_ok = True
for label, loss, delay, jitter in test_cases:
    vec   = make_vector(loss, delay, jitter)
    probs = model.predict_proba(vec)[0]
    best  = max(range(len(probs)), key=lambda i: probs[i])
    pred  = classes[best]
    conf  = probs[best]
    predictions.append(pred)
    match = "✓" if pred == label else "✗ UNEXPECTED"
    print(f"    {label:<12}  {loss*100:>5.1f}%  {delay:>6.1f}ms  {jitter:>6.1f}ms  → {pred:<14}  {conf:>9.1%}  {match}")
    if pred != label:
        all_ok = False

# ── 4. Class collapse check ─────────────────────────────────────────────────
print(f"\n[3] Class collapse check: unique predictions = {set(predictions)}")
if len(set(predictions)) < 3:
    print("    [WARN] Not all three classes predicted — model may still be biased")
else:
    print("    [OK] All three classes predicted correctly")

# ── 5. Feature name alignment spot-check ────────────────────────────────────
EXPECTED_FEATURES = {
    "bandwidth_usage_bps", "throughput_bps", "packet_loss_rate",
    "jitter", "avg_delay", "rolling_loss_mean", "rolling_loss_std",
    "rolling_jitter_mean", "rolling_delay_mean", "rolling_throughput_mean",
    "rolling_throughput_std", "loss_delta", "jitter_delta", "delay_delta",
    "loss_accel", "loss_trend_3", "throughput_trend_3", "delay_trend_3",
    "delay_slope", "loss_slope", "active_devices", "packets_per_window",
    "loss_x_delay", "loss_x_jitter",
}
missing = EXPECTED_FEATURES - set(feat_names)
extra   = set(feat_names) - EXPECTED_FEATURES
if missing:
    print(f"\n[FAIL] Model is missing expected features: {missing}")
    all_ok = False
if extra:
    print(f"\n[WARN] Model has unexpected extra features: {extra}")
if not missing and not extra:
    print("\n[4] Feature alignment: [OK] 24/24 features match health_receiver.py")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
if all_ok:
    print("  RESULT: ALL CHECKS PASSED — safe to start live experiment")
else:
    print("  RESULT: SOME CHECKS FAILED — DO NOT run the experiment yet")
print("=" * 64 + "\n")
sys.exit(0 if all_ok else 1)
