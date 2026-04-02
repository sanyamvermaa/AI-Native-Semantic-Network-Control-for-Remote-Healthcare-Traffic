"""
Quick sanity-check: retrained best_network_model must correctly classify
three representative operating points in ms-scale features.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

MODEL_PATH = PROJECT_ROOT / "models" / "best_network_model.pkl"
m = joblib.load(MODEL_PATH)
feats = list(m.feature_names_in_)
print(f"[+] Loaded: {MODEL_PATH.name}  ({type(m).__name__})")
print()


def make_vector(loss, delay_ms, jitter_ms, bw_bps=350_000.0):
    """
    Build a steady-state feature vector — all temporal/rolling fields reflect
    a system that has been in this condition for many windows.
    bandwidth_usage_bps ~350 kbps matches realistic 8-device ECG/BP traffic.
    """
    thr = bw_bps * (1.0 - loss)
    row = {
        "packet_loss_rate":           loss,
        "avg_delay":                  delay_ms,
        "jitter":                     jitter_ms,
        "bandwidth_usage_bps":        bw_bps,
        "throughput_bps":             thr,
        "active_devices":             8.0,
        "packets_per_window":         80.0,
        # Rolling means = instantaneous at steady state
        "rolling_loss_mean":          loss,
        "rolling_loss_std":           loss * 0.05,
        "rolling_delay_mean":         delay_ms,
        "rolling_jitter_mean":        jitter_ms,
        "rolling_throughput_mean":    thr,
        "rolling_throughput_std":     thr * 0.05,
        # Deltas/trends = 0 at steady state
        "loss_delta":                 0.0,
        "jitter_delta":               0.0,
        "delay_delta":                0.0,
        "loss_accel":                 0.0,
        "loss_trend_3":               0.0,
        "throughput_trend_3":         0.0,
        "delay_trend_3":              0.0,
        "delay_slope":                0.0,
        "loss_slope":                 0.0,
        # Interaction features
        "loss_x_delay":               loss * delay_ms,
        "loss_x_jitter":              loss * jitter_ms,
    }
    return pd.DataFrame([row], columns=feats)


CASES = [
    ("Stable",   0.002, 12.0,   2.0),
    ("Unstable", 0.050, 70.0,  15.0),
    ("Critical", 0.140, 230.0, 60.0),
]

print("─── Smoke Test ──────────────────────────────────────────────────")
all_pass = True
for expected, loss, delay, jitter in CASES:
    v = make_vector(loss, delay, jitter)
    pred = m.predict(v)[0]
    proba = m.predict_proba(v)[0]
    conf = max(proba)
    ok = (pred == expected)
    all_pass = all_pass and ok
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  loss={loss*100:.1f}%  delay={delay:>6.1f}ms  jitter={jitter:>5.1f}ms"
          f"  →  predicted={pred:<10}  expected={expected:<10}  conf={conf:.2f}")

print()
if all_pass:
    print("✓  ALL PASS — model is correctly scaled")
else:
    print("✗  SOME TESTS FAILED — check unit conversion")
    sys.exit(1)



def make_vector(loss, delay_ms, jitter_ms, bw_bps=25_000.0):
    """
    Build a steady-state feature vector — all temporal/rolling fields reflect
    a system that has been in this condition for many windows (no transients).
    """
    row = {f: 0.0 for f in feats}
    row["packet_loss_rate"]           = loss
    row["avg_delay"]                  = delay_ms
    row["jitter"]                     = jitter_ms
    row["bandwidth_usage_bps"]        = bw_bps
    row["throughput_bps"]             = bw_bps * (1.0 - loss)
    row["active_devices"]             = 8.0
    row["packets_per_window"]         = 80.0
    # Rolling means at steady state equal the instantaneous value
    row["rolling_loss_mean"]          = loss
    row["rolling_loss_std"]           = 0.001 * loss   # near-zero variance at steady state
    row["rolling_delay_mean"]         = delay_ms
    row["rolling_jitter_mean"]        = jitter_ms
    row["rolling_throughput_mean"]    = bw_bps * (1.0 - loss)
    row["rolling_throughput_std"]     = 0.0
    # Deltas/trends are zero at steady state
    row["loss_delta"]                 = 0.0
    row["jitter_delta"]               = 0.0
    row["delay_delta"]                = 0.0
    row["loss_accel"]                 = 0.0
    row["loss_trend_3"]               = 0.0
    row["throughput_trend_3"]         = 0.0
    row["delay_trend_3"]              = 0.0
    row["delay_slope"]                = 0.0
    row["loss_slope"]                 = 0.0
    # Interaction features
    row["loss_x_delay"]               = loss * delay_ms
    row["loss_x_jitter"]              = loss * jitter_ms
    return [[row[f] for f in feats]]


CASES = [
    ("Stable",   0.00, 12.0,  2.0),
    ("Unstable", 0.05, 60.0,  12.0),
    ("Critical", 0.15, 200.0, 55.0),
]

print("─── Smoke Test ──────────────────────────────────────────────────")
all_pass = True
for expected, loss, delay, jitter in CASES:
    v = make_vector(loss, delay, jitter)
    pred = m.predict(v)[0]
    proba = m.predict_proba(v)[0]
    conf = max(proba)
    ok = (pred == expected)
    all_pass = all_pass and ok
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  loss={loss*100:.0f}%  delay={delay:>6.1f}ms  jitter={jitter:>5.1f}ms"
          f"  →  predicted={pred:<10}  expected={expected:<10}  conf={conf:.2f}")

print()
if all_pass:
    print("✓  ALL PASS — model is correctly scaled")
else:
    print("✗  SOME TESTS FAILED — check unit conversion")
    sys.exit(1)
