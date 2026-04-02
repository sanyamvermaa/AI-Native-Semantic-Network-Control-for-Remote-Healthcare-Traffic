"""
Standalone: compute and compare heuristic vs new-model prediction distributions
on a given telemetry CSV, using the same rolling feature engineering as
analyze_results.py's ml_predict_series().
"""
import sys, argparse
from pathlib import Path
import pandas as pd
import joblib

# ── feature engineering (mirrors analyze_results._build_feature_row) ────────
def _build(curr, history):
    W, S = 10, 8
    def r(vals, n): return vals[-n:] if len(vals) >= n else vals
    def m(v): return sum(v)/len(v) if v else 0.0
    def sd(v):
        if len(v)<2: return 0.0
        mu=m(v); return (sum((x-mu)**2 for x in v)/len(v))**0.5
    def slope(v):
        n=len(v)
        if n<2: return 0.0
        xm=(n-1)/2; ym=m(v)
        num=sum((i-xm)*(y-ym) for i,y in enumerate(v))
        den=sum((i-xm)**2 for i in range(n))
        return num/den if den else 0.0

    lv=[h["packet_loss_rate"] for h in history]
    jv=[h["jitter"]           for h in history]
    dv=[h["avg_delay"]        for h in history]
    tv=[h["throughput_bps"]   for h in history]

    ld=lv[-1]-lv[-2] if len(lv)>=2 else 0.0
    jd=jv[-1]-jv[-2] if len(jv)>=2 else 0.0
    dd=dv[-1]-dv[-2] if len(dv)>=2 else 0.0
    pld=lv[-2]-lv[-3] if len(lv)>=3 else 0.0

    l3=m(r(lv,3)); l8=m(r(lv,8))
    t3=m(r(tv,3)); t8=m(r(tv,8))
    d3=m(r(dv,3)); d8=m(r(dv,8))

    return {
        "bandwidth_usage_bps":     curr["bandwidth_usage_bps"],
        "throughput_bps":          curr["throughput_bps"],
        "packet_loss_rate":        curr["packet_loss_rate"],
        "jitter":                  curr["jitter"],
        "avg_delay":               curr["avg_delay"],
        "rolling_loss_mean":       m(r(lv,W)),
        "rolling_loss_std":        sd(r(lv,W)),
        "rolling_jitter_mean":     m(r(jv,W)),
        "rolling_delay_mean":      m(r(dv,W)),
        "rolling_throughput_mean": m(r(tv,W)),
        "rolling_throughput_std":  sd(r(tv,W)),
        "loss_delta":              ld,
        "jitter_delta":            jd,
        "delay_delta":             dd,
        "loss_accel":              ld-pld,
        "loss_trend_3":            l3-l8,
        "throughput_trend_3":      t3-t8,
        "delay_trend_3":           d3-d8,
        "delay_slope":             slope(r(dv,S)),
        "loss_slope":              slope(r(lv,S)),
        "active_devices":          curr["active_devices"],
        "packets_per_window":      curr["packets_per_window"],
        "loss_x_delay":            curr["packet_loss_rate"]*curr["avg_delay"],
        "loss_x_jitter":           curr["packet_loss_rate"]*curr["jitter"],
    }

def heuristic(row):
    if row["packet_loss_rate"]>=0.08 or row["avg_delay"]>=130.0 or row["jitter"]>=35.0:
        return "Critical"
    if row["packet_loss_rate"]>=0.03 or row["avg_delay"]>=35.0 or row["jitter"]>=8.0:
        return "Unstable"
    return "Stable"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("telemetry_csv")
    p.add_argument("--model", default="models/best_network_model.pkl")
    args = p.parse_args()

    tel = pd.read_csv(args.telemetry_csv)
    tel["packet_loss_rate"] = pd.to_numeric(tel["packet_loss_rate"], errors="coerce").fillna(0.0)
    tel["jitter"]           = pd.to_numeric(tel["jitter"],           errors="coerce").fillna(0.0)
    tel["avg_delay"]        = pd.to_numeric(tel["avg_delay"],        errors="coerce").fillna(0.0)
    tel["throughput_bps"]   = pd.to_numeric(tel.get("throughput_bps", pd.Series(0,index=tel.index)), errors="coerce").fillna(0.0)
    tel["bandwidth_usage_bps"] = pd.to_numeric(tel.get("bandwidth_usage_bps", pd.Series(0,index=tel.index)), errors="coerce").fillna(0.0)
    tel["active_devices"]   = pd.to_numeric(tel.get("active_devices", pd.Series(0,index=tel.index)), errors="coerce").fillna(0.0)
    tel["packets_per_window"]= pd.to_numeric(tel.get("packets_per_window", pd.Series(0,index=tel.index)), errors="coerce").fillna(0.0)

    model = joblib.load(args.model)
    feat_names = list(model.feature_names_in_)

    history, ml_preds, heur_preds = [], [], []
    MIN_HIST = 8
    for _, row in tel.iterrows():
        curr = {k: float(row.get(k,0)) for k in
                ["bandwidth_usage_bps","throughput_bps","packet_loss_rate",
                 "jitter","avg_delay","active_devices","packets_per_window"]}
        history.append(curr)
        if len(history) > 64: history.pop(0)

        h = heuristic(curr)
        heur_preds.append(h)

        if len(history) >= MIN_HIST:
            feat_row = _build(curr, history)
            vector = pd.DataFrame([feat_row], columns=feat_names)
            ml_preds.append(model.predict(vector)[0])
        else:
            ml_preds.append(None)

    heur_s   = pd.Series(heur_preds)
    ml_valid = [(h,m) for h,m in zip(heur_preds, ml_preds) if m is not None]
    ml_s     = pd.Series([x[1] for x in ml_valid])
    h_v      = pd.Series([x[0] for x in ml_valid])

    total = len(heur_s)
    print(f"Total telemetry windows: {total}")
    print()
    print("Heuristic distribution (all windows):")
    for state, cnt in heur_s.value_counts().items():
        print(f"  {state:<12}: {cnt:>5}  ({cnt/total*100:.1f}%)")
    print()
    print(f"New model offline predictions ({len(ml_s)} windows with history ≥ {MIN_HIST}):")
    for state, cnt in ml_s.value_counts().items():
        print(f"  {state:<12}: {cnt:>5}  ({cnt/len(ml_s)*100:.1f}%)")
    print()
    agreement = (ml_s.values == h_v.values).mean()
    print(f"ML vs Heuristic agreement: {agreement*100:.1f}%")


if __name__ == "__main__":
    main()
