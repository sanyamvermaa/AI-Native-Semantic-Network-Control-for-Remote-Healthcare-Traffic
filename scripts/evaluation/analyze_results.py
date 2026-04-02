#!/usr/bin/env python3
"""
analyze_results.py — Generate all 10 evaluation figures + summary_report.txt

Usage:
    python scripts/evaluation/analyze_results.py \\
        --baseline-dir   outputs/evaluation/baseline_YYYYMMDD_HHMMSS \\
        --closedloop-dir outputs/evaluation/closedloop_YYYYMMDD_HHMMSS \\
        --output-dir     outputs/evaluation/figures

Produces:
    01_bandwidth_over_time.png
    02_packet_loss_comparison.png
    03_semantic_suppression.png
    04_adaptation_latency.png
    05_command_distribution.png
    06_clinical_delivery.png
    07_bandwidth_savings_bar.png
    08_telemetry_timeline.png
    09_network_state_agreement.png
    10_suppression_over_time.png
    summary_report.txt
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
})

STATE_COLORS = {
    "Stable":   "#2ca02c",
    "Unstable": "#ff7f0e",
    "Critical": "#d62728",
    "UNKNOWN":  "#999999",
}

COMMAND_COLORS = {
    "FULL_ECG":           "#1f77b4",
    "FULL_ECG_PRIORITY":  "#17becf",
    "DOWNSAMPLED_ECG":    "#ff7f0e",
    "SEMANTIC_ALERT":     "#e377c2",
    "SEMANTIC_CRITICAL":  "#d62728",
    "SEMANTIC_SUMMARY":   "#7f7f7f",
}

RUN_COLORS = {
    "baseline":   "#7570b3",
    "closedloop": "#1b9e77",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_csv_safe(path: str) -> pd.DataFrame:
    """Read a CSV file, returning an empty DataFrame if missing or malformed."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"  [WARN] Could not read {path}: {exc}")
        return pd.DataFrame()


def load_telemetry(run_dir: str) -> pd.DataFrame:
    df = _read_csv_safe(os.path.join(run_dir, "network_telemetry.csv"))
    if df.empty:
        return df
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["elapsed"] = df["timestamp"] - df["timestamp"].iloc[0]
    for col in ["bandwidth_usage_bps", "throughput_bps", "packet_loss_rate",
                "jitter", "avg_delay", "active_devices", "packets_per_window"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_command_log(run_dir: str) -> pd.DataFrame:
    df = _read_csv_safe(os.path.join(run_dir, "command_log.csv"))
    if df.empty:
        return df
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_sender_stats(run_dir: str) -> pd.DataFrame:
    pattern = os.path.join(run_dir, "sender_semantic_stats_dev*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            dfs.append(pd.read_csv(fp, low_memory=False))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in ["total_samples", "total_sent", "total_suppressed",
                "raw_sent", "raw_suppressed",
                "delta_sent", "delta_suppressed",
                "summary_sent", "summary_suppressed",
                "critical_only_sent", "critical_only_suppressed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_sender_logs(run_dir: str) -> pd.DataFrame:
    pattern = os.path.join(run_dir, "sender_log_dev*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            dfs.append(pd.read_csv(fp, low_memory=False))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"]     = pd.to_numeric(df["value"],     errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rule-based heuristic (matches common.py choose_network_state)
# ---------------------------------------------------------------------------

def heuristic_state_series(df: pd.DataFrame) -> pd.Series:
    """Compute the rule-based state for every row in a telemetry DataFrame."""
    def _classify(row):
        loss   = row.get("packet_loss_rate", 0.0)
        delay  = row.get("avg_delay", 0.0)
        jitter = row.get("jitter", 0.0)
        if loss >= 0.08 or delay >= 130.0 or jitter >= 35.0:
            return "Critical"
        if loss >= 0.03 or delay >= 35.0 or jitter >= 8.0:
            return "Unstable"
        return "Stable"
    return df.apply(_classify, axis=1)


# ---------------------------------------------------------------------------
# Offline ML re-prediction (mirrors health_receiver.py feature engineering)
# ---------------------------------------------------------------------------

def _build_feature_row(curr: Dict, history: List[Dict]) -> Dict:
    W, S = 10, 8

    def rolling(vals, n):
        chunk = vals[-n:] if len(vals) >= n else vals
        return chunk

    def mean_or_zero(v): return sum(v) / len(v) if v else 0.0
    def std_or_zero(v):
        if len(v) < 2: return 0.0
        m = mean_or_zero(v)
        return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    def simple_slope(v):
        n = len(v)
        if n < 2: return 0.0
        xm = (n - 1) / 2.0; ym = mean_or_zero(v)
        num = sum((i - xm) * (y - ym) for i, y in enumerate(v))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    loss_vals   = [h["packet_loss_rate"] for h in history]
    jitter_vals = [h["jitter"]           for h in history]
    delay_vals  = [h["avg_delay"]        for h in history]
    thr_vals    = [h["throughput_bps"]   for h in history]

    loss_delta   = loss_vals[-1]   - loss_vals[-2]   if len(loss_vals)   >= 2 else 0.0
    jitter_delta = jitter_vals[-1] - jitter_vals[-2] if len(jitter_vals) >= 2 else 0.0
    delay_delta  = delay_vals[-1]  - delay_vals[-2]  if len(delay_vals)  >= 2 else 0.0
    prev_loss_delta = loss_vals[-2] - loss_vals[-3]  if len(loss_vals)   >= 3 else 0.0
    loss_accel = loss_delta - prev_loss_delta

    loss3  = mean_or_zero(rolling(loss_vals, 3));  loss8  = mean_or_zero(rolling(loss_vals, 8))
    thr3   = mean_or_zero(rolling(thr_vals, 3));   thr8   = mean_or_zero(rolling(thr_vals, 8))
    delay3 = mean_or_zero(rolling(delay_vals, 3)); delay8 = mean_or_zero(rolling(delay_vals, 8))

    return {
        "bandwidth_usage_bps":    curr["bandwidth_usage_bps"],
        "throughput_bps":         curr["throughput_bps"],
        "packet_loss_rate":       curr["packet_loss_rate"],
        "jitter":                 curr["jitter"],
        "avg_delay":              curr["avg_delay"],
        "rolling_loss_mean":      mean_or_zero(rolling(loss_vals, W)),
        "rolling_loss_std":       std_or_zero(rolling(loss_vals, W)),
        "rolling_jitter_mean":    mean_or_zero(rolling(jitter_vals, W)),
        "rolling_delay_mean":     mean_or_zero(rolling(delay_vals, W)),
        "rolling_throughput_mean":mean_or_zero(rolling(thr_vals, W)),
        "rolling_throughput_std": std_or_zero(rolling(thr_vals, W)),
        "loss_delta":             loss_delta,
        "jitter_delta":           jitter_delta,
        "delay_delta":            delay_delta,
        "loss_accel":             loss_accel,
        "loss_trend_3":           loss3 - loss8,
        "throughput_trend_3":     thr3 - thr8,
        "delay_trend_3":          delay3 - delay8,
        "delay_slope":            simple_slope(rolling(delay_vals, S)),
        "loss_slope":             simple_slope(rolling(loss_vals, S)),
        "active_devices":         curr["active_devices"],
        "packets_per_window":     curr["packets_per_window"],
        "loss_x_delay":           curr["packet_loss_rate"] * curr["avg_delay"],
        "loss_x_jitter":          curr["packet_loss_rate"] * curr["jitter"],
    }


def ml_predict_series(cl_tel: pd.DataFrame, model_path: str) -> Optional[pd.Series]:
    """
    Load the trained model and re-predict network state offline for every row
    in cl_tel, using the same 24-feature engineering as health_receiver.py.
    Returns a Series of predicted state labels, or None if model can't be loaded.
    """
    try:
        import joblib
    except ImportError:
        print("  [WARN] joblib not installed — cannot load model for offline prediction")
        return None

    try:
        model = joblib.load(model_path)
        print(f"  [MODEL] Loaded: {model_path}")
    except Exception as exc:
        print(f"  [WARN] Could not load model {model_path}: {exc}")
        return None

    CLASS_MAP = {0: "Critical", 1: "Stable", 2: "Unstable"}  # matches training encoding
    feat_names = list(getattr(model, "feature_names_in_", []))

    history: List[Dict] = []
    predictions = []

    for _, row in cl_tel.iterrows():
        curr = {
            "bandwidth_usage_bps": float(row.get("bandwidth_usage_bps", 0)),
            "throughput_bps":      float(row.get("throughput_bps", 0)),
            "packet_loss_rate":    float(row.get("packet_loss_rate", 0)),
            "jitter":              float(row.get("jitter", 0)),
            "avg_delay":           float(row.get("avg_delay", 0)),
            "active_devices":      float(row.get("active_devices", 0)),
            "packets_per_window":  float(row.get("packets_per_window", 0)),
        }
        history.append(curr)
        if len(history) > 16:
            history.pop(0)

        feat = _build_feature_row(curr, history)
        if feat_names:
            vector = pd.DataFrame([[feat.get(n, 0.0) for n in feat_names]], columns=feat_names)
        else:
            cols = sorted(feat)
            vector = pd.DataFrame([[feat[k] for k in cols]], columns=cols)

        try:
            raw = model.predict(vector)[0]
            if isinstance(raw, str):
                pred = raw
            else:
                pred = CLASS_MAP.get(int(raw), "Critical")
        except Exception:
            pred = heuristic_state_series(pd.DataFrame([curr])).iloc[0]

        predictions.append(pred)

    return pd.Series(predictions, index=cl_tel.index)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _shade_by_state(ax, df: pd.DataFrame, state_col: str, x_col: str) -> None:
    """Add translucent background shading keyed to network_condition."""
    if df.empty or state_col not in df.columns:
        return
    xs = df[x_col].values
    states = df[state_col].values
    prev_x, prev_s = xs[0], states[0]
    for x, s in zip(xs[1:], states[1:]):
        if s != prev_s:
            ax.axvspan(prev_x, x, alpha=0.06, color=STATE_COLORS.get(prev_s, "#ccc"), linewidth=0)
            prev_x, prev_s = x, s
    ax.axvspan(prev_x, xs[-1], alpha=0.06, color=STATE_COLORS.get(prev_s, "#ccc"), linewidth=0)


def _state_legend_patches() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(color=STATE_COLORS[s], alpha=0.4, label=s)
        for s in ("Stable", "Unstable", "Critical")
    ]


def _save(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}.png")


def _latest_stats_per_device(stats: pd.DataFrame) -> pd.DataFrame:
    """Return the last stats row per (device_id, device_type) pair."""
    if stats.empty:
        return stats
    return (
        stats.sort_values("timestamp")
             .groupby(["device_id", "device_type"], as_index=False)
             .last()
    )


# ---------------------------------------------------------------------------
# Figure 01 — Bandwidth over Time
# ---------------------------------------------------------------------------

def fig01_bandwidth_over_time(b_tel: pd.DataFrame, cl_tel: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    state_patches = _state_legend_patches()

    for ax, (label, df) in zip(axes, [("Baseline", b_tel), ("Closed-Loop", cl_tel)]):
        if df.empty:
            ax.text(0.5, 0.5, f"No data ({label})", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue
        key = "baseline" if label == "Baseline" else "closedloop"
        color = RUN_COLORS[key]
        _shade_by_state(ax, df, "network_condition", "elapsed")
        ax.plot(df["elapsed"], df["throughput_bps"] / 1000,
                color=color, linewidth=1.4, label="Throughput (kbps)")
        ax.plot(df["elapsed"], df["bandwidth_usage_bps"] / 1000,
                "--", color=color, alpha=0.45, linewidth=0.9, label="Attempted BW (kbps)")
        ax.set_ylabel("kbps")
        ax.set_title(f"{label} — Bandwidth over Time")
        line_handles, line_labels = ax.get_legend_handles_labels()
        ax.legend(handles=state_patches + line_handles, fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle("Figure 01 — Bandwidth over Time", fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "01_bandwidth_over_time")


# ---------------------------------------------------------------------------
# Figure 02 — Packet Loss Comparison
# ---------------------------------------------------------------------------

def fig02_packet_loss_comparison(b_tel: pd.DataFrame, cl_tel: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    for df, label in [(b_tel, "Baseline"), (cl_tel, "Closed-Loop")]:
        if df.empty:
            continue
        color = RUN_COLORS["baseline" if label == "Baseline" else "closedloop"]
        ax.plot(df["elapsed"], df["packet_loss_rate"] * 100,
                color=color, linewidth=1.3, label=label, alpha=0.85)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Packet loss (%)")
    ax.set_title("Figure 02 — Packet Loss Comparison (same netem, different response)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "02_packet_loss_comparison")


# ---------------------------------------------------------------------------
# Figure 03 — Semantic Suppression
# ---------------------------------------------------------------------------

def fig03_semantic_suppression(cl_stats: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if cl_stats.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No stats data (closed-loop)", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle("Figure 03 — Semantic Suppression Distribution", fontweight="bold")
        _save(fig, out_dir, "03_semantic_suppression")
        return

    latest = _latest_stats_per_device(cl_stats)

    modes = ["raw", "delta", "summary", "critical_only"]
    mode_labels = ["RAW", "DELTA", "SUMMARY", "CRITICAL_ONLY"]
    mode_colors = ["#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728"]

    sent_totals = [
        int(latest.get(f"{m}_sent", pd.Series([0])).sum()) for m in modes
    ]
    supp_totals = [
        int(latest.get(f"{m}_suppressed", pd.Series([0])).sum()) for m in modes
    ]

    # Panel 1: sent vs suppressed by mode
    ax = axes[0]
    x = np.arange(len(modes))
    w = 0.38
    ax.bar(x - w / 2, sent_totals, w, label="Sent",       color="#1b9e77", alpha=0.9)
    ax.bar(x + w / 2, supp_totals, w, label="Suppressed", color="#d95f02", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=9)
    ax.set_ylabel("Packets")
    ax.set_title("Packets Sent vs Suppressed by Mode")
    ax.legend()

    # Panel 2: pie — sent packets by mode
    ax2 = axes[1]
    nonzero = [(lbl, s, c) for lbl, s, c in zip(mode_labels, sent_totals, mode_colors) if s > 0]
    if nonzero:
        labels, sizes, colors = zip(*nonzero)
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=140, pctdistance=0.82,
        )
        for t in autotexts:
            t.set_fontsize(8)
    else:
        ax2.text(0.5, 0.5, "All modes zero", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("Sent Packets by Transmission Mode")

    fig.suptitle("Figure 03 — Semantic Suppression Distribution (Closed-Loop)",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "03_semantic_suppression")


# ---------------------------------------------------------------------------
# Figure 04 — Adaptation Latency CDF
# ---------------------------------------------------------------------------

def fig04_adaptation_latency(cmd_log: pd.DataFrame, out_dir: str) -> Optional[float]:
    fig, ax = plt.subplots(figsize=(8, 5))

    if cmd_log.empty or "latency_ms" not in cmd_log.columns:
        ax.text(0.5, 0.5, "No command log data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Figure 04 — Adaptation Latency CDF", fontweight="bold")
        _save(fig, out_dir, "04_adaptation_latency")
        return None

    latencies = cmd_log["latency_ms"].dropna()
    latencies = latencies[latencies > 0]

    if latencies.empty:
        ax.text(0.5, 0.5, "No positive latency values", ha="center", va="center",
                transform=ax.transAxes)
        _save(fig, out_dir, "04_adaptation_latency")
        return None

    sorted_lat = np.sort(latencies.values)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat) * 100

    ax.plot(sorted_lat, cdf, color=RUN_COLORS["closedloop"], linewidth=2.0)
    ax.fill_between(sorted_lat, cdf, alpha=0.1, color=RUN_COLORS["closedloop"])

    for pctile, ls in [(50, "--"), (90, ":"), (95, "-.")]:
        p = float(np.percentile(sorted_lat, pctile))
        ax.axvline(p, linestyle=ls, color="#d62728", alpha=0.8, linewidth=1.3)
        ax.text(p + max(1, p * 0.02), 15 + pctile * 0.35,
                f"p{pctile}={p:.0f} ms", fontsize=8, color="#d62728")

    p50 = float(np.percentile(sorted_lat, 50))
    ax.set_xlabel("Adaptation Latency (ms)")
    ax.set_ylabel("CDF (%)")
    ax.set_ylim(0, 108)
    ax.set_title(
        "Figure 04 — Adaptation Latency CDF\n"
        "(per-window: time from telemetry packet to command delivery)",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out_dir, "04_adaptation_latency")
    return p50


# ---------------------------------------------------------------------------
# Figure 05 — Command Distribution Pie
# ---------------------------------------------------------------------------

def fig05_command_distribution(cmd_log: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    if cmd_log.empty or "command" not in cmd_log.columns:
        ax.text(0.5, 0.5, "No command log data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Figure 05 — Command Distribution", fontweight="bold")
        _save(fig, out_dir, "05_command_distribution")
        return

    counts = cmd_log["command"].value_counts()
    colors = [COMMAND_COLORS.get(c, "#aaaaaa") for c in counts.index]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.80,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title(
        "Figure 05 — Command Distribution\n(9-case semantic policy in action)",
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out_dir, "05_command_distribution")


# ---------------------------------------------------------------------------
# Figure 06 — Clinical Delivery
# ---------------------------------------------------------------------------

def fig06_clinical_delivery(
    b_tel: pd.DataFrame, cl_tel: pd.DataFrame,
    b_stats: pd.DataFrame, cl_stats: pd.DataFrame,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: overall transmission efficiency (sent / generated)
    ax = axes[0]
    bar_labels_eff, eff_vals, supp_vals = [], [], []
    for label, stats in [("Baseline", b_stats), ("Closed-Loop", cl_stats)]:
        if stats.empty:
            continue
        latest = _latest_stats_per_device(stats)
        total_s = latest["total_samples"].sum()
        total_t = latest["total_sent"].sum()
        total_p = latest["total_suppressed"].sum()
        bar_labels_eff.append(label)
        eff_vals.append(total_t / total_s * 100 if total_s > 0 else 0.0)
        supp_vals.append(total_p / total_s * 100 if total_s > 0 else 0.0)

    if bar_labels_eff:
        xb = np.arange(len(bar_labels_eff))
        w = 0.35
        bar_colors = [RUN_COLORS["baseline"], RUN_COLORS["closedloop"]][:len(bar_labels_eff)]
        ax.bar(xb - w / 2, eff_vals,  w, label="Transmitted (%)", color=bar_colors, alpha=0.9)
        ax.bar(xb + w / 2, supp_vals, w, label="Suppressed (%)",
               color=["#d95f02", "#e377c2"][:len(bar_labels_eff)], alpha=0.9)
        ax.set_xticks(xb)
        ax.set_xticklabels(bar_labels_eff)
        ax.set_ylim(0, 115)
        ax.set_ylabel("% of generated samples")
        ax.set_title("Transmission Rate\n(Baseline = 100% RAW; CL filters semantically)")
        ax.legend(fontsize=8)

    # Panel 2: CRITICAL_ONLY mode clinical retention
    #   clinical_only_sent   = packets with importance≥0.7 (ALERT+CRITICAL) — retained
    #   critical_only_suppressed = NORMAL packets — intentionally dropped
    ax2 = axes[1]
    bar_labels_clin, retain_pct = [], []
    for label, stats in [("Baseline", b_stats), ("Closed-Loop", cl_stats)]:
        if stats.empty:
            continue
        latest = _latest_stats_per_device(stats)
        co_sent = int(latest.get("critical_only_sent",       pd.Series([0])).sum())
        co_supp = int(latest.get("critical_only_suppressed", pd.Series([0])).sum())
        total_co = co_sent + co_supp
        if total_co > 0:
            pct = co_sent / total_co * 100
        else:
            # Baseline never enters CRITICAL_ONLY — clinically 100% by definition
            pct = 100.0
        bar_labels_clin.append(label)
        retain_pct.append(pct)

    if bar_labels_clin:
        xc = np.arange(len(bar_labels_clin))
        bar_colors2 = [RUN_COLORS["baseline"], RUN_COLORS["closedloop"]][:len(bar_labels_clin)]
        bars = ax2.bar(xc, retain_pct, width=0.45, color=bar_colors2, alpha=0.9)
        for bar, val in zip(bars, retain_pct):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax2.set_xticks(xc)
        ax2.set_xticklabels(bar_labels_clin)
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Clinical retention (%)")
        ax2.set_title("CRITICAL_ONLY Mode — Clinical Packet Retention\n"
                      "(importance≥0.7 packets = ALERT + CRITICAL kept)")
        ax2.axhline(100, linestyle="--", color="#888888", linewidth=0.8, alpha=0.6)

    fig.suptitle("Figure 06 — Clinical Delivery: Semantic Compression vs Patient Safety",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "06_clinical_delivery")


# ---------------------------------------------------------------------------
# Figure 07 — Bandwidth Savings Bar
# ---------------------------------------------------------------------------

def fig07_bandwidth_savings_bar(
    b_tel: pd.DataFrame, cl_tel: pd.DataFrame, out_dir: str
) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(9, 5))
    savings: Dict[str, float] = {}

    if b_tel.empty or cl_tel.empty:
        ax.text(0.5, 0.5, "Missing telemetry — run both experiments first",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Figure 07 — Bandwidth Savings by Network State", fontweight="bold")
        _save(fig, out_dir, "07_bandwidth_savings_bar")
        return savings

    states = ["Stable", "Unstable", "Critical"]
    b_means, cl_means, pct_list = [], [], []

    for state in states:
        b_rows  = b_tel[b_tel["network_condition"] == state]["throughput_bps"]
        cl_rows = cl_tel[cl_tel["network_condition"] == state]["throughput_bps"]
        b_mean  = float(b_rows.mean())  if len(b_rows)  > 0 else 0.0
        cl_mean = float(cl_rows.mean()) if len(cl_rows) > 0 else 0.0
        pct     = (b_mean - cl_mean) / b_mean * 100 if b_mean > 0 else 0.0
        b_means.append(b_mean / 1000)
        cl_means.append(cl_mean / 1000)
        pct_list.append(pct)
        savings[state] = pct

    x = np.arange(len(states))
    w = 0.35
    bars_b  = ax.bar(x - w / 2, b_means,  w, label="Baseline",    color=RUN_COLORS["baseline"],   alpha=0.9)
    bars_cl = ax.bar(x + w / 2, cl_means, w, label="Closed-Loop", color=RUN_COLORS["closedloop"],  alpha=0.9)

    for i, (bb, bc, pct) in enumerate(zip(bars_b, bars_cl, pct_list)):
        y = max(bb.get_height(), bc.get_height()) + max(b_means) * 0.02
        ax.text(x[i], y, f"{pct:+.1f}%", ha="center", fontsize=9,
                color="#d62728" if pct > 0 else "#2ca02c", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylabel("Mean throughput (kbps)")
    ax.set_title("Figure 07 — Bandwidth Savings by Network State\n"
                 "(positive % = closed-loop saves bandwidth vs baseline)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir, "07_bandwidth_savings_bar")
    return savings


# ---------------------------------------------------------------------------
# Figure 08 — Telemetry Timeline (3-panel)
# ---------------------------------------------------------------------------

def fig08_telemetry_timeline(b_tel: pd.DataFrame, cl_tel: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    metrics = [
        ("packet_loss_rate", "Packet Loss",   lambda v: v * 100, "%"),
        ("avg_delay",        "Average Delay", lambda v: v,       "ms"),
        ("jitter",           "Jitter",        lambda v: v,       "ms"),
    ]
    for ax, (col, title, transform, unit) in zip(axes, metrics):
        for df, label in [(b_tel, "Baseline"), (cl_tel, "Closed-Loop")]:
            if df.empty or col not in df.columns:
                continue
            color = RUN_COLORS["baseline" if label == "Baseline" else "closedloop"]
            ax.plot(df["elapsed"], transform(df[col]),
                    color=color, linewidth=1.2, label=label, alpha=0.85)
        ax.set_ylabel(f"{title} ({unit})")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle("Figure 08 — Network Telemetry Timeline: Loss / Delay / Jitter",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "08_telemetry_timeline")


# ---------------------------------------------------------------------------
# Figure 09 — Network State Agreement (ML vs Heuristic)
# ---------------------------------------------------------------------------

def fig09_network_state_agreement(
    cl_tel: pd.DataFrame, out_dir: str,
    ml_predictions: Optional[pd.Series] = None,
) -> Optional[float]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if cl_tel.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No closed-loop telemetry", ha="center", va="center",
                    transform=ax.transAxes)
        fig.suptitle("Figure 09 — Network State Agreement: ML vs Heuristic", fontweight="bold")
        _save(fig, out_dir, "09_network_state_agreement")
        return None

    cl = cl_tel.copy()
    cl["heuristic"] = heuristic_state_series(cl)

    # Use offline ML predictions if available, else fall back to the recorded column
    if ml_predictions is not None:
        cl["ml_pred"] = ml_predictions.values
        compare_col   = "ml_pred"
        mode_label    = "Offline ML re-prediction"
    else:
        cl["ml_pred"] = cl["network_condition"]
        compare_col   = "ml_pred"
        mode_label    = "Recorded state (heuristic — model not loaded during run)"

    cl["agree"]    = cl[compare_col] == cl["heuristic"]
    agreement_rate = float(cl["agree"].mean() * 100)

    # Panel 1: agreement over time + where they diverge
    ax = axes[0]
    cl["agree_roll"] = cl["agree"].astype(float).rolling(5, min_periods=1).mean()
    ax.fill_between(cl["elapsed"], cl["agree_roll"],
                    alpha=0.35, color=RUN_COLORS["closedloop"], label="Agreement (rolling)")

    # Shade disagreement windows in red
    disagree = cl[~cl["agree"]]
    if not disagree.empty:
        ax.scatter(disagree["elapsed"], [0.05] * len(disagree),
                   color="#d62728", s=12, zorder=4, alpha=0.7, label="Disagreement windows")

    ax.axhline(agreement_rate / 100, linestyle="--", color="black", linewidth=1.1,
               label=f"Mean agreement: {agreement_rate:.1f}%")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Agreement fraction (1 = match)")
    ax.set_title(f"ML vs Heuristic Agreement over Time\n({mode_label})")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(fontsize=8)

    # Panel 2: confusion matrix — rows = heuristic, cols = ML pred
    ax2 = axes[1]
    states = ["Stable", "Unstable", "Critical"]
    confusion = np.zeros((3, 3), dtype=int)
    for i, h_s in enumerate(states):
        for j, m_s in enumerate(states):
            confusion[i, j] = int(
                ((cl["heuristic"] == h_s) & (cl[compare_col] == m_s)).sum()
            )
    im = ax2.imshow(confusion, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(states, fontsize=9)
    ax2.set_yticklabels(states, fontsize=9)
    ax2.set_xlabel("ML Predicted State")
    ax2.set_ylabel("Heuristic State (ground truth proxy)")
    ax2.set_title(f"Confusion Matrix\n(Overall agreement: {agreement_rate:.1f}%)")
    cmax = max(confusion.max(), 1)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=9,
                     color="white" if confusion[i, j] > cmax * 0.55 else "black")
    plt.colorbar(im, ax=ax2)

    fig.suptitle("Figure 09 — Network State Agreement: ML Predictor vs Rule-Based Heuristic",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "09_network_state_agreement")
    return agreement_rate


# ---------------------------------------------------------------------------
# Figure 10 — Throughput over Time with Command Change Markers
# ---------------------------------------------------------------------------

def fig10_suppression_over_time(
    cl_tel: pd.DataFrame, cmd_log: pd.DataFrame, out_dir: str
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    if cl_tel.empty:
        ax.text(0.5, 0.5, "No closed-loop telemetry", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Figure 10 — Throughput over Time with Command Markers", fontweight="bold")
        _save(fig, out_dir, "10_suppression_over_time")
        return

    cl = cl_tel.copy()
    t0 = float(cl["timestamp"].iloc[0])
    cl["rolling_bw"] = cl["throughput_bps"].rolling(5, min_periods=1).mean() / 1000

    # Shaded background by network state
    _shade_by_state(ax, cl, "network_condition", "elapsed")

    ax.plot(cl["elapsed"], cl["rolling_bw"],
            color=RUN_COLORS["closedloop"], linewidth=1.8,
            label="CL Throughput (5-window rolling, kbps)", zorder=3)

    # Command change markers
    if not cmd_log.empty and "command" in cmd_log.columns:
        cmd = cmd_log.copy()
        cmd["elapsed"] = cmd["timestamp"] - t0
        prev_cmd = None
        for _, row in cmd.iterrows():
            if row["command"] != prev_cmd:
                color = COMMAND_COLORS.get(row["command"], "#999999")
                ax.axvline(row["elapsed"], color=color, linewidth=1.0, alpha=0.75, linestyle="--")
                # Annotate every 2nd change to avoid overlap
                if prev_cmd is None or row["command"] != prev_cmd:
                    ylim = ax.get_ylim()
                    ypos = ylim[1] * 0.92 if ylim[1] > 0 else 5
                    ax.text(row["elapsed"] + 0.2, ypos, row["command"],
                            rotation=90, fontsize=5, color=color, va="top", alpha=0.85)
                prev_cmd = row["command"]

    # Legend: command colors + state shading
    cmd_patches  = [mpatches.Patch(color=v, label=k) for k, v in COMMAND_COLORS.items()]
    state_patches = _state_legend_patches()
    ax.legend(handles=cmd_patches + state_patches, fontsize=6, loc="upper right",
              ncol=3, framealpha=0.85)

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Throughput (kbps, rolling avg)")
    ax.set_title("Figure 10 — Closed-Loop Throughput with Semantic Command Change Markers",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "10_suppression_over_time")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary_report(
    b_tel: pd.DataFrame, cl_tel: pd.DataFrame,
    b_stats: pd.DataFrame, cl_stats: pd.DataFrame,
    cmd_log: pd.DataFrame,
    savings: Dict[str, float],
    agreement_rate: Optional[float],
    latency_p50: Optional[float],
    out_dir: str,
) -> None:
    W = 68
    lines: List[str] = []

    def hr(char="─"):
        lines.append(char * W)

    def section(title):
        lines.append(f"── {title} " + "─" * (W - len(title) - 4))

    hr("=")
    lines.append("  SEMANTIC NETWORK CONTROL — EVALUATION SUMMARY REPORT")
    hr("=")
    lines.append("")

    def safe_mean(df, col):
        if df.empty or col not in df.columns:
            return float("nan")
        return float(df[col].mean())

    # 1. Bandwidth
    section("1. Bandwidth")
    b_bw  = safe_mean(b_tel,  "throughput_bps") / 1000
    cl_bw = safe_mean(cl_tel, "throughput_bps") / 1000
    overall_saving = (b_bw - cl_bw) / b_bw * 100 if b_bw > 0 else float("nan")
    lines.append(f"  Baseline mean throughput    : {b_bw:.2f} kbps")
    lines.append(f"  Closed-loop mean throughput : {cl_bw:.2f} kbps")
    lines.append(f"  Overall bandwidth saving    : {overall_saving:.1f}%")
    lines.append("")
    lines.append("  Per network-state savings:")
    for state in ("Stable", "Unstable", "Critical"):
        pct = savings.get(state, float("nan"))
        lines.append(f"    {state:<12}: {pct:+.1f}%  saving vs baseline")
    lines.append("")

    # 2. Packet loss
    section("2. Packet Loss")
    b_loss  = safe_mean(b_tel,  "packet_loss_rate") * 100
    cl_loss = safe_mean(cl_tel, "packet_loss_rate") * 100
    lines.append(f"  Baseline mean loss    : {b_loss:.2f}%")
    lines.append(f"  Closed-loop mean loss : {cl_loss:.2f}%")
    lines.append("")

    # 3. Semantic suppression
    section("3. Semantic Suppression  (Closed-Loop)")
    if not cl_stats.empty:
        latest = _latest_stats_per_device(cl_stats)
        ts = int(latest["total_samples"].sum())
        tx = int(latest["total_sent"].sum())
        su = int(latest["total_suppressed"].sum())
        lines.append(f"  Total samples generated : {ts:>12,}")
        if ts > 0:
            lines.append(f"  Total transmitted       : {tx:>12,}  ({tx / ts * 100:.1f}%)")
            lines.append(f"  Total suppressed        : {su:>12,}  ({su / ts * 100:.1f}%)")
        lines.append("")
        lines.append(f"  {'Mode':<18}  {'Sent':>10}  {'Suppressed':>12}")
        lines.append(f"  {'─'*18}  {'─'*10}  {'─'*12}")
        for mode in ["raw", "delta", "summary", "critical_only"]:
            ms  = int(latest.get(f"{mode}_sent",       pd.Series([0])).sum())
            msp = int(latest.get(f"{mode}_suppressed", pd.Series([0])).sum())
            if ms + msp > 0:
                lines.append(f"  {mode.upper():<18}  {ms:>10,}  {msp:>12,}")
    lines.append("")

    # 3b. Suppression-equivalent bandwidth saving
    section("3b. Suppression-Equivalent Bandwidth Saving")
    if not cl_stats.empty and not b_stats.empty:
        cl_latest = _latest_stats_per_device(cl_stats)
        b_latest  = _latest_stats_per_device(b_stats)
        cl_tx = int(cl_latest["total_sent"].sum())
        b_tx  = int(b_latest["total_sent"].sum())
        if b_tx > 0:
            bw_saving_pct = (1.0 - cl_tx / b_tx) * 100.0
            lines.append(f"  CL transmitted packets  : {cl_tx:>10,}")
            lines.append(f"  Baseline transmitted    : {b_tx:>10,}")
            lines.append(f"  Bandwidth saving        : {bw_saving_pct:>9.1f}%")
        else:
            lines.append("  Insufficient data for bandwidth saving calculation.")
    else:
        lines.append("  No suppression stats available.")
    lines.append("")

    # 4. Adaptation latency
    section("4. Adaptation Latency")
    if not cmd_log.empty and "latency_ms" in cmd_log.columns:
        lats = cmd_log["latency_ms"].dropna()
        lats = lats[lats > 0]
        if len(lats) > 0:
            lines.append(f"  Median  (p50) : {np.percentile(lats, 50):>7.0f} ms")
            lines.append(f"  p75           : {np.percentile(lats, 75):>7.0f} ms")
            lines.append(f"  p90           : {np.percentile(lats, 90):>7.0f} ms")
            lines.append(f"  p95           : {np.percentile(lats, 95):>7.0f} ms")
            lines.append(f"  N events     : {len(cmd_log):>7,}")
    else:
        lines.append("  No command log available.")
    lines.append("")

    # 5. Command distribution
    section("5. Command Distribution")
    if not cmd_log.empty and "command" in cmd_log.columns:
        total_cmds = len(cmd_log)
        lines.append(f"  {'Command':<24}  {'Count':>7}  {'%':>7}")
        lines.append(f"  {'─'*24}  {'─'*7}  {'─'*7}")
        for cmd, cnt in cmd_log["command"].value_counts().items():
            pct = cnt / total_cmds * 100
            lines.append(f"  {str(cmd):<24}  {cnt:>7,}  {pct:>6.1f}%")
    else:
        lines.append("  No command log available.")
    lines.append("")

    # 6. Network state coverage
    section("6. Network State Coverage")
    for lbl, df in [("Baseline", b_tel), ("Closed-Loop", cl_tel)]:
        if df.empty:
            continue
        lines.append(f"  {lbl}:")
        total_rows = len(df)
        for state in ("Stable", "Unstable", "Critical"):
            n = int((df["network_condition"] == state).sum())
            pct = n / total_rows * 100 if total_rows > 0 else 0.0
            lines.append(f"    {state:<12}: {n:>5,} windows ({pct:.1f}%)")
    lines.append("")

    # 7. ML vs heuristic agreement
    section("7. ML vs Heuristic Agreement  (Closed-Loop)")
    if agreement_rate is not None:
        lines.append(f"  Overall agreement rate : {agreement_rate:.1f}%")
        lines.append(f"  (windows where ML predictor matched rule-based heuristic)")
    else:
        lines.append("  Not computed (no closed-loop telemetry).")
    lines.append("")

    hr("=")
    lines.append("")

    report = "\n".join(lines)
    report_path = os.path.join(out_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also print to terminal
    print("\n" + report)
    print(f"  [OK] summary_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 10 evaluation figures + summary report"
    )
    parser.add_argument("--baseline-dir",   required=True,
                        help="Path to baseline run output directory")
    parser.add_argument("--closedloop-dir", required=True,
                        help="Path to closed-loop run output directory")
    parser.add_argument("--output-dir",     required=True,
                        help="Directory to write PNGs and summary_report.txt")
    parser.add_argument("--model", default=None,
                        help="Path to trained .pkl model for offline ML re-prediction (Fig 09)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[ANALYZE] Loading data...")
    print(f"  Baseline    : {args.baseline_dir}")
    print(f"  Closed-Loop : {args.closedloop_dir}")
    print(f"  Output      : {args.output_dir}")
    print()

    b_tel    = load_telemetry(args.baseline_dir)
    cl_tel   = load_telemetry(args.closedloop_dir)
    b_stats  = load_sender_stats(args.baseline_dir)
    cl_stats = load_sender_stats(args.closedloop_dir)
    cmd_log  = load_command_log(args.closedloop_dir)

    # Offline ML re-prediction for Fig 09
    ml_predictions = None
    if args.model:
        print(f"[ANALYZE] Running offline ML re-prediction with model: {args.model}")
        ml_predictions = ml_predict_series(cl_tel, args.model)
        if ml_predictions is not None:
            print(f"  [MODEL] Re-predicted {len(ml_predictions):,} windows offline")
        else:
            print("  [WARN] Offline ML prediction failed — Fig 09 will use recorded state")
    else:
        print("[ANALYZE] No --model provided; Fig 09 will compare recorded state vs heuristic")

    # Diagnostics
    print(f"[ANALYZE] Data loaded:")
    print(f"  Baseline  telemetry  : {len(b_tel):,} rows")
    print(f"  Closed-loop telemetry: {len(cl_tel):,} rows")
    print(f"  Baseline  stats      : {len(b_stats):,} rows")
    print(f"  Closed-loop stats    : {len(cl_stats):,} rows")
    print(f"  Command log          : {len(cmd_log):,} rows")
    print()

    if b_tel.empty and cl_tel.empty:
        print("[ERROR] Both telemetry files are empty — nothing to plot.")
        print("        Run the baseline and closed-loop experiments first.")
        sys.exit(1)

    print("[ANALYZE] Generating figures...")
    fig01_bandwidth_over_time(b_tel, cl_tel, args.output_dir)
    fig02_packet_loss_comparison(b_tel, cl_tel, args.output_dir)
    fig03_semantic_suppression(cl_stats, args.output_dir)
    latency_p50 = fig04_adaptation_latency(cmd_log, args.output_dir)
    fig05_command_distribution(cmd_log, args.output_dir)
    fig06_clinical_delivery(b_tel, cl_tel, b_stats, cl_stats, args.output_dir)
    savings = fig07_bandwidth_savings_bar(b_tel, cl_tel, args.output_dir)
    fig08_telemetry_timeline(b_tel, cl_tel, args.output_dir)
    agreement = fig09_network_state_agreement(cl_tel, args.output_dir, ml_predictions=ml_predictions)
    fig10_suppression_over_time(cl_tel, cmd_log, args.output_dir)

    generate_summary_report(
        b_tel, cl_tel, b_stats, cl_stats, cmd_log,
        savings, agreement, latency_p50, args.output_dir,
    )

    print(f"\n[ANALYZE] All 10 figures + summary_report.txt written to:")
    print(f"          {args.output_dir}")


if __name__ == "__main__":
    main()
