#!/usr/bin/env python3
"""
analyze_results.py  —  AI-Native Semantic Network Control
==========================================================
Modular figure generation for baseline vs closed-loop evaluation.

Usage
-----
  python3 analyze_results.py \\
      --baseline-dir   outputs/evaluation/baseline_natural_YYYYMMDD \\
      --closedloop-dir outputs/evaluation/closedloop_natural_YYYYMMDD \\
      --output-dir     outputs/figures \\
      [--no-pdf] [--dpi 200] [--format png pdf] [--figures 1 2 3]

Each figure function is independent. Pass --figures to generate a subset.
If only one run directory is provided, single-run figures are still produced.

Figure Catalogue
----------------
  1   Network metrics timeline (3-panel: loss / delay / jitter)
  2   Throughput timeline with loss overlay + state strip
  3   Violin distributions by network state
  4   ML confidence KDE distribution (the "normal distribution" figure)
  5   ML confidence density heatmap over time
  6   Adaptation latency distribution (histogram + per-command KDE)
  7   Baseline vs closed-loop packet loss comparison
  8   Baseline vs closed-loop throughput savings bar chart
  9   Network state agreement: ML predictor vs heuristic
  10  Composite 6-panel summary figure
  11  Command distribution pie chart  (closed-loop only)
  12  Semantic suppression counters   (closed-loop only)

Author: ANSC Research
"""

# ── Standard library ───────────────────────────────────────────────────────
import argparse
import glob
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba
from scipy import stats
from scipy.ndimage import uniform_filter1d

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Global Style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_style(dpi: int = 180) -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Georgia", "Times New Roman"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        dpi,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#fafafa",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.55,
        "grid.linestyle":    "--",
        "lines.linewidth":   1.6,
        "savefig.dpi":       dpi,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


# ── Colour constants (edit here to retheme everything) ─────────────────────
STATE_COLORS = {
    "Stable":   "#2ecc71",
    "Unstable": "#f39c12",
    "Critical": "#e74c3c",
}
STATE_PALETTE = {          # slightly darker for lines / text
    "Stable":   "#27ae60",
    "Unstable": "#e67e22",
    "Critical": "#c0392b",
}
STATE_ORDER = ["Stable", "Unstable", "Critical"]
STATE_ALPHA = 0.12         # background shading opacity

RUN_COLORS = {
    "baseline":   "#7570b3",
    "closedloop": "#1b9e77",
}

COMMAND_COLORS = {
    "FULL_ECG":           "#1f77b4",
    "FULL_ECG_PRIORITY":  "#17becf",
    "DOWNSAMPLED_ECG":    "#ff7f0e",
    "SEMANTIC_ALERT":     "#e377c2",
    "SEMANTIC_CRITICAL":  "#d62728",
    "SEMANTIC_SUMMARY":   "#7f8c8d",
}

SCENARIO_PHASES = [
    (0,   80,  "Morning\nStable"),
    (80,  160, "Gradual\nDegradation"),
    (160, 200, "Partial\nRecovery"),
    (200, 280, "Escalation"),
    (280, 360, "Peak\nCritical"),
    (360, 430, "Slow\nRecovery"),
    (430, 500, "Stable\nRecovery"),
    (500, 530, "Burst"),
    (530, 570, "Fast\nRecovery"),
    (570, 700, "Stable\nFinish"),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_telemetry(run_dir: str) -> pd.DataFrame:
    """
    Load network_telemetry.csv.
    Adds 'elapsed' (seconds from first timestamp) and smoothed columns.
    Returns empty DataFrame if file missing.
    """
    path = os.path.join(run_dir, "network_telemetry.csv")
    if not os.path.exists(path):
        print(f"  [WARN] telemetry not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["elapsed"] = df["timestamp"] - df["timestamp"].iloc[0]

    num_cols = ["packet_loss_rate", "avg_delay", "jitter",
                "throughput_bps", "bandwidth_usage_bps",
                "active_devices", "packets_per_window"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "prediction_confidence" in df.columns:
        df["prediction_confidence"] = pd.to_numeric(
            df["prediction_confidence"], errors="coerce")

    if "network_condition" not in df.columns and "network_state" in df.columns:
        df.rename(columns={"network_state": "network_condition"}, inplace=True)
    if "network_condition" not in df.columns:
        df["network_condition"] = "Stable"

    # Smoothed columns (9-point centred rolling average)
    W = 9
    df_act = df[df["throughput_bps"] > 0].copy()
    for col in ["packet_loss_rate", "avg_delay", "jitter",
                "throughput_bps", "bandwidth_usage_bps"]:
        if col in df.columns:
            df[f"{col}_sm"] = (df[col]
                               .rolling(W, center=True, min_periods=1)
                               .mean())
            if col in df_act.columns:
                df_act[f"{col}_sm"] = (df_act[col]
                                       .rolling(W, center=True, min_periods=1)
                                       .mean())
    df._active = df_act          # stash active-only subset on the object

    return df


def load_command_log(run_dir: str) -> pd.DataFrame:
    """Load command_log.csv. Returns empty DataFrame if file missing."""
    path = os.path.join(run_dir, "command_log.csv")
    if not os.path.exists(path):
        print(f"  [WARN] command_log not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df["timestamp"]  = pd.to_numeric(df["timestamp"],  errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_sender_stats(run_dir: str) -> pd.DataFrame:
    """Aggregate all sender_semantic_stats_dev*.csv files."""
    pattern = os.path.join(run_dir, "sender_semantic_stats_dev*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  [WARN] no sender stats in: {run_dir}")
        return pd.DataFrame()

    frames = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp, low_memory=False))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    int_cols = ["total_samples", "total_sent", "total_suppressed",
                "raw_sent", "raw_suppressed",
                "delta_sent", "delta_suppressed",
                "summary_sent", "summary_suppressed",
                "critical_only_sent", "critical_only_suppressed"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def parse_receiver_log(run_dir: str) -> pd.DataFrame:
    """
    Parse PREDICT lines from receiver.log into a DataFrame.
    Columns: source, state, loss_pct, delay_ms, jitter_ms, devices, conf
    """
    path = os.path.join(run_dir, "receiver.log")
    if not os.path.exists(path):
        return pd.DataFrame()

    pat = (r'\[PREDICT\] source=(\S+) state=(\S+) '
           r'loss=([\d.]+)% delay=([\d.]+)ms jitter=([\d.]+)ms '
           r'devices=(\d+)(?:\s+conf=([\d.]+))?')
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(pat, line)
            if m:
                rows.append({
                    "source":    m.group(1),
                    "state":     m.group(2),
                    "loss_pct":  float(m.group(3)),
                    "delay_ms":  float(m.group(4)),
                    "jitter_ms": float(m.group(5)),
                    "devices":   int(m.group(6)),
                    "conf":      float(m.group(7)) if m.group(7) else np.nan,
                })
    return pd.DataFrame(rows)


def heuristic_state(loss: float, delay_ms: float, jitter_ms: float) -> str:
    """Rule-based classifier mirroring common.py choose_network_state."""
    if loss >= 0.08 or delay_ms >= 130.0 or jitter_ms >= 35.0:
        return "Critical"
    if loss >= 0.03 or delay_ms >= 35.0 or jitter_ms >= 8.0:
        return "Unstable"
    return "Stable"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Shared Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def state_legend_patches(alpha: float = 0.55) -> List[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor=STATE_COLORS[s], alpha=alpha,
                       label=s, edgecolor="#888", linewidth=0.5)
        for s in STATE_ORDER
    ]


def shade_states(ax: plt.Axes, df: pd.DataFrame,
                 x_col: str = "elapsed",
                 alpha: float = STATE_ALPHA) -> None:
    """Paint translucent background strips by network_condition."""
    if df.empty or "network_condition" not in df.columns:
        return
    xs  = df[x_col].values
    sts = df["network_condition"].values
    px, ps = xs[0], sts[0]
    for x, s in zip(xs[1:], sts[1:]):
        if s != ps:
            ax.axvspan(px, x, alpha=alpha,
                       color=STATE_COLORS.get(ps, "#ccc"), linewidth=0)
            px, ps = x, s
    ax.axvspan(px, xs[-1], alpha=alpha,
               color=STATE_COLORS.get(ps, "#ccc"), linewidth=0)


def add_phase_vlines(ax: plt.Axes,
                     phases: List[Tuple] = SCENARIO_PHASES,
                     top_labels: bool = True) -> None:
    """Draw dotted phase separator lines and optional labels on top axis."""
    for i, (s, e, label) in enumerate(phases):
        if i > 0:
            ax.axvline(s, color="#bbb", linewidth=0.55,
                       linestyle=":", zorder=1)
        if top_labels:
            mid = (s + e) / 2
            ylim = ax.get_ylim()
            ypos = ylim[1] * 0.93 if ylim[1] > 0 else 5
            ax.text(mid, ypos, label.replace("\n", " "),
                    ha="center", va="top", fontsize=6.2,
                    color="#555", fontstyle="italic", clip_on=True)


def save_fig(fig: plt.Figure, out_dir: str,
             name: str, formats: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"))
    plt.close(fig)
    print(f"  ✓ {name}  ({', '.join(formats)})")


def active(df: pd.DataFrame) -> pd.DataFrame:
    """Return the active-only subset (throughput > 0)."""
    if hasattr(df, "_active"):
        return df._active
    return df[df.get("throughput_bps", pd.Series(dtype=float)) > 0].copy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Figure Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Figure 1: Network metrics timeline ────────────────────────────────────
def fig1_network_timeline(df: pd.DataFrame, out_dir: str,
                          formats: List[str], run_label: str = "Closed-Loop",
                          phases: bool = True) -> None:
    """3-panel timeline: packet loss / avg delay / jitter."""
    df_a = active(df)
    if df_a.empty:
        print("  [SKIP] fig1: no active rows")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.07)

    metrics = [
        ("packet_loss_rate_sm", "Packet Loss Rate (%)",
         "#c0392b", lambda v: v * 100),
        ("avg_delay_sm",        "Avg. Delay (ms)",
         "#2980b9", lambda v: v),
        ("jitter_sm",           "Jitter (ms)",
         "#8e44ad", lambda v: v),
    ]
    for ax, (col, ylabel, color, tfm) in zip(axes, metrics):
        shade_states(ax, df_a)
        col_sm = col if col in df_a.columns else col.replace("_sm", "")
        y = tfm(df_a[col_sm].values)
        ax.fill_between(df_a["elapsed"], 0, y, alpha=0.18, color=color)
        ax.plot(df_a["elapsed"], y, color=color, linewidth=1.5, zorder=3)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_ylim(bottom=0)

    if phases:
        add_phase_vlines(axes[0], top_labels=True)
        for ax in axes[1:]:
            add_phase_vlines(ax, top_labels=False)

    axes[0].set_title(
        f"{run_label} — Network Metrics over Time", pad=8)
    axes[0].legend(handles=state_legend_patches(), loc="upper right",
                   ncol=3, fontsize=8.5, framealpha=0.9)
    axes[-1].set_xlabel("Elapsed Time (s)", labelpad=4)
    axes[-1].set_xlim(0, df["elapsed"].max())

    save_fig(fig, out_dir, "fig1_network_timeline", formats)


# ── Figure 2: Throughput timeline with loss overlay + state strip ──────────
def fig2_throughput_timeline(df: pd.DataFrame, out_dir: str,
                              formats: List[str],
                              run_label: str = "Closed-Loop",
                              phases: bool = True) -> None:
    df_a = active(df)
    if df_a.empty:
        print("  [SKIP] fig2: no active rows")
        return

    fig, (ax1, ax_strip) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [5, 1]})
    fig.subplots_adjust(hspace=0.05)

    shade_states(ax1, df_a)
    thr = df_a.get("throughput_bps_sm", df_a.get("throughput_bps", 0)) / 1000
    bw  = df_a.get("bandwidth_usage_bps", pd.Series(
                   np.zeros(len(df_a)), index=df_a.index)).rolling(
                   9, center=True, min_periods=1).mean() / 1000

    ax1.fill_between(df_a["elapsed"], thr, alpha=0.22, color="#2980b9")
    ax1.plot(df_a["elapsed"], thr,  color="#2980b9",
             linewidth=1.8, label="Network Load (kbps)")
    ax1.plot(df_a["elapsed"], bw,   color="#7f8c8d",
             linewidth=0.9, linestyle="--", alpha=0.7,
             label="Bandwidth usage (kbps)")

    ax1_r = ax1.twinx()
    loss_col = ("packet_loss_rate_sm"
                if "packet_loss_rate_sm" in df_a.columns
                else "packet_loss_rate")
    ax1_r.plot(df_a["elapsed"], df_a[loss_col] * 100,
               color="#e74c3c", linewidth=1.1, linestyle=":",
               alpha=0.75, label="Loss %")
    ax1_r.set_ylabel("Packet Loss (%)", color="#e74c3c", labelpad=4)
    ax1_r.tick_params(axis="y", labelcolor="#e74c3c")
    ax1_r.spines["top"].set_visible(False)
    ax1_r.set_ylim(0, None)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right",
               framealpha=0.92, fontsize=8.5, ncol=3)
    ax1.set_ylabel("Network Load / BW (kbps)")
    ax1.set_title(f"{run_label} — Network Load over Time", pad=8)

    if phases:
        add_phase_vlines(ax1, top_labels=True)

    # State strip
    for _, row in df.iterrows():
        s = row.get("network_condition", "Stable")
        ax_strip.axvspan(row["elapsed"], row["elapsed"] + 0.28,
                         ymin=0, ymax=1,
                         color=STATE_COLORS.get(s, "#ccc"),
                         alpha=0.78, linewidth=0)
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("State", fontsize=8.5)
    ax_strip.set_xlabel("Elapsed Time (s)")
    ax_strip.set_xlim(0, df["elapsed"].max())
    ax_strip.grid(False)
    for i, s in enumerate(STATE_ORDER):
        ax_strip.text(10 + i * 160, 0.5, s, ha="left", va="center",
                      color="white", fontsize=8, fontweight="bold")

    save_fig(fig, out_dir, "fig2_throughput_timeline", formats)


# ── Figure 3: Violin distributions by state ────────────────────────────────
def fig3_violin_distributions(df: pd.DataFrame, out_dir: str,
                               formats: List[str],
                               run_label: str = "Closed-Loop") -> None:
    df_a = active(df)
    if df_a.empty:
        print("  [SKIP] fig3: no active rows")
        return

    metrics = [
        ("packet_loss_rate", "Packet Loss (%)",   lambda v: v * 100),
        ("avg_delay",        "Avg. Delay (ms)",   lambda v: v),
        ("jitter",           "Jitter (ms)",        lambda v: v),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    np.random.seed(42)

    for ax, (col, ylabel, tfm) in zip(axes, metrics):
        data_by_state, cols_list = [], []
        for s in STATE_ORDER:
            sub = df_a[df_a["network_condition"] == s][col].dropna()
            vals = tfm(sub.values)
            data_by_state.append(vals[vals >= 0])
            cols_list.append(STATE_PALETTE[s])

        non_empty = [i for i, d in enumerate(data_by_state) if len(d) >= 4]
        if not non_empty:
            continue

        parts = ax.violinplot(
            [data_by_state[i] for i in non_empty],
            positions=non_empty, widths=0.65,
            showmeans=True, showmedians=False, showextrema=False)

        for pc, i in zip(parts["bodies"], non_empty):
            pc.set_facecolor(cols_list[i])
            pc.set_alpha(0.50)
            pc.set_edgecolor(cols_list[i])
        parts["cmeans"].set_color("#2c3e50")
        parts["cmeans"].set_linewidth(2.2)

        for i in non_empty:
            d = data_by_state[i]
            if len(d) < 4:
                continue
            q25, q75 = np.percentile(d, [25, 75])
            ax.vlines(i, q25, q75, color=cols_list[i],
                      linewidth=4.5, alpha=0.85, zorder=3)
            n_show  = min(100, len(d))
            idx     = np.random.choice(len(d), n_show, replace=False)
            jit_x   = np.random.normal(i, 0.065, n_show)
            ax.scatter(jit_x, d[idx], color=cols_list[i],
                       s=4, alpha=0.28, zorder=2, rasterized=True)

        ax.set_xticks(range(3))
        ax.set_xticklabels(STATE_ORDER, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip())

    patches = [mpatches.Patch(facecolor=STATE_PALETTE[s],
                              alpha=0.65, label=s) for s in STATE_ORDER]
    fig.legend(handles=patches, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9, fontsize=9.5)
    fig.suptitle(
        f"{run_label} — Network Metric Distributions by Operating State\n"
        "(filled bar = IQR, diamond = mean)",
        y=1.07, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig3_violin_distributions", formats)


# ── Figure 4: ML Confidence KDE (the "normal distribution" figure) ─────────
def fig4_confidence_kde(df: pd.DataFrame, out_dir: str,
                         formats: List[str],
                         run_label: str = "Closed-Loop") -> None:
    if "prediction_confidence" not in df.columns:
        print("  [SKIP] fig4: no prediction_confidence column")
        return

    conf_data: Dict[str, np.ndarray] = {}
    for s in STATE_ORDER:
        sub = df[df["network_condition"] == s]["prediction_confidence"].dropna()
        sub = sub[(sub >= 0) & (sub <= 1)]
        conf_data[s] = sub.values

    if all(len(v) == 0 for v in conf_data.values()):
        print("  [SKIP] fig4: no confidence data")
        return

    fig, (ax_kde, ax_ts) = plt.subplots(1, 2, figsize=(12, 5.2))
    x_range = np.linspace(0.35, 1.02, 600)

    # Left: KDE per state
    for s in STATE_ORDER:
        vals = conf_data[s]
        if len(vals) < 5:
            continue
        try:
            kde = stats.gaussian_kde(vals, bw_method=0.08)
            y   = kde(x_range)
            ax_kde.plot(x_range, y, color=STATE_PALETTE[s],
                        linewidth=2.2, label=s, zorder=3)
            ax_kde.fill_between(x_range, y, alpha=0.17,
                                color=STATE_PALETTE[s])
            m = np.mean(vals)
            ax_kde.axvline(m, color=STATE_PALETTE[s], linewidth=0.9,
                           linestyle="--", alpha=0.65)
            ax_kde.text(m - 0.003,
                        kde(np.array([m]))[0] * 0.6,
                        f"$\\mu$={m:.3f}", rotation=90,
                        ha="right", va="bottom",
                        color=STATE_PALETTE[s], fontsize=7.5)
        except Exception:
            pass

    ax_kde.set_xlabel("ML Predictor Confidence Score")
    ax_kde.set_ylabel("Probability Density")
    ax_kde.set_title("Confidence Distribution\nby Network State")
    ax_kde.set_xlim(0.35, 1.02)
    ax_kde.legend(handles=state_legend_patches(), framealpha=0.9)

    # Right: confidence scatter over time, coloured by state
    df_c = df[df["prediction_confidence"].notna()].copy()
    df_act = active(df)
    if not df_act.empty:
        df_c = df_c[df_c["throughput_bps"] > 0]

    for s in STATE_ORDER:
        sub = df_c[df_c["network_condition"] == s]
        ax_ts.scatter(sub["elapsed"], sub["prediction_confidence"],
                      color=STATE_PALETTE[s], s=2.5, alpha=0.40,
                      rasterized=True, label=s, zorder=2)

    roll = df_c["prediction_confidence"].rolling(15, min_periods=1).mean()
    ax_ts.plot(df_c["elapsed"], roll, color="#2c3e50",
               linewidth=1.4, zorder=5, label="Rolling mean")
    ax_ts.set_xlabel("Elapsed Time (s)")
    ax_ts.set_ylabel("Confidence Score")
    ax_ts.set_title("Confidence over Time\n(Coloured by State)")
    ax_ts.set_ylim(0.30, 1.02)
    ax_ts.set_xlim(0, df["elapsed"].max())
    patches2 = state_legend_patches() + [
        mpatches.Patch(facecolor="#2c3e50", label="Rolling mean")]
    ax_ts.legend(handles=patches2, framealpha=0.9, fontsize=8)

    fig.suptitle(
        f"{run_label} — XGBoost Classifier Confidence Analysis",
        y=1.01, fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig4_confidence_kde", formats)


# ── Figure 5: Confidence density heatmap over time ─────────────────────────
def fig5_confidence_heatmap(df: pd.DataFrame, out_dir: str,
                             formats: List[str],
                             run_label: str = "Closed-Loop") -> None:
    if "prediction_confidence" not in df.columns:
        print("  [SKIP] fig5: no prediction_confidence column")
        return

    df_c = df[df["prediction_confidence"].notna()].copy()
    if df_c.empty:
        print("  [SKIP] fig5: no confidence data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                              gridspec_kw={"height_ratios": [4, 1]})
    fig.subplots_adjust(hspace=0.05)
    ax, ax_strip = axes

    x_bins = np.linspace(0, df["elapsed"].max(), 160)
    y_bins = np.linspace(0.30, 1.01, 90)
    H, xe, ye = np.histogram2d(
        df_c["elapsed"], df_c["prediction_confidence"],
        bins=[x_bins, y_bins])
    H = H.T.astype(float)
    H_sm = uniform_filter1d(H, size=3, axis=1)
    H_sm[H_sm < 0.5] = np.nan

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")
    im = ax.pcolormesh(xe, ye, H_sm, cmap=cmap, shading="auto",
                       rasterized=True, vmin=0, vmax=H.max() * 0.65)
    plt.colorbar(im, ax=ax, label="Window count",
                 fraction=0.025, pad=0.01)

    roll_vals = df_c["prediction_confidence"].rolling(20, min_periods=1).mean()
    ax.plot(df_c["elapsed"].values, roll_vals.values,
            color="#2c3e50", linewidth=1.8,
            label="Rolling mean (n=20)", zorder=5)

    ax.set_ylabel("Classifier Confidence")
    ax.set_ylim(0.30, 1.01)
    ax.set_title(f"{run_label} — Prediction Confidence Density", pad=6)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)

    add_phase_vlines(ax, top_labels=False)

    # State strip
    for _, row in df.iterrows():
        s = row.get("network_condition", "Stable")
        ax_strip.axvspan(row["elapsed"], row["elapsed"] + 0.28,
                         ymin=0, ymax=1,
                         color=STATE_COLORS.get(s, "#ccc"),
                         alpha=0.78, linewidth=0)
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("State", fontsize=8.5)
    ax_strip.set_xlabel("Elapsed Time (s)")
    ax_strip.set_xlim(0, df["elapsed"].max())
    ax_strip.grid(False)
    for i, s in enumerate(STATE_ORDER):
        ax_strip.text(10 + i * 160, 0.5, s, ha="left", va="center",
                      color="white", fontsize=8, fontweight="bold")

    save_fig(fig, out_dir, "fig5_confidence_heatmap", formats)


# ── Figure 6: Adaptation latency distribution ──────────────────────────────
def fig6_adaptation_latency(cmd_log: pd.DataFrame, out_dir: str,
                             formats: List[str],
                             run_label: str = "Closed-Loop") -> None:
    """
    Histogram + per-command KDE of command adaptation latency.
    Works with real command_log data; falls back to model-based synthetic
    data when command_log is missing or has < 10 rows.
    """
    WINDOW_MS = 250.0
    SLA_MS    = 1000.0

    def model_latency(n: int, lam: float) -> np.ndarray:
        """Synthetic latency: (1 + Poisson(λ)) × 250ms + Gaussian transport."""
        extra     = np.random.poisson(lam, n).clip(0, 4)
        transport = np.random.normal(3.0, 1.5, n).clip(0.5, 20.0)
        return (1 + extra) * WINDOW_MS + transport

    np.random.seed(42)

    # --- Build per-command data ---
    use_real = (not cmd_log.empty
                and "latency_ms" in cmd_log.columns
                and len(cmd_log) >= 10)

    cmd_datasets: Dict[str, np.ndarray] = {}
    if use_real:
        for cmd in cmd_log["command"].dropna().unique():
            vals = cmd_log[cmd_log["command"] == cmd]["latency_ms"].dropna().values
            if len(vals) >= 3:
                cmd_datasets[cmd] = vals
        print(f"  Using real command_log: {len(cmd_log)} events")
    else:
        print("  command_log absent/small — using model-based synthetic latency")
        cmd_datasets = {
            "FULL_ECG":           model_latency(180, 0.8),
            "DOWNSAMPLED_ECG":    model_latency(90,  1.0),
            "SEMANTIC_ALERT":     model_latency(140, 1.2),
            "SEMANTIC_CRITICAL":  model_latency(110, 0.5),
            "SEMANTIC_SUMMARY":   model_latency(60,  2.5),
        }

    if not cmd_datasets:
        print("  [SKIP] fig6: no latency data")
        return

    ALL = np.concatenate(list(cmd_datasets.values()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: aggregate histogram + KDE + observed ticks
    bins = np.arange(150, min(ALL.max() + 100, 2200), 60)
    ax1.hist(ALL, bins=bins, density=True, color="#b0c4de",
             edgecolor="white", linewidth=0.6, alpha=0.50,
             label="Histogram (all commands)", zorder=2)

    if use_real:
        obs = cmd_log["latency_ms"].dropna().values
        ax1.plot(obs, np.zeros_like(obs) - 0.0001, "|",
                 color="#2c3e50", markersize=10, markeredgewidth=1.5,
                 label="Observed events", zorder=5)

    x_rng = np.linspace(max(100, ALL.min() - 50),
                         min(ALL.max() + 200, 2500), 800)
    try:
        kde_all = stats.gaussian_kde(ALL, bw_method=0.20)
        ax1.plot(x_rng, kde_all(x_rng), color="#2c3e50",
                 linewidth=2.2, label="KDE (aggregate)", zorder=4)
    except Exception:
        kde_all = None

    # Window boundary lines
    mode_colors = ["#27ae60", "#e67e22", "#e74c3c", "#8e44ad"]
    for i, n_win in enumerate([1, 2, 3, 4]):
        xv = n_win * WINDOW_MS + 3
        c  = mode_colors[i]
        ax1.axvline(xv, color=c, linewidth=1.0, linestyle="--",
                    alpha=0.65, zorder=3)
        ax1.text(xv + 8, ax1.get_ylim()[1] * 0.80 if ax1.get_ylim()[1] > 0 else 0.005,
                 f"{n_win}W\n({n_win*250}\u2009ms)",
                 fontsize=7, color=c, va="top", clip_on=True)

    ax1.axvline(SLA_MS, color="#c0392b", linewidth=1.5,
                linestyle="-.", alpha=0.85, zorder=3)
    ax1.text(SLA_MS + 20, ax1.get_ylim()[1] * 0.3 if ax1.get_ylim()[1] > 0 else 0.002,
             f"SLA\n{SLA_MS:.0f}\u2009ms",
             fontsize=8.5, color="#c0392b", fontweight="bold")

    within_pct = np.mean(ALL <= SLA_MS) * 100
    ax1.text(0.03, 0.97, f"Within SLA: {within_pct:.1f}%",
             transform=ax1.transAxes, va="top", fontsize=9,
             color="#2c3e50", bbox=dict(boxstyle="round,pad=0.3",
                                        fc="white", ec="#ccc", alpha=0.85))

    ax1.set_xlabel("Adaptation Latency (ms)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Overall Latency Distribution\n"
                  "(discretised at 250\u2009ms window boundaries)")
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    ax1.set_xlim(100, min(ALL.max() + 200, 2200))

    # Right: per-command KDE
    x_rng2 = np.linspace(100, min(ALL.max() + 200, 2200), 800)
    for cmd, data in cmd_datasets.items():
        if len(data) < 5:
            continue
        try:
            kde = stats.gaussian_kde(data, bw_method=0.22)
            y   = kde(x_rng2)
            c   = COMMAND_COLORS.get(cmd, "#888")
            ax2.plot(x_rng2, y, color=c, linewidth=2.0,
                     label=cmd, zorder=3)
            ax2.fill_between(x_rng2, y, alpha=0.09, color=c)
        except Exception:
            pass

    ax2.axvline(SLA_MS, color="#c0392b", linewidth=1.5,
                linestyle="-.", alpha=0.85, label="SLA (1000\u2009ms)", zorder=4)
    ax2.set_xlabel("Adaptation Latency (ms)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Per-Command Latency KDE")
    ax2.set_xlim(100, min(ALL.max() + 200, 2200))
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=8)

    # Stats printout
    print(f"  Latency stats ({run_label}):")
    for cmd, data in cmd_datasets.items():
        p50  = np.percentile(data, 50)
        p95  = np.percentile(data, 95)
        sla  = np.mean(data <= SLA_MS) * 100
        print(f"    {cmd:<24} p50={p50:.0f}ms  p95={p95:.0f}ms  "
              f"within_SLA={sla:.1f}%")

    src_note = "real command_log" if use_real else "model-based synthetic"
    fig.suptitle(
        f"{run_label} — Policy Adaptation Latency  [{src_note}]\n"
        "Ward Controller → Sender Command Delivery",
        y=1.01, fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig6_adaptation_latency", formats)


# ── Figure 7: Baseline vs CL packet loss comparison ───────────────────────
def fig7_loss_comparison(b_tel: pd.DataFrame, cl_tel: pd.DataFrame,
                          out_dir: str, formats: List[str]) -> None:
    if b_tel.empty and cl_tel.empty:
        print("  [SKIP] fig7: both telemetry empty")
        return

    fig, ax = plt.subplots(figsize=(13, 4.5))
    for df, label, color in [
        (b_tel,  "Baseline",    RUN_COLORS["baseline"]),
        (cl_tel, "Closed-Loop", RUN_COLORS["closedloop"]),
    ]:
        if df.empty:
            continue
        col = ("packet_loss_rate_sm"
               if "packet_loss_rate_sm" in df.columns
               else "packet_loss_rate")
        ax.plot(df["elapsed"], df[col] * 100,
                color=color, linewidth=1.4, label=label, alpha=0.85)

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Packet Loss (%)")
    ax.set_title("Packet Loss: Baseline vs Closed-Loop\n"
                 "(same netem stress scenario, different response)")
    ax.legend(framealpha=0.9)
    save_fig(fig, out_dir, "fig7_loss_comparison", formats)


# ── Figure 8: Baseline vs CL throughput savings bar ───────────────────────
def fig8_throughput_savings(b_tel: pd.DataFrame, cl_tel: pd.DataFrame,
                             out_dir: str, formats: List[str]) -> None:
    if b_tel.empty or cl_tel.empty:
        print("  [SKIP] fig8: need both telemetry files")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b_means, cl_means, savings = [], [], []

    for s in STATE_ORDER:
        b_sub  = b_tel[b_tel["network_condition"] == s]["throughput_bps"]
        cl_sub = cl_tel[cl_tel["network_condition"] == s]["throughput_bps"]
        bm = float(b_sub.mean())  if len(b_sub)  > 0 else 0.0
        cm = float(cl_sub.mean()) if len(cl_sub) > 0 else 0.0
        pct = (bm - cm) / bm * 100 if bm > 0 else 0.0
        b_means.append(bm / 1000)
        cl_means.append(cm / 1000)
        savings.append(pct)

    x = np.arange(len(STATE_ORDER))
    w = 0.35
    bars_b  = ax.bar(x - w / 2, b_means,  w, label="Baseline",
                     color=RUN_COLORS["baseline"],   alpha=0.88)
    bars_cl = ax.bar(x + w / 2, cl_means, w, label="Closed-Loop",
                     color=RUN_COLORS["closedloop"],  alpha=0.88)

    for xi, (bb, bc, pct) in enumerate(zip(bars_b, bars_cl, savings)):
        y = max(bb.get_height(), bc.get_height()) + max(b_means) * 0.025
        ax.text(x[xi], y, f"{pct:+.1f}%", ha="center", fontsize=9.5,
                color="#c0392b" if pct > 0 else "#27ae60",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_ORDER)
    ax.set_ylabel("Mean Network Load (kbps)")
    ax.set_title("Network Load Savings by Network State\n"
                 "(+ = closed-loop uses less bandwidth)")
    ax.legend(framealpha=0.9)
    save_fig(fig, out_dir, "fig8_throughput_savings", formats)


# ── Figure 9: ML predictor vs heuristic agreement ─────────────────────────
def fig9_ml_vs_heuristic(df: pd.DataFrame, out_dir: str,
                           formats: List[str],
                           run_label: str = "Closed-Loop") -> None:
    if "prediction_confidence" not in df.columns:
        print("  [SKIP] fig9: no ML confidence data")
        return

    cl = df.copy()
    cl["heuristic"] = cl.apply(
        lambda r: heuristic_state(
            r.get("packet_loss_rate", 0),
            r.get("avg_delay", 0),
            r.get("jitter", 0)),
        axis=1)

    pred_col = "network_condition"   # recorded ML prediction
    cl["agree"] = (cl[pred_col] == cl["heuristic"])
    agreement   = float(cl["agree"].mean() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    roll = cl["agree"].astype(float).rolling(8, min_periods=1).mean()
    ax1.fill_between(cl["elapsed"], roll, alpha=0.30,
                     color=RUN_COLORS["closedloop"])
    ax1.plot(cl["elapsed"], roll, color=RUN_COLORS["closedloop"],
             linewidth=1.4, label="Rolling agreement (n=8)")
    disagree = cl[~cl["agree"]]
    if not disagree.empty:
        ax1.scatter(disagree["elapsed"],
                    [0.04] * len(disagree),
                    color="#e74c3c", s=9, alpha=0.65, zorder=4,
                    label="Disagreement windows")
    ax1.axhline(agreement / 100, linestyle="--", color="#2c3e50",
                linewidth=1.1, label=f"Mean: {agreement:.1f}%")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlabel("Elapsed Time (s)")
    ax1.set_ylabel("Agreement (1.0 = match)")
    ax1.set_title(f"ML vs Heuristic Agreement over Time\n"
                  f"(Overall: {agreement:.1f}%)")
    ax1.legend(fontsize=8.5, framealpha=0.9)

    # Confusion matrix
    confusion = np.zeros((3, 3), dtype=int)
    for i, h_s in enumerate(STATE_ORDER):
        for j, m_s in enumerate(STATE_ORDER):
            confusion[i, j] = int(
                ((cl["heuristic"] == h_s) & (cl[pred_col] == m_s)).sum())

    im = ax2.imshow(confusion, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(STATE_ORDER, fontsize=9)
    ax2.set_yticklabels(STATE_ORDER, fontsize=9)
    ax2.set_xlabel("ML Predicted State")
    ax2.set_ylabel("Heuristic State")
    ax2.set_title(f"Confusion Matrix\n(agreement={agreement:.1f}%)")
    cmax = max(confusion.max(), 1)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, str(confusion[i, j]),
                     ha="center", va="center", fontsize=9.5,
                     color="white" if confusion[i, j] > cmax * 0.5 else "black")
    plt.colorbar(im, ax=ax2)

    fig.suptitle(f"{run_label} — Network State: ML Predictor vs Rule-Based Heuristic",
                 y=1.01, fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig9_ml_vs_heuristic", formats)


# ── Figure 10: Composite 6-panel summary ──────────────────────────────────
def fig10_composite_summary(df: pd.DataFrame, out_dir: str,
                              formats: List[str],
                              run_label: str = "Closed-Loop") -> None:
    df_a = active(df)
    if df_a.empty:
        print("  [SKIP] fig10: no active rows")
        return

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.36)

    conf_data: Dict[str, np.ndarray] = {}
    for s in STATE_ORDER:
        sub = df[df["network_condition"] == s].get(
            "prediction_confidence", pd.Series(dtype=float)).dropna()
        conf_data[s] = sub[(sub >= 0) & (sub <= 1)].values

    # (A) Packet loss timeline
    ax_a = fig.add_subplot(gs[0, :2])
    shade_states(ax_a, df_a, alpha=0.14)
    col = ("packet_loss_rate_sm" if "packet_loss_rate_sm" in df_a.columns
           else "packet_loss_rate")
    ax_a.fill_between(df_a["elapsed"], df_a[col] * 100, alpha=0.22, color="#c0392b")
    ax_a.plot(df_a["elapsed"], df_a[col] * 100, color="#c0392b", linewidth=1.5)
    ax_a.set_ylabel("Loss (%)")
    ax_a.set_xlabel("Elapsed Time (s)")
    ax_a.set_title("(A) Packet Loss", loc="left")
    ax_a.set_xlim(0, df["elapsed"].max())
    add_phase_vlines(ax_a, top_labels=False)

    # (B) Delay
    ax_b = fig.add_subplot(gs[1, :2])
    shade_states(ax_b, df_a, alpha=0.14)
    col2 = ("avg_delay_sm" if "avg_delay_sm" in df_a.columns else "avg_delay")
    ax_b.fill_between(df_a["elapsed"], df_a[col2], alpha=0.22, color="#2980b9")
    ax_b.plot(df_a["elapsed"], df_a[col2], color="#2980b9", linewidth=1.5)
    ax_b.set_ylabel("Delay (ms)")
    ax_b.set_xlabel("Elapsed Time (s)")
    ax_b.set_title("(B) Average Delay", loc="left")
    ax_b.set_xlim(0, df["elapsed"].max())
    add_phase_vlines(ax_b, top_labels=False)

    # (C) Throughput
    ax_c = fig.add_subplot(gs[2, :2])
    shade_states(ax_c, df_a, alpha=0.14)
    col3 = ("throughput_bps_sm" if "throughput_bps_sm" in df_a.columns
            else "throughput_bps")
    ax_c.fill_between(df_a["elapsed"], df_a[col3] / 1000, alpha=0.22, color="#27ae60")
    ax_c.plot(df_a["elapsed"], df_a[col3] / 1000, color="#27ae60", linewidth=1.5)
    ax_c.set_ylabel("Network Load (kbps)")
    ax_c.set_xlabel("Elapsed Time (s)")
    ax_c.set_title("(C) Network Load", loc="left")
    ax_c.set_xlim(0, df["elapsed"].max())
    add_phase_vlines(ax_c, top_labels=False)

    # (D) Confidence KDE
    ax_d = fig.add_subplot(gs[0, 2])
    x_rng = np.linspace(0.35, 1.02, 400)
    for s in STATE_ORDER:
        vals = conf_data[s]
        if len(vals) < 5:
            continue
        try:
            kde = stats.gaussian_kde(vals, bw_method=0.08)
            ax_d.plot(x_rng, kde(x_rng), color=STATE_PALETTE[s],
                      linewidth=2.0, label=s)
            ax_d.fill_between(x_rng, kde(x_rng), alpha=0.18,
                               color=STATE_PALETTE[s])
        except Exception:
            pass
    ax_d.set_xlabel("Confidence")
    ax_d.set_ylabel("Density")
    ax_d.set_title("(D) ML Confidence\nby State", loc="left")
    ax_d.set_xlim(0.35, 1.02)
    ax_d.legend(handles=state_legend_patches(), fontsize=7.5, framealpha=0.9)

    # (E) Time-fraction pie
    ax_e = fig.add_subplot(gs[1, 2])
    counts = [len(df[df["network_condition"] == s]) for s in STATE_ORDER]
    wedge_colors = [STATE_COLORS[s] for s in STATE_ORDER]
    wedges, texts, autos = ax_e.pie(
        counts, labels=STATE_ORDER, autopct="%1.1f%%",
        colors=wedge_colors, startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for t in autos: t.set_fontsize(8.5)
    for t in texts: t.set_fontsize(8.5)
    ax_e.set_title("(E) Time Fraction\nby State", loc="left")

    # (F) Box: delay per state
    ax_f = fig.add_subplot(gs[2, 2])
    data_delay = [df_a[df_a["network_condition"] == s]["avg_delay"].dropna().values
                  for s in STATE_ORDER]
    bp = ax_f.boxplot(data_delay, positions=[0, 1, 2], widths=0.5,
                       patch_artist=True, showfliers=False,
                       medianprops={"color": "white", "linewidth": 2})
    for patch, s in zip(bp["boxes"], STATE_ORDER):
        patch.set_facecolor(STATE_PALETTE[s])
        patch.set_alpha(0.72)
    for w in bp["whiskers"]: w.set(color="#888", linewidth=0.9)
    for c in bp["caps"]:     c.set(color="#888", linewidth=0.9)
    ax_f.set_xticks([0, 1, 2])
    ax_f.set_xticklabels(STATE_ORDER, fontsize=8.5)
    ax_f.set_ylabel("Delay (ms)")
    ax_f.set_title("(F) Delay IQR\nby State", loc="left")

    # Shared legend
    fig.legend(handles=state_legend_patches(),
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.38, 0.995),
               framealpha=0.92, fontsize=9.5)

    fig.suptitle(
        f"{run_label} — Closed-Loop Performance Summary\n"
        "Natural Hospital Ward Stress Scenario",
        y=1.01, fontsize=12, fontweight="bold")
    save_fig(fig, out_dir, "fig10_composite_summary", formats)


# ── Figure 11: Command distribution pie (closed-loop only) ────────────────
def fig11_command_distribution(cmd_log: pd.DataFrame, out_dir: str,
                                formats: List[str],
                                run_label: str = "Closed-Loop") -> None:
    if cmd_log.empty or "command" not in cmd_log.columns:
        print("  [SKIP] fig11: no command log")
        return

    counts = cmd_log["command"].value_counts()
    colors = [COMMAND_COLORS.get(c, "#aaaaaa") for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autos = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.80,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for t in autos: t.set_fontsize(9)
    ax.set_title(
        f"{run_label} — Policy Command Distribution\n"
        "(9-case semantic policy in action)",
        fontsize=11, fontweight="bold")
    save_fig(fig, out_dir, "fig11_command_distribution", formats)


# ── Figure 12: Semantic suppression counters ──────────────────────────────
def fig12_semantic_suppression(stats: pd.DataFrame, out_dir: str,
                                formats: List[str],
                                run_label: str = "Closed-Loop") -> None:
    if stats.empty:
        print("  [SKIP] fig12: no sender stats")
        return

    # Take last row per device
    last = (stats.sort_values("timestamp")
            .groupby(["device_id", "device_type"], as_index=False)
            .last())

    modes   = ["raw", "delta", "summary", "critical_only"]
    labels  = ["RAW", "DELTA", "SUMMARY", "CRITICAL_ONLY"]
    mcolors = ["#1f77b4", "#ff7f0e", "#7f8c8d", "#d62728"]

    sent_t = [int(last.get(f"{m}_sent",       pd.Series([0])).sum()) for m in modes]
    supp_t = [int(last.get(f"{m}_suppressed", pd.Series([0])).sum()) for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x  = np.arange(len(modes))
    w  = 0.38
    ax1.bar(x - w / 2, sent_t, w, label="Sent",        color="#1b9e77", alpha=0.88)
    ax1.bar(x + w / 2, supp_t, w, label="Suppressed",  color="#d95f02", alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Packet Count")
    ax1.set_title("Sent vs Suppressed by Tx Mode")
    ax1.legend(framealpha=0.9)

    nonzero = [(l, s, c) for l, s, c in zip(labels, sent_t, mcolors) if s > 0]
    if nonzero:
        lbs, szs, cls_ = zip(*nonzero)
        ax2.pie(szs, labels=lbs, autopct="%1.1f%%", colors=cls_,
                startangle=140, pctdistance=0.80,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title("Sent Packets by Tx Mode")

    fig.suptitle(f"{run_label} — Semantic Suppression Statistics",
                 y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_dir, "fig12_semantic_suppression", formats)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — Summary Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_summary_report(
    b_tel: pd.DataFrame,
    cl_tel: pd.DataFrame,
    b_stats: pd.DataFrame,
    cl_stats: pd.DataFrame,
    cmd_log: pd.DataFrame,
    out_dir: str,
) -> None:
    W = 66
    lines: List[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)
    def hr(c: str = "─") -> None:
        emit(c * W)
    def section(title: str) -> None:
        emit(f"\n── {title} " + "─" * (W - len(title) - 4))

    hr("═")
    emit("  AI-Native Semantic Network Control — Results Summary")
    hr("═")

    def safe_mean(df, col):
        if df.empty or col not in df.columns:
            return float("nan")
        return float(df[col].mean())

    section("1. Network Metrics")
    for label, df in [("Baseline", b_tel), ("Closed-Loop", cl_tel)]:
        if df.empty:
            continue
        dur = df["elapsed"].max() if "elapsed" in df.columns else 0
        emit(f"  {label} ({dur:.0f}s):")
        emit(f"    Loss  : {safe_mean(df,'packet_loss_rate')*100:.2f}%")
        emit(f"    Delay : {safe_mean(df,'avg_delay'):.1f} ms")
        emit(f"    Jitter: {safe_mean(df,'jitter'):.1f} ms")
        emit(f"    Load  : {safe_mean(df,'throughput_bps')/1000:.1f} kbps")
        emit(f"    State dist: " +
             ", ".join(f"{s}={len(df[df['network_condition']==s])}"
                       for s in STATE_ORDER))

    section("2. Network Load Savings (CL vs Baseline)")
    if not b_tel.empty and not cl_tel.empty:
        for s in STATE_ORDER:
            bm = b_tel[b_tel['network_condition']==s]['throughput_bps'].mean()
            cm = cl_tel[cl_tel['network_condition']==s]['throughput_bps'].mean()
            if bm > 0:
                emit(f"  {s:<12}: {(bm-cm)/bm*100:+.1f}% saving"
                     f"  (baseline={bm/1000:.1f}kbps, CL={cm/1000:.1f}kbps)")
    else:
        emit("  Need both telemetry files for comparison.")

    section("3. ML Predictor (Closed-Loop)")
    if not cl_tel.empty and "prediction_confidence" in cl_tel.columns:
        conf = cl_tel["prediction_confidence"].dropna()
        src  = cl_tel.get("predictor_source", pd.Series(dtype=str))
        emit(f"  Model active : {(src=='model').sum()}/{len(cl_tel)} windows")
        emit(f"  Confidence   : mean={conf.mean():.4f}  min={conf.min():.4f}")

    section("4. Adaptation Latency")
    if not cmd_log.empty and "latency_ms" in cmd_log.columns:
        lats = cmd_log["latency_ms"].dropna()
        lats = lats[lats > 0]
        if len(lats) > 0:
            emit(f"  p50={np.percentile(lats,50):.0f}ms  "
                 f"p90={np.percentile(lats,90):.0f}ms  "
                 f"p95={np.percentile(lats,95):.0f}ms  "
                 f"N={len(cmd_log)}")
            sc = cmd_log[cmd_log['command']=='SEMANTIC_CRITICAL']['latency_ms'].dropna()
            if len(sc) > 0:
                emit(f"  SEMANTIC_CRITICAL SLA (<=1000ms): "
                     f"{np.mean(sc<=1000)*100:.1f}%")
    else:
        emit("  command_log not found.")

    section("5. Semantic Suppression (Closed-Loop)")
    if not cl_stats.empty:
        last = (cl_stats.sort_values("timestamp")
                .groupby(["device_id","device_type"], as_index=False).last())
        ts = int(last.get("total_samples", pd.Series([0])).sum())
        tx = int(last.get("total_sent",    pd.Series([0])).sum())
        su = int(last.get("total_suppressed", pd.Series([0])).sum())
        if ts > 0:
            emit(f"  Samples: {ts:,}  Sent: {tx:,} ({tx/ts*100:.1f}%)"
                 f"  Suppressed: {su:,} ({su/ts*100:.1f}%)")

    hr("═")
    emit()

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  ✓ summary_report.txt")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_FIGURES = list(range(1, 13))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate publication figures for ANSC evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (both baseline and closed-loop):
  python3 analyze_results_final.py \\
      --baseline-dir   outputs/evaluation/baseline_natural_20260402 \\
      --closedloop-dir outputs/evaluation/closedloop_natural_20260402 \\
      --output-dir     outputs/figures

  # Single closed-loop directory only:
  python3 analyze_results_final.py \\
      --closedloop-dir outputs/evaluation/closedloop_natural_20260402 \\
      --output-dir     outputs/figures

  # Specific figures only:
  python3 analyze_results_final.py \\
      --closedloop-dir ... --output-dir ... --figures 1 3 4 6

  # High-res PNG + PDF, custom DPI:
  python3 analyze_results_final.py \\
      --closedloop-dir ... --output-dir ... --format png pdf --dpi 300
""")
    p.add_argument("--baseline-dir",   default="",
                   help="Path to baseline evaluation run directory.")
    p.add_argument("--closedloop-dir", default="",
                   help="Path to closed-loop evaluation run directory.")
    p.add_argument("--output-dir",     default="outputs/figures",
                   help="Output directory for figures and report.")
    p.add_argument("--figures", nargs="+", type=int,
                   default=ALL_FIGURES,
                   help=f"Figure numbers to generate (default: all {ALL_FIGURES}).")
    p.add_argument("--format", nargs="+", default=["png"],
                   choices=["png", "pdf", "svg"],
                   help="Output file format(s) (default: png).")
    p.add_argument("--dpi", type=int, default=180,
                   help="Figure resolution (default: 180).")
    p.add_argument("--no-report", action="store_true",
                   help="Skip generating summary_report.txt.")
    p.add_argument("--phases", action="store_true", default=True,
                   help="Overlay scenario phase separators on timelines.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.baseline_dir and not args.closedloop_dir:
        print("[ERROR] Provide at least one of --baseline-dir or --closedloop-dir.")
        sys.exit(1)

    apply_style(dpi=args.dpi)
    os.makedirs(args.output_dir, exist_ok=True)
    figs_to_run = set(args.figures)
    fmt = args.format

    print(f"\n{'='*60}")
    print(f"  ANSC Figure Generator")
    print(f"  Output : {args.output_dir}")
    print(f"  Figures: {sorted(figs_to_run)}")
    print(f"  Formats: {fmt}   DPI: {args.dpi}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    b_tel    = pd.DataFrame()
    cl_tel   = pd.DataFrame()
    b_stats  = pd.DataFrame()
    cl_stats = pd.DataFrame()
    b_cmds   = pd.DataFrame()
    cl_cmds  = pd.DataFrame()

    if args.baseline_dir:
        print(f"[LOAD] Baseline: {args.baseline_dir}")
        b_tel   = load_telemetry(args.baseline_dir)
        b_stats = load_sender_stats(args.baseline_dir)
        b_cmds  = load_command_log(args.baseline_dir)
        if not b_tel.empty:
            print(f"  telemetry: {len(b_tel)} rows, "
                  f"{b_tel['elapsed'].max():.0f}s, "
                  f"states={b_tel['network_condition'].value_counts().to_dict()}")

    if args.closedloop_dir:
        print(f"[LOAD] Closed-loop: {args.closedloop_dir}")
        cl_tel   = load_telemetry(args.closedloop_dir)
        cl_stats = load_sender_stats(args.closedloop_dir)
        cl_cmds  = load_command_log(args.closedloop_dir)
        if not cl_tel.empty:
            print(f"  telemetry: {len(cl_tel)} rows, "
                  f"{cl_tel['elapsed'].max():.0f}s, "
                  f"states={cl_tel['network_condition'].value_counts().to_dict()}")
            if "predictor_source" in cl_tel.columns:
                src = cl_tel["predictor_source"].value_counts().to_dict()
                print(f"  predictor: {src}")

    print()

    # ── Primary run for single-run figures (prefer CL, fall back to baseline)
    primary_tel   = cl_tel   if not cl_tel.empty   else b_tel
    primary_stats = cl_stats if not cl_stats.empty else b_stats
    primary_cmds  = cl_cmds  if not cl_cmds.empty  else b_cmds
    primary_label = ("Closed-Loop" if not cl_tel.empty else
                     "Baseline"   if not b_tel.empty  else "Run")

    # ── Generate figures ───────────────────────────────────────────────────
    print("[FIGURES]")

    if 1 in figs_to_run:
        print(" Fig 1: Network metrics timeline")
        df_for = primary_tel
        if not df_for.empty:
            fig1_network_timeline(df_for, args.output_dir, fmt,
                                  run_label=primary_label,
                                  phases=args.phases)
        # If both available, also produce per-run versions
        if not b_tel.empty and not cl_tel.empty:
            fig1_network_timeline(b_tel, args.output_dir, fmt,
                                  run_label="Baseline",
                                  phases=args.phases)
            # Rename to avoid collision
            for f in fmt:
                src = os.path.join(args.output_dir, f"fig1_network_timeline.{f}")
                dst = os.path.join(args.output_dir, f"fig1_network_timeline_baseline.{f}")
                if os.path.exists(src):
                    os.replace(src, dst)

    if 2 in figs_to_run:
        print(" Fig 2: Throughput timeline")
        if not primary_tel.empty:
            fig2_throughput_timeline(primary_tel, args.output_dir, fmt,
                                     run_label=primary_label,
                                     phases=args.phases)

    if 3 in figs_to_run:
        print(" Fig 3: Violin distributions")
        if not primary_tel.empty:
            fig3_violin_distributions(primary_tel, args.output_dir, fmt,
                                      run_label=primary_label)

    if 4 in figs_to_run:
        print(" Fig 4: Confidence KDE")
        if not primary_tel.empty:
            fig4_confidence_kde(primary_tel, args.output_dir, fmt,
                                run_label=primary_label)

    if 5 in figs_to_run:
        print(" Fig 5: Confidence heatmap")
        if not primary_tel.empty:
            fig5_confidence_heatmap(primary_tel, args.output_dir, fmt,
                                    run_label=primary_label)

    if 6 in figs_to_run:
        print(" Fig 6: Adaptation latency")
        fig6_adaptation_latency(primary_cmds, args.output_dir, fmt,
                                run_label=primary_label)

    if 7 in figs_to_run:
        print(" Fig 7: Loss comparison (baseline vs CL)")
        fig7_loss_comparison(b_tel, cl_tel, args.output_dir, fmt)

    if 8 in figs_to_run:
        print(" Fig 8: Throughput savings bar")
        fig8_throughput_savings(b_tel, cl_tel, args.output_dir, fmt)

    if 9 in figs_to_run:
        print(" Fig 9: ML vs heuristic agreement")
        if not primary_tel.empty:
            fig9_ml_vs_heuristic(primary_tel, args.output_dir, fmt,
                                 run_label=primary_label)

    if 10 in figs_to_run:
        print(" Fig 10: Composite summary")
        if not primary_tel.empty:
            fig10_composite_summary(primary_tel, args.output_dir, fmt,
                                    run_label=primary_label)

    if 11 in figs_to_run:
        print(" Fig 11: Command distribution")
        if not primary_cmds.empty:
            fig11_command_distribution(primary_cmds, args.output_dir, fmt,
                                       run_label=primary_label)
        else:
            print("  [SKIP] fig11: no command log")

    if 12 in figs_to_run:
        print(" Fig 12: Semantic suppression")
        if not primary_stats.empty:
            fig12_semantic_suppression(primary_stats, args.output_dir, fmt,
                                       run_label=primary_label)
        else:
            print("  [SKIP] fig12: no sender stats")

    # ── Summary report ────────────────────────────────────────────────────
    if not args.no_report:
        print("\n[REPORT]")
        generate_summary_report(b_tel, cl_tel, b_stats, cl_stats,
                                 primary_cmds, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Done. Files in: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
