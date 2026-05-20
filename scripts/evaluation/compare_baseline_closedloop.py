#!/usr/bin/env python3
"""
compare_baseline_closedloop.py
==============================
Pure comparison script: Baseline vs Closed-Loop on the SAME stress run.

Figures generated:
  A  Grouped bar  — mean metrics (loss, delay, jitter, throughput)
  B  Overlapping timeline — packet loss %
  C  Overlapping timeline — throughput kbps
  D  Box plots    — per-metric spread (loss, delay, jitter)
  E  State distribution — stacked bar (Stable/Unstable/Critical %)
  F  CDF curves   — empirical CDF of packet loss
  G  Violin plots — per-metric spread both runs side by side

Usage:
  python scripts/evaluation/compare_baseline_closedloop.py \
      --baseline   outputs/evaluation/baseline_natural_20260510_113738 \
      --closedloop outputs/evaluation/closedloop_natural_20260510_131407 \
      --out        outputs/evaluation/figures_comparison
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────

BL_COLOR = "#7570b3"   # baseline purple
CL_COLOR = "#1b9e77"   # closed-loop green
STATE_COLORS = {"Stable": "#2ecc71", "Unstable": "#f39c12", "Critical": "#e74c3c"}
STATE_ORDER   = ["Stable", "Unstable", "Critical"]

def apply_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        180,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#fafafa",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.55,
        "grid.linestyle":    "--",
        "lines.linewidth":   1.6,
        "savefig.dpi":       180,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

# ── Data loading ───────────────────────────────────────────────────────────

def load_tel(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "network_telemetry.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"telemetry not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["elapsed"] = df["timestamp"] - df["timestamp"].iloc[0]
    for c in ["packet_loss_rate","avg_delay","jitter","throughput_bps","bandwidth_usage_bps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "network_condition" not in df.columns and "network_state" in df.columns:
        df.rename(columns={"network_state": "network_condition"}, inplace=True)
    if "network_condition" not in df.columns:
        df["network_condition"] = "Stable"
    W = 9
    for col in ["packet_loss_rate","avg_delay","jitter","throughput_bps"]:
        if col in df.columns:
            df[f"{col}_sm"] = df[col].rolling(W, center=True, min_periods=1).mean()
    df_act = df[df["throughput_bps"] > 0].copy()
    df._active = df_act
    return df

def active(df):
    return df

# ── Save helper ────────────────────────────────────────────────────────────

def save_fig(fig, out_dir, name, fmts=("png","pdf")):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in fmts:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"))
    plt.close(fig)
    print(f"  [OK] {name}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE A — Grouped bar chart of mean metrics
# ══════════════════════════════════════════════════════════════════════════

def fig_a_grouped_bar(bl, cl, out_dir, fmts):
    bl_a, cl_a = active(bl), active(cl)
    metrics = {
        "Packet Loss\n(%)":    (bl_a["packet_loss_rate"].mean()*100,
                                cl_a["packet_loss_rate"].mean()*100),
        "Avg. Delay\n(ms)":    (bl_a["avg_delay"].mean(),
                                cl_a["avg_delay"].mean()),
        "Jitter\n(ms)":        (bl_a["jitter"].mean(),
                                cl_a["jitter"].mean()),
        "Throughput\n(kbps)":  (bl_a["throughput_bps"].mean()/1000,
                                cl_a["throughput_bps"].mean()/1000),
    }

    labels = list(metrics.keys())
    bl_vals = [v[0] for v in metrics.values()]
    cl_vals = [v[1] for v in metrics.values()]

    x  = np.arange(len(labels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_bl = ax.bar(x - w/2, bl_vals, w, label="Baseline",    color=BL_COLOR, alpha=0.85, edgecolor="white")
    bars_cl = ax.bar(x + w/2, cl_vals, w, label="Closed-Loop", color=CL_COLOR, alpha=0.85, edgecolor="white")

    for bar in list(bars_bl) + list(bars_cl):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + h*0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    # Improvement annotations
    for i, (bv, cv) in enumerate(zip(bl_vals, cl_vals)):
        if bv > 0:
            pct = (bv - cv) / bv * 100
            sign = "↓" if pct > 0 else "↑"
            color = "#27ae60" if pct > 0 else "#e74c3c"
            ax.text(x[i], max(bv, cv) * 1.15,
                    f"{sign}{abs(pct):.1f}%", ha="center", fontsize=8,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Value")
    ax.set_title("Figure A — Mean Network Metrics: Baseline vs Closed-Loop\n"
                 "(arrows show % change; ↓ = improvement)", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, out_dir, "figA_grouped_bar", fmts)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE B — Overlapping timeline: packet loss
# ══════════════════════════════════════════════════════════════════════════

def fig_b_loss_timeline(bl, cl, out_dir, fmts):
    bl_a, cl_a = active(bl), active(cl)
    col = "packet_loss_rate_sm" if "packet_loss_rate_sm" in bl_a.columns else "packet_loss_rate"

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(bl_a["elapsed"], bl_a[col]*100,
            color=BL_COLOR, linewidth=1.4, alpha=0.85, label="Baseline")
    ax.fill_between(bl_a["elapsed"], 0, bl_a[col]*100,
                    color=BL_COLOR, alpha=0.12)

    ax.plot(cl_a["elapsed"], cl_a[col]*100,
            color=CL_COLOR, linewidth=1.4, alpha=0.85, label="Closed-Loop")
    ax.fill_between(cl_a["elapsed"], 0, cl_a[col]*100,
                    color=CL_COLOR, alpha=0.12)

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Packet Loss Rate (%)")
    ax.set_title("Figure B — Packet Loss Rate over Time: Baseline vs Closed-Loop")
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max(bl["elapsed"].max(), cl["elapsed"].max()))
    plt.tight_layout()
    save_fig(fig, out_dir, "figB_loss_timeline", fmts)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE C — Overlapping timeline: throughput
# ══════════════════════════════════════════════════════════════════════════

def fig_c_throughput_timeline(bl, cl, out_dir, fmts):
    bl_a, cl_a = active(bl), active(cl)
    col_b = "throughput_bps_sm" if "throughput_bps_sm" in bl_a.columns else "throughput_bps"
    col_c = "throughput_bps_sm" if "throughput_bps_sm" in cl_a.columns else "throughput_bps"

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(bl_a["elapsed"], bl_a[col_b]/1000,
            color=BL_COLOR, linewidth=1.4, alpha=0.85, label="Baseline")
    ax.fill_between(bl_a["elapsed"], 0, bl_a[col_b]/1000,
                    color=BL_COLOR, alpha=0.12)

    ax.plot(cl_a["elapsed"], cl_a[col_c]/1000,
            color=CL_COLOR, linewidth=1.4, alpha=0.85, label="Closed-Loop")
    ax.fill_between(cl_a["elapsed"], 0, cl_a[col_c]/1000,
                    color=CL_COLOR, alpha=0.12)

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Throughput (kbps)")
    ax.set_title("Figure C — Throughput over Time: Baseline vs Closed-Loop\n"
                 "(lower CL = semantic suppression saving bandwidth)")
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max(bl["elapsed"].max(), cl["elapsed"].max()))
    plt.tight_layout()
    save_fig(fig, out_dir, "figC_throughput_timeline", fmts)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE D — Box plots: per-metric spread
# ══════════════════════════════════════════════════════════════════════════

def fig_d_boxplots(bl, cl, out_dir, fmts):
    bl_a, cl_a = active(bl), active(cl)
    metrics = [
        ("packet_loss_rate", "Packet Loss (%)",  lambda v: v*100),
        ("avg_delay",        "Avg. Delay (ms)",  lambda v: v),
        ("jitter",           "Jitter (ms)",       lambda v: v),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))

    for ax, (col, label, tfm) in zip(axes, metrics):
        bl_vals = tfm(bl_a[col].dropna().values)
        cl_vals = tfm(cl_a[col].dropna().values)
        bl_vals = bl_vals[bl_vals >= 0]
        cl_vals = cl_vals[cl_vals >= 0]

        bp = ax.boxplot(
            [bl_vals, cl_vals],
            labels=["Baseline", "Closed-Loop"],
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="#2c3e50", linewidth=2),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.3),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], [BL_COLOR, CL_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # Overlay individual points
        np.random.seed(42)
        for i, (vals, color) in enumerate([(bl_vals, BL_COLOR), (cl_vals, CL_COLOR)], 1):
            n = min(200, len(vals))
            idx = np.random.choice(len(vals), n, replace=False)
            jit = np.random.normal(0, 0.07, n)
            ax.scatter(np.full(n, i) + jit, vals[idx],
                       color=color, s=3, alpha=0.25, zorder=2)

        ax.set_ylabel(label)
        ax.set_title(label.split("(")[0].strip())

    fig.suptitle("Figure D — Per-Metric Spread: Baseline vs Closed-Loop\n"
                 "(box = IQR, line = median, dots = samples)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, "figD_boxplots", fmts)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE E — State distribution stacked bar
# ══════════════════════════════════════════════════════════════════════════

def fig_e_state_bar(bl, cl, out_dir, fmts):
    def state_pcts(df):
        counts = df["network_condition"].value_counts()
        total  = len(df)
        return {s: counts.get(s, 0)/total*100 for s in STATE_ORDER}

    bl_pcts = state_pcts(bl)
    cl_pcts = state_pcts(cl)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)

    for ax, pcts, label, color in [
        (axes[0], bl_pcts, "Baseline",    BL_COLOR),
        (axes[1], cl_pcts, "Closed-Loop", CL_COLOR),
    ]:
        bars = ax.bar(STATE_ORDER,
                      [pcts[s] for s in STATE_ORDER],
                      color=[STATE_COLORS[s] for s in STATE_ORDER],
                      edgecolor="white", linewidth=0.8, alpha=0.80)
        for bar, s in zip(bars, STATE_ORDER):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                    f"{h:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
        ax.set_title(label, pad=8)
        ax.set_ylabel("% of time windows")
        ax.set_ylim(0, 70)
        ax.set_xlabel("Network State")

    fig.suptitle("Figure E — Network State Distribution: Baseline vs Closed-Loop\n"
                 "(closed-loop should show more Stable, less Critical)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, "figE_state_distribution", fmts)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE F — CDF of packet loss
# ══════════════════════════════════════════════════════════════════════════

def fig_f_cdf(bl, cl, out_dir, fmts):
    bl_a, cl_a = active(bl), active(cl)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for df_a, label, color in [(bl_a, "Baseline", BL_COLOR), (cl_a, "Closed-Loop", CL_COLOR)]:
        vals = np.sort(df_a["packet_loss_rate"].dropna().values * 100)
        cdf  = np.arange(1, len(vals)+1) / len(vals)
        ax.plot(vals, cdf, color=color, linewidth=2.2, label=label)
        # median marker
        med = np.median(vals)
        ax.axvline(med, color=color, linestyle="--", linewidth=1.0, alpha=0.65)
        ax.text(med + 0.1, 0.08 if label == "Baseline" else 0.02,
                f"med={med:.2f}%", color=color, fontsize=8)

    ax.set_xlabel("Packet Loss Rate (%)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Figure F — Empirical CDF of Packet Loss: Baseline vs Closed-Loop\n"
                 "(curve shifted left = improvement)")
    ax.legend(framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    save_fig(fig, out_dir, "figF_cdf_loss", fmts)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE G — Violin plots side-by-side
# ══════════════════════════════════════════════════════════════════════════

def fig_g_violins(bl, cl, out_dir, fmts):
    bl_a, cl_a = active(bl), active(cl)
    metrics = [
        ("packet_loss_rate", "Packet Loss (%)",  lambda v: v*100),
        ("avg_delay",        "Avg. Delay (ms)",  lambda v: v),
        ("jitter",           "Jitter (ms)",       lambda v: v),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    np.random.seed(42)

    for ax, (col, label, tfm) in zip(axes, metrics):
        bl_vals = tfm(bl_a[col].dropna().values); bl_vals = bl_vals[bl_vals >= 0]
        cl_vals = tfm(cl_a[col].dropna().values); cl_vals = cl_vals[cl_vals >= 0]

        data = [bl_vals, cl_vals]
        positions = [1, 2]
        colors = [BL_COLOR, CL_COLOR]
        labels_v = ["Baseline", "Closed-Loop"]

        parts = ax.violinplot(data, positions=positions, widths=0.55,
                              showmeans=True, showmedians=False, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
            pc.set_edgecolor(color)
        parts["cmeans"].set_color("#2c3e50")
        parts["cmeans"].set_linewidth(2.2)

        for i, (vals, color) in enumerate(zip(data, colors), 1):
            q25, q75 = np.percentile(vals, [25, 75])
            ax.vlines(i, q25, q75, color=color, linewidth=5, alpha=0.80, zorder=3)
            n_show = min(150, len(vals))
            idx = np.random.choice(len(vals), n_show, replace=False)
            jit = np.random.normal(0, 0.06, n_show)
            ax.scatter(np.full(n_show, i) + jit, vals[idx],
                       color=color, s=3.5, alpha=0.25, zorder=2)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_v, fontsize=9.5)
        ax.set_ylabel(label)
        ax.set_title(label.split("(")[0].strip())

    fig.suptitle("Figure G — Distribution Spread: Baseline vs Closed-Loop\n"
                 "(filled bar = IQR, diamond = mean, dots = samples)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, "figG_violin_spread", fmts)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE H — Dual-timeline: loss + delay overlapping (4-panel grid)
# ══════════════════════════════════════════════════════════════════════════

def fig_h_four_panel(bl, cl, out_dir, fmts):
    bl_a, cl_a = active(bl), active(cl)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.32)

    panels = [
        ("packet_loss_rate_sm", "packet_loss_rate", "Packet Loss (%)", lambda v: v*100),
        ("avg_delay_sm",        "avg_delay",         "Avg. Delay (ms)", lambda v: v),
        ("jitter_sm",           "jitter",             "Jitter (ms)",     lambda v: v),
        ("throughput_bps_sm",   "throughput_bps",     "Throughput (kbps)", lambda v: v/1000),
    ]

    for ax, (sm_col, raw_col, ylabel, tfm) in zip(axes.flat, panels):
        for df_a, label, color in [(bl_a, "Baseline", BL_COLOR), (cl_a, "Closed-Loop", CL_COLOR)]:
            col = sm_col if sm_col in df_a.columns else raw_col
            if col not in df_a.columns:
                continue
            y = tfm(df_a[col].values)
            ax.plot(df_a["elapsed"], y, color=color, linewidth=1.3,
                    alpha=0.85, label=label)
            ax.fill_between(df_a["elapsed"], 0, y, color=color, alpha=0.07)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Elapsed (s)")
        ax.set_title(ylabel.split("(")[0].strip())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, framealpha=0.85)

    fig.suptitle("Figure H — All Metrics Side-by-Side Timeline: Baseline vs Closed-Loop",
                 fontsize=12, fontweight="bold")
    save_fig(fig, out_dir, "figH_four_panel_timeline", fmts)

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",   required=True)
    ap.add_argument("--closedloop", required=True)
    ap.add_argument("--out",        default="outputs/evaluation/figures_comparison")
    ap.add_argument("--no-pdf",     action="store_true")
    args = ap.parse_args()

    fmts = ["png"] if args.no_pdf else ["png", "pdf"]
    out  = args.out

    apply_style()

    print(f"\nLoading baseline   : {args.baseline}")
    bl = load_tel(args.baseline)
    print(f"Loading closed-loop: {args.closedloop}")
    cl = load_tel(args.closedloop)

    print(f"\nGenerating figures -> {out}\n")
    fig_a_grouped_bar(bl, cl, out, fmts)
    fig_b_loss_timeline(bl, cl, out, fmts)
    fig_c_throughput_timeline(bl, cl, out, fmts)
    fig_d_boxplots(bl, cl, out, fmts)
    fig_e_state_bar(bl, cl, out, fmts)
    fig_f_cdf(bl, cl, out, fmts)
    fig_g_violins(bl, cl, out, fmts)
    fig_h_four_panel(bl, cl, out, fmts)

    # Print quick summary
    bl_a, cl_a = active(bl), active(cl)
    print("\n-- Summary (ACTIVE Windows only) --------------------")
    for metric, col, tfm, unit in [
        ("Loss",       "packet_loss_rate", lambda v: v*100, "%"),
        ("Delay",      "avg_delay",         lambda v: v,    "ms"),
        ("Jitter",     "jitter",             lambda v: v,    "ms"),
        ("Throughput", "throughput_bps",     lambda v: v/1000, "kbps"),
    ]:
        bv = tfm(bl_a[col].mean())
        cv = tfm(cl_a[col].mean())
        pct = (bv - cv)/bv*100 if bv > 0 else 0
        arrow = "v" if pct > 0 else "^"
        print(f"  {metric:<12}: BL={bv:.2f}{unit}  CL={cv:.2f}{unit}  {arrow}{abs(pct):.1f}%")
    print("-----------------------------------------------------")

    print("\n-- Summary (ALL Windows / Fair Overall Comparison) --")
    for metric, col, tfm, unit in [
        ("Loss",       "packet_loss_rate", lambda v: v*100, "%"),
        ("Delay",      "avg_delay",         lambda v: v,    "ms"),
        ("Jitter",     "jitter",             lambda v: v,    "ms"),
        ("Throughput", "throughput_bps",     lambda v: v/1000, "kbps"),
    ]:
        bv = tfm(bl[col].mean())
        cv = tfm(cl[col].mean())
        pct = (bv - cv)/bv*100 if bv > 0 else 0
        arrow = "v" if pct > 0 else "^"
        print(f"  {metric:<12}: BL={bv:.2f}{unit}  CL={cv:.2f}{unit}  {arrow}{abs(pct):.1f}%")
    print("-----------------------------------------------------\n")


if __name__ == "__main__":
    main()
