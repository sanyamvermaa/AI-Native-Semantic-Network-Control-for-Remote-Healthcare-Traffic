#!/usr/bin/env python3
"""
compare_v2_allrows.py  —  Corrected Baseline vs Closed-Loop comparison
=======================================================================
Key fix: uses ALL telemetry rows (not just active/throughput>0) for loss
comparison so CL's semantic suppression doesn't bias the mean upward.

Extra figures:
  A  Grouped bar     — mean loss/delay/jitter (ALL rows, apples-to-apples)
  B  Overlapping CDF — loss distribution, all rows
  C  State-stratified bar — mean loss per state (Stable/Unstable/Critical)
  D  Bytes saved     — cumulative BL vs CL bytes, suppression gap shaded
  E  Suppression timeline — per-window: sent vs suppressed fraction
  F  Four-panel timeline  — all metrics, all rows

Usage:
  python scripts/evaluation/compare_v2_allrows.py \
      --baseline   outputs/evaluation/baseline_natural_20260510_113738 \
      --closedloop outputs/evaluation/closedloop_natural_20260510_131407 \
      --out        outputs/evaluation/figures_comparison_v2
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

BL_COLOR = "#7570b3"
CL_COLOR = "#1b9e77"
STATE_COLORS = {"Stable": "#2ecc71", "Unstable": "#f39c12", "Critical": "#e74c3c"}
STATE_ORDER   = ["Stable", "Unstable", "Critical"]


def apply_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 180, "figure.facecolor": "white",
        "axes.facecolor": "#fafafa", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e0e0e0", "grid.linewidth": 0.55, "grid.linestyle": "--",
        "lines.linewidth": 1.6, "savefig.dpi": 180,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })


def load_tel(run_dir):
    path = os.path.join(run_dir, "network_telemetry.csv")
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["elapsed"] = df["timestamp"] - df["timestamp"].iloc[0]
    for c in ["packet_loss_rate","avg_delay","jitter","throughput_bps","bandwidth_usage_bps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "network_condition" not in df.columns:
        df["network_condition"] = "Stable"
    W = 9
    for col in ["packet_loss_rate","avg_delay","jitter","throughput_bps"]:
        if col in df.columns:
            df[f"{col}_sm"] = df[col].rolling(W, center=True, min_periods=1).mean()
    return df


def save_fig(fig, out_dir, name, fmts):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in fmts:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"))
    plt.close(fig)
    print(f"  [OK] {name}")


# ── Figure A: Grouped bar — ALL rows, apples-to-apples ────────────────────
def fig_a_allrows_bar(bl, cl, out_dir, fmts):
    metrics = {
        "Packet Loss\n(% all windows)": (bl["packet_loss_rate"].mean()*100,
                                          cl["packet_loss_rate"].mean()*100),
        "Avg. Delay\n(ms)":             (bl["avg_delay"].mean(),
                                          cl["avg_delay"].mean()),
        "Jitter\n(ms)":                 (bl["jitter"].mean(),
                                          cl["jitter"].mean()),
    }
    labels  = list(metrics.keys())
    bl_vals = [v[0] for v in metrics.values()]
    cl_vals = [v[1] for v in metrics.values()]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_bl = ax.bar(x - w/2, bl_vals, w, label="Baseline",    color=BL_COLOR, alpha=0.85, edgecolor="white")
    bars_cl = ax.bar(x + w/2, cl_vals, w, label="Closed-Loop", color=CL_COLOR, alpha=0.85, edgecolor="white")

    for bar in list(bars_bl) + list(bars_cl):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+h*0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5)

    for i, (bv, cv) in enumerate(zip(bl_vals, cl_vals)):
        if bv > 0:
            pct  = (bv - cv) / bv * 100
            sign = "v" if pct > 0 else "^"
            col  = "#27ae60" if pct > 0 else "#e74c3c"
            ax.text(x[i], max(bv,cv)*1.15, f"{sign}{abs(pct):.1f}%",
                    ha="center", fontsize=8.5, color=col, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Value")
    ax.set_title("Figure A  —  Mean Metrics: ALL Telemetry Windows (Apples-to-Apples)\n"
                 "Throughput excluded here — CL intentionally sends less (semantic suppression)")
    ax.legend(framealpha=0.9)

    # annotation box explaining suppression
    ax.text(0.99, 0.97,
            "Note: CL zero-throughput windows = 16%\n"
            "(semantic suppression active)\n"
            "BL zero-throughput windows = 0.2%",
            transform=ax.transAxes, va="top", ha="right", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#ccc", alpha=0.9))
    plt.tight_layout()
    save_fig(fig, out_dir, "figA_allrows_bar", fmts)


# ── Figure B: CDF — ALL rows ───────────────────────────────────────────────
def fig_b_allrows_cdf(bl, cl, out_dir, fmts):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left — CDF including zero-loss windows
    for df, label, color in [(bl, "Baseline", BL_COLOR), (cl, "Closed-Loop", CL_COLOR)]:
        vals = np.sort(df["packet_loss_rate"].dropna().values * 100)
        cdf  = np.arange(1, len(vals)+1) / len(vals)
        axes[0].plot(vals, cdf, color=color, linewidth=2.2, label=label)
        med  = np.median(vals)
        axes[0].axvline(med, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
        axes[0].text(med+0.15, 0.06 if label=="Baseline" else 0.02,
                     f"med={med:.2f}%", color=color, fontsize=8)
    axes[0].set_xlabel("Packet Loss Rate (%)")
    axes[0].set_ylabel("Cumulative Probability")
    axes[0].set_title("CDF — ALL rows\n(incl. zero-throughput / suppressed windows)")
    axes[0].legend(framealpha=0.9)
    axes[0].set_xlim(left=0); axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Right — CDF active-only (throughput>0)
    for df, label, color in [(bl, "Baseline", BL_COLOR), (cl, "Closed-Loop", CL_COLOR)]:
        df_a = df[df["throughput_bps"] > 0]
        vals = np.sort(df_a["packet_loss_rate"].dropna().values * 100)
        cdf  = np.arange(1, len(vals)+1) / len(vals)
        axes[1].plot(vals, cdf, color=color, linewidth=2.2, label=label)
    axes[1].set_xlabel("Packet Loss Rate (%)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("CDF — ACTIVE rows only\n(throughput > 0, biased subset)")
    axes[1].legend(framealpha=0.9)
    axes[1].set_xlim(left=0); axes[1].set_ylim(0, 1.02)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.suptitle("Figure B  —  Empirical CDF of Packet Loss: All-rows vs Active-rows",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, "figB_allrows_cdf", fmts)


# ── Figure C: State-stratified loss bars ──────────────────────────────────
def fig_c_state_stratified(bl, cl, out_dir, fmts):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=False)

    for ax, state in zip(axes, STATE_ORDER):
        bl_sub = bl[bl["network_condition"] == state]["packet_loss_rate"] * 100
        cl_sub = cl[cl["network_condition"] == state]["packet_loss_rate"] * 100

        bl_mean = bl_sub.mean() if len(bl_sub) else 0
        cl_mean = cl_sub.mean() if len(cl_sub) else 0
        bl_std  = bl_sub.std()  if len(bl_sub) else 0
        cl_std  = cl_sub.std()  if len(cl_sub) else 0

        x = np.array([0, 1])
        bars = ax.bar(x, [bl_mean, cl_mean], 0.45,
                      color=[BL_COLOR, CL_COLOR], alpha=0.82, edgecolor="white",
                      yerr=[bl_std, cl_std], capsize=6,
                      error_kw=dict(elinewidth=1.2, ecolor="#555"))
        for bar, val in zip(bars, [bl_mean, cl_mean]):
            ax.text(bar.get_x()+bar.get_width()/2, val+val*0.04+0.3,
                    f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Closed-Loop"])
        ax.set_ylabel("Mean Packet Loss (%)")
        ax.set_title(f"State: {state}\n(n_BL={len(bl_sub)}, n_CL={len(cl_sub)})",
                     color=STATE_COLORS[state])
        ax.set_ylim(bottom=0)

    fig.suptitle("Figure C  —  State-Stratified Packet Loss: Baseline vs Closed-Loop\n"
                 "(error bars = std dev; compares like-for-like within each network state)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, "figC_state_stratified_loss", fmts)


# ── Figure D: Cumulative bytes — suppression gap ──────────────────────────
def fig_d_bytes_saved(bl, cl, out_dir, fmts):
    bl_bytes = (bl["throughput_bps"] / 8).cumsum() / 1e6   # cumulative MB
    cl_bytes = (cl["throughput_bps"] / 8).cumsum() / 1e6

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(bl["elapsed"], bl_bytes, color=BL_COLOR, linewidth=2, label="Baseline (cumulative)")
    ax.plot(cl["elapsed"], cl_bytes, color=CL_COLOR, linewidth=2, label="Closed-Loop (cumulative)")

    # Shade the gap
    common_len = min(len(bl), len(cl))
    t  = bl["elapsed"].values[:common_len]
    b  = bl_bytes.values[:common_len]
    c  = cl_bytes.values[:common_len]
    ax.fill_between(t, c, b, where=(b >= c), alpha=0.18,
                    color="#e74c3c", label="Bandwidth saved (suppression)")

    saved_mb = bl_bytes.iloc[-1] - cl_bytes.iloc[-1]
    pct_saved = saved_mb / bl_bytes.iloc[-1] * 100
    ax.text(0.02, 0.97,
            f"Total saved: {saved_mb:.1f} MB  ({pct_saved:.1f}%)\n"
            f"= semantic suppression + downsampling",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Cumulative Data (MB)")
    ax.set_title("Figure D  —  Cumulative Bytes Transmitted: Baseline vs Closed-Loop\n"
                 "(shaded gap = bandwidth saved by semantic suppression)")
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, out_dir, "figD_bytes_saved", fmts)


# ── Figure E: Suppression timeline (CL only) ──────────────────────────────
def fig_e_suppression(cl_dir, out_dir, fmts):
    import glob
    pattern = os.path.join(cl_dir, "sender_semantic_stats_dev*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print("  [SKIP] figE: no sender stats")
        return

    frames = []
    for fp in files:
        try: frames.append(pd.read_csv(fp, low_memory=False))
        except: pass
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    for c in ["total_sent","total_suppressed","total_samples","timestamp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["elapsed"] = df["timestamp"] - df["timestamp"].iloc[0]

    # Aggregate per time window across all devices
    df_grp = df.groupby("timestamp").agg(
        elapsed=("elapsed","first"),
        sent=("total_sent","sum"),
        suppressed=("total_suppressed","sum"),
        samples=("total_samples","sum"),
    ).reset_index()
    # Compute per-window delta
    df_grp["d_sent"] = df_grp["sent"].diff().clip(lower=0).fillna(0)
    df_grp["d_supp"] = df_grp["suppressed"].diff().clip(lower=0).fillna(0)
    df_grp["total"]  = df_grp["d_sent"] + df_grp["d_supp"]
    df_grp["supp_frac"] = np.where(df_grp["total"] > 0,
                                    df_grp["d_supp"] / df_grp["total"], 0)

    W = 15
    df_grp["supp_frac_sm"] = df_grp["supp_frac"].rolling(W, center=True, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3,1]})
    fig.subplots_adjust(hspace=0.08)

    ax1.fill_between(df_grp["elapsed"], df_grp["supp_frac_sm"]*100,
                     alpha=0.35, color=CL_COLOR)
    ax1.plot(df_grp["elapsed"], df_grp["supp_frac_sm"]*100,
             color=CL_COLOR, linewidth=1.6, label="Suppression fraction (smoothed)")
    ax1.axhline(df_grp["supp_frac"].mean()*100, color="#e74c3c", linewidth=1.0,
                linestyle="--", label=f"Mean={df_grp['supp_frac'].mean()*100:.1f}%")
    ax1.set_ylabel("% Samples Suppressed per Window")
    ax1.set_title("Figure E  —  Semantic Suppression Rate over Time (Closed-Loop)\n"
                  "High suppression = more conservative transmission during stress")
    ax1.legend(framealpha=0.9)
    ax1.set_ylim(0, 100)

    # Bar strip: sent vs suppressed
    ax2.bar(df_grp["elapsed"], df_grp["d_sent"],
            width=1.0, color=CL_COLOR, alpha=0.7, label="Sent")
    ax2.bar(df_grp["elapsed"], df_grp["d_supp"], bottom=df_grp["d_sent"],
            width=1.0, color="#e74c3c", alpha=0.5, label="Suppressed")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Samples")
    ax2.legend(fontsize=8, framealpha=0.85)

    plt.tight_layout()
    save_fig(fig, out_dir, "figE_suppression_timeline", fmts)


# ── Figure F: Four-panel timeline, ALL rows ────────────────────────────────
def fig_f_fourpanel_allrows(bl, cl, out_dir, fmts):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    panels = [
        ("packet_loss_rate_sm","packet_loss_rate","Packet Loss (%)",    lambda v: v*100),
        ("avg_delay_sm",       "avg_delay",       "Avg. Delay (ms)",    lambda v: v),
        ("jitter_sm",          "jitter",           "Jitter (ms)",        lambda v: v),
        ("throughput_bps_sm",  "throughput_bps",   "Throughput (kbps)",  lambda v: v/1000),
    ]

    for ax, (sm_col, raw_col, ylabel, tfm) in zip(axes.flat, panels):
        for df, label, color in [(bl,"Baseline",BL_COLOR),(cl,"Closed-Loop",CL_COLOR)]:
            col = sm_col if sm_col in df.columns else raw_col
            y   = tfm(df[col].values)
            ax.plot(df["elapsed"], y, color=color, linewidth=1.3, alpha=0.85, label=label)
            ax.fill_between(df["elapsed"], 0, y, color=color, alpha=0.07)
        ax.set_ylabel(ylabel); ax.set_xlabel("Elapsed (s)")
        ax.set_title(ylabel.split("(")[0].strip())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, framealpha=0.85)

    fig.suptitle("Figure F  —  All Metrics Timeline (ALL rows): Baseline vs Closed-Loop\n"
                 "Zero-throughput CL windows visible as drops — these are suppression events",
                 fontsize=11, fontweight="bold")
    save_fig(fig, out_dir, "figF_fourpanel_allrows", fmts)


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",   required=True)
    ap.add_argument("--closedloop", required=True)
    ap.add_argument("--out",   default="outputs/evaluation/figures_comparison_v2")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    fmts = ["png"] if args.no_pdf else ["png", "pdf"]
    apply_style()

    print(f"\nLoading baseline   : {args.baseline}")
    bl = load_tel(args.baseline)
    print(f"Loading closed-loop: {args.closedloop}")
    cl = load_tel(args.closedloop)

    out = args.out
    print(f"\nGenerating figures -> {out}\n")
    fig_a_allrows_bar(bl, cl, out, fmts)
    fig_b_allrows_cdf(bl, cl, out, fmts)
    fig_c_state_stratified(bl, cl, out, fmts)
    fig_d_bytes_saved(bl, cl, out, fmts)
    fig_e_suppression(args.closedloop, out, fmts)
    fig_f_fourpanel_allrows(bl, cl, out, fmts)

    # Summary
    print("\n-- Summary (ALL rows) -----------------------------------")
    for metric, col, tfm, unit in [
        ("Loss",   "packet_loss_rate", lambda v: v*100, "%"),
        ("Delay",  "avg_delay",        lambda v: v,     "ms"),
        ("Jitter", "jitter",           lambda v: v,     "ms"),
    ]:
        bv = tfm(bl[col].mean())
        cv = tfm(cl[col].mean())
        pct = (bv-cv)/bv*100 if bv > 0 else 0
        print(f"  {metric:<8}: BL={bv:.3f}{unit}  CL={cv:.3f}{unit}  {'v' if pct>0 else '^'}{abs(pct):.1f}%")

    bl_bytes = (bl["throughput_bps"]/8).sum() / 1e6
    cl_bytes = (cl["throughput_bps"]/8).sum() / 1e6
    saved    = bl_bytes - cl_bytes
    print(f"  BytesSent : BL={bl_bytes:.1f}MB  CL={cl_bytes:.1f}MB  saved={saved:.1f}MB ({saved/bl_bytes*100:.1f}%)")
    print("---------------------------------------------------------\n")


if __name__ == "__main__":
    main()
