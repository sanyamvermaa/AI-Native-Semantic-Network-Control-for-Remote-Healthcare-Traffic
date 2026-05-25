#!/usr/bin/env python3
"""
semantic_fidelity.py — Measure whether semantic transmission preserved clinical meaning.

Reads CSV logs produced by the closed-loop system and generates 4 figures
plus a summary report. Does NOT require PyTorch at runtime.

Usage (project root, WSL):
    python3 scripts/evaluation/semantic_fidelity.py \\
        [--logs-dir  data/logs] \\
        [--output-dir plots/semantic_fidelity]

Outputs:
    fig1_fidelity_heatmap.{pdf,png}
    fig2_bandwidth_by_mode.{pdf,png}
    fig3_confidence_vs_f1.{pdf,png}
    fig4_latency_cdf.{pdf,png}
    summary_report.txt
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("[WARN] scipy not installed — regression line in fig3 will be skipped.")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

plt.rcParams.update({
    "figure.dpi":    150,
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
})

RAW_PACKET_SIZE = 64   # bytes — baseline for bandwidth reduction calculation
SLA_MS          = 1000.0

NETWORK_STATES  = ["Stable", "Unstable", "Critical"]
COMMANDS        = ["FULL_ECG", "SEMANTIC_ALERT", "SEMANTIC_CRITICAL", "SEMANTIC_SUMMARY"]

STATE_COLORS = {
    "Stable":   "#2ca02c",
    "Unstable": "#ff7f0e",
    "Critical": "#d62728",
    "UNKNOWN":  "#999999",
}
COMMAND_MARKERS = {
    "FULL_ECG":           "o",
    "FULL_ECG_PRIORITY":  "s",
    "DOWNSAMPLED_ECG":    "D",
    "SEMANTIC_ALERT":     "^",
    "SEMANTIC_CRITICAL":  "v",
    "SEMANTIC_SUMMARY":   "P",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Severity ordering used for worst-case window aggregation
SEVERITY = {"NORMAL": 0, "ALERT": 1, "CRITICAL": 2}


def _worst_label(series: pd.Series) -> str:
    """Return the most severe clinical label in a Series (CRITICAL > ALERT > NORMAL)."""
    worst = max(series, key=lambda x: SEVERITY.get(str(x).strip().upper(), 0))
    return str(worst).strip().upper()


def _save(fig: plt.Figure, path_no_ext: str) -> None:
    for ext in ("pdf", "png", "eps"):
        fig.savefig(f"{path_no_ext}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def semantic_critical_sla_pct(commands: pd.DataFrame) -> Optional[float]:
    """Return percent of SEMANTIC_CRITICAL commands within SLA, or None if unavailable."""
    if commands.empty or "latency_ms" not in commands.columns or "command" not in commands.columns:
        return None
    sc = commands[commands["command"] == "SEMANTIC_CRITICAL"]["latency_ms"].dropna()
    if len(sc) == 0:
        return None
    return float(np.mean(sc <= SLA_MS)) * 100.0


# ---------------------------------------------------------------------------
# STEP 1 — Load and align data
# ---------------------------------------------------------------------------

def load_sender_logs(logs_dir: str) -> pd.DataFrame:
    """Load all sender_log_dev*.csv, merge into one sorted DataFrame."""
    pattern = os.path.join(logs_dir, "sender_log_dev*.csv")
    frames: List[pd.DataFrame] = []
    for fpath in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(fpath, dtype=str)
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Could not read {fpath}: {exc}")
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"]  = pd.to_numeric(df["timestamp"],  errors="coerce")
    df["seq"]        = pd.to_numeric(df["seq"],         errors="coerce")
    df["device_id"]  = pd.to_numeric(df["device_id"],  errors="coerce")
    df["value"]      = pd.to_numeric(df["value"],       errors="coerce")
    df["label"]      = df["label"].str.strip().str.upper().fillna("NORMAL")
    # "encoded" column added by updated health_sender.py; treat missing as False
    if "encoded" in df.columns:
        df["encoded"] = df["encoded"].map(
            lambda x: str(x).strip().lower() in ("true", "1", "yes")
        )
    else:
        df["encoded"] = False

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_telemetry(logs_dir: str) -> pd.DataFrame:
    path = os.path.join(logs_dir, "network_telemetry.csv")
    if not os.path.isfile(path):
        print(f"[WARN] network_telemetry.csv not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Support both old (no decode_confidence) and new schema
    if "decode_confidence" not in df.columns:
        df["decode_confidence"] = float("nan")
    else:
        df["decode_confidence"] = pd.to_numeric(df["decode_confidence"], errors="coerce")

    if "semantic_packets_in_window" not in df.columns:
        df["semantic_packets_in_window"] = 0
    else:
        df["semantic_packets_in_window"] = pd.to_numeric(
            df["semantic_packets_in_window"], errors="coerce"
        ).fillna(0)

    # Normalise network_condition column name (may be network_condition or network_state)
    if "network_condition" not in df.columns and "network_state" in df.columns:
        df = df.rename(columns={"network_state": "network_condition"})
    if "network_condition" not in df.columns:
        df["network_condition"] = "Stable"

    return df


def load_command_log(logs_dir: str) -> pd.DataFrame:
    path = os.path.join(logs_dir, "command_log.csv")
    if not os.path.isfile(path):
        print(f"[WARN] command_log.csv not found at {path}")
        return pd.DataFrame(
            columns=["timestamp", "network_state", "health_state",
                     "command", "latency_ms", "consecutive_windows"]
        )
    df = pd.read_csv(path)
    df["timestamp"]  = pd.to_numeric(df["timestamp"],  errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if "command" not in df.columns:
        df["command"] = "FULL_ECG"
    return df


def align_packets(
    sender: pd.DataFrame,
    telemetry: pd.DataFrame,
    commands: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge sender packets with telemetry (nearest 0.3s) and commands
    (most-recent-before using merge_asof).
    """
    if sender.empty:
        return pd.DataFrame()

    merged = sender.copy()

    # --- align to telemetry (network_condition, decode_confidence)
    if not telemetry.empty:
        tele_cols = ["timestamp", "network_condition", "decode_confidence",
                     "semantic_packets_in_window"]
        tele_small = telemetry[tele_cols].sort_values("timestamp")
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            tele_small,
            on="timestamp",
            tolerance=0.3,
            direction="nearest",
            suffixes=("", "_tele"),
        )
    else:
        merged["network_condition"]         = "Stable"
        merged["decode_confidence"]         = float("nan")
        merged["semantic_packets_in_window"] = 0

    merged["network_condition"] = merged["network_condition"].fillna("Stable")

    # --- align to command_log (most recent command before each packet)
    if not commands.empty:
        cmd_small = commands[["timestamp", "command"]].sort_values("timestamp")
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            cmd_small,
            on="timestamp",
            direction="backward",
        )
    else:
        merged["command"] = "FULL_ECG"

    merged["command"] = merged["command"].fillna("FULL_ECG")

    # Packet size heuristic: encoded packets are JSON (~200 bytes), raw are 64 bytes
    merged["packet_bytes"] = merged["encoded"].apply(
        lambda enc: 200 if enc else RAW_PACKET_SIZE
    )

    return merged.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 2 — Assign decoded labels
# ---------------------------------------------------------------------------

def assign_decoded_labels(
    merged: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    For encoded packets: decoded label = health_state from the nearest telemetry
    window. For non-encoded packets: decoded label = ground-truth label.
    """
    df = merged.copy()

    if not telemetry.empty and "health_state" in telemetry.columns:
        # Build a "health_state" column from telemetry for encoded packets
        tele_hs = telemetry[["timestamp", "health_state"]].dropna()
        if not tele_hs.empty:
            tele_hs = tele_hs.sort_values("timestamp")
            hs_merged = pd.merge_asof(
                df[["timestamp"]].sort_values("timestamp").copy(),
                tele_hs,
                on="timestamp",
                tolerance=0.3,
                direction="nearest",
            )
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["_tele_health_state"] = hs_merged["health_state"].values
        else:
            df["_tele_health_state"] = float("nan")
    else:
        df["_tele_health_state"] = float("nan")

    def _decoded(row) -> str:
        if row["encoded"] and pd.notna(row["_tele_health_state"]):
            return str(row["_tele_health_state"]).strip().upper()
        return str(row["label"]).strip().upper()

    df["decoded_label"] = df.apply(_decoded, axis=1)
    df["decoded_label"] = df["decoded_label"].map(
        lambda x: x if x in ("NORMAL", "ALERT", "CRITICAL") else "NORMAL"
    )
    return df


# ---------------------------------------------------------------------------
# STEP 3 — Per-cell metrics
# ---------------------------------------------------------------------------

def compute_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (network_condition × command) cell:
      clinical_state_f1, bandwidth_bytes, bandwidth_reduction, decode_confidence_mean

    F1 is computed at the 250ms-window level (floor timestamp to nearest 0.25s),
    taking the worst-case ground-truth label and worst-case decoded label per window.
    """
    if df.empty:
        return pd.DataFrame()

    # Pre-compute 250ms window key once for all rows
    df = df.copy()
    df["_win_key"] = (df["timestamp"] // 0.25).astype("int64")

    records = []
    for net_cond in df["network_condition"].unique():
        for cmd in df["command"].unique():
            cell = df[(df["network_condition"] == net_cond) & (df["command"] == cmd)]
            if cell.empty:
                continue

            # --- Window-level aggregation — encoded packets only ---
            enc_cell = cell[cell["encoded"] == True]
            win_agg = (
                enc_cell.groupby("_win_key")
                .agg(
                    true_label=("label", _worst_label),
                    dec_label=("decoded_label", _worst_label),
                )
                if not enc_cell.empty
                else pd.DataFrame(columns=["true_label", "dec_label"])
            )

            true_labels = win_agg["true_label"].tolist()
            dec_labels  = win_agg["dec_label"].tolist()

            if len(true_labels) < 1:
                f1 = float("nan")
            else:
                try:
                    f1 = f1_score(
                        true_labels, dec_labels,
                        labels=["NORMAL", "ALERT", "CRITICAL"],
                        average="weighted",
                        zero_division=0,
                    )
                except Exception:
                    f1 = float("nan")

            # Bandwidth and confidence remain packet-level metrics
            bw_bytes       = cell["packet_bytes"].sum()
            raw_baseline   = len(cell) * RAW_PACKET_SIZE
            bw_reduction   = 1.0 - bw_bytes / raw_baseline if raw_baseline > 0 else 0.0

            enc_subset     = cell[cell["encoded"] == True]
            conf_mean      = (
                enc_subset["decode_confidence"].mean()
                if not enc_subset.empty and enc_subset["decode_confidence"].notna().any()
                else float("nan")
            )

            records.append({
                "network_condition":    net_cond,
                "command":              cmd,
                "clinical_state_f1":   f1,
                "bandwidth_bytes":     bw_bytes,
                "bandwidth_reduction": bw_reduction,
                "decode_confidence_mean": conf_mean,
                "n_packets":           len(cell),
                "n_encoded":           int(cell["encoded"].sum()),
                "n_windows":           len(win_agg),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# STEP 4 — Figures
# ---------------------------------------------------------------------------

def fig1_heatmap(cells: pd.DataFrame, output_dir: str) -> None:
    """Figure 1: Clinical state F1 heatmap (network state × command)."""
    # Build pivot table; rows = network state, cols = command
    all_states   = NETWORK_STATES
    all_commands = COMMANDS

    f1_matrix  = np.full((len(all_states), len(all_commands)), float("nan"))
    bw_matrix  = np.full((len(all_states), len(all_commands)), float("nan"))

    for i, state in enumerate(all_states):
        for j, cmd in enumerate(all_commands):
            row = cells[(cells["network_condition"] == state) &
                        (cells["command"] == cmd)]
            if not row.empty:
                f1_matrix[i, j]  = float(row["clinical_state_f1"].iloc[0])
                bw_matrix[i, j]  = float(row["bandwidth_reduction"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom green-red colormap: NaN → grey
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#cccccc")

    masked = np.ma.masked_invalid(f1_matrix)
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Clinical state F1 (weighted)")

    ax.set_xticks(range(len(all_commands)))
    ax.set_xticklabels(all_commands, rotation=25, ha="right")
    ax.set_yticks(range(len(all_states)))
    ax.set_yticklabels(all_states)
    ax.set_title("Semantic fidelity heatmap (clinical state F1 by network × command)")
    ax.set_xlabel("Transmission command")
    ax.set_ylabel("Network condition")

    # Annotate cells
    for i in range(len(all_states)):
        for j in range(len(all_commands)):
            f1_val = f1_matrix[i, j]
            bw_val = bw_matrix[i, j]
            if not np.isnan(f1_val):
                bw_str = f"{bw_val*100:+.0f}% BW" if not np.isnan(bw_val) else ""
                cell_text = f"{f1_val:.2f}\n{bw_str}"
                text_color = "black" if 0.35 < f1_val < 0.75 else "white"
                ax.text(j, i, cell_text, ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="#888888")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig1_fidelity_heatmap"))
    print("[FIG1] Saved fidelity heatmap.")


def fig2_bandwidth(cells: pd.DataFrame, df_all: pd.DataFrame, output_dir: str) -> None:
    """Figure 2: Bandwidth bytes/sec grouped by network state and command."""
    if cells.empty:
        print("[FIG2] No data — skipping bandwidth chart.")
        return

    # Estimate window duration per cell for bytes/sec conversion
    # Use total time span / total windows as an approximation
    total_ts_span = df_all["timestamp"].max() - df_all["timestamp"].min() if not df_all.empty else 1.0
    total_windows = max(1, total_ts_span / 0.25)

    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.arange(len(NETWORK_STATES))
    n_cmds  = len(COMMANDS)
    width   = 0.18
    offsets = np.linspace(-(n_cmds - 1) / 2, (n_cmds - 1) / 2, n_cmds) * width

    COMMAND_COLORS = {
        "FULL_ECG":           "#1f77b4",
        "FULL_ECG_PRIORITY":  "#17becf",
        "DOWNSAMPLED_ECG":    "#ff7f0e",
        "SEMANTIC_ALERT":     "#e377c2",
        "SEMANTIC_CRITICAL":  "#d62728",
        "SEMANTIC_SUMMARY":   "#7f7f7f",
    }

    for ci, cmd in enumerate(COMMANDS):
        bw_per_state = []
        for state in NETWORK_STATES:
            row = cells[(cells["network_condition"] == state) & (cells["command"] == cmd)]
            if not row.empty:
                # Scale bytes to per-second
                bw_per_state.append(float(row["bandwidth_bytes"].iloc[0]) / max(1.0, total_ts_span))
            else:
                bw_per_state.append(0.0)

        ax.bar(
            x + offsets[ci],
            bw_per_state,
            width=width,
            label=cmd,
            color=COMMAND_COLORS.get(cmd, "#888888"),
            alpha=0.85,
        )

    # Raw baseline reference line (bytes/sec for 8 devices at ECG rate + 4 slower)
    # 8 devices × 100Hz × 64 bytes = 51 200 bytes/sec approximate upper bound
    raw_bps_ref = 8 * 64 * 100
    ax.axhline(raw_bps_ref, color="black", linestyle="--", linewidth=1.2,
               label=f"Raw baseline ({raw_bps_ref/1000:.0f} KB/s)")

    ax.set_xticks(x)
    ax.set_xticklabels(NETWORK_STATES)
    ax.set_xlabel("Network condition")
    ax.set_ylabel("Bandwidth (bytes/sec)")
    ax.set_title("Bandwidth usage by network condition and command")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.1f} KB"))
    ax.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig2_bandwidth_by_mode"))
    print("[FIG2] Saved bandwidth chart.")


def fig3_confidence_vs_f1(cells: pd.DataFrame, output_dir: str) -> None:
    """Figure 3: Decode confidence vs clinical state F1 scatter with regression."""
    valid = cells.dropna(subset=["decode_confidence_mean", "clinical_state_f1"])
    if valid.empty:
        print("[FIG3] No encoded/confidence data — skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    for state, grp in valid.groupby("network_condition"):
        for cmd, sub in grp.groupby("command"):
            marker = COMMAND_MARKERS.get(cmd, "o")
            color  = STATE_COLORS.get(state, "#888888")
            ax.scatter(
                sub["decode_confidence_mean"],
                sub["clinical_state_f1"],
                c=color, marker=marker, s=80, alpha=0.85,
                label=f"{state}/{cmd}",
            )

    # Linear regression
    x_all = valid["decode_confidence_mean"].values
    y_all = valid["clinical_state_f1"].values
    if len(x_all) >= 2 and _SCIPY_OK:
        slope, intercept, r, p, _ = _scipy_stats.linregress(x_all, y_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", linewidth=1.2,
                label=f"Regression (r={r:.2f}, p={p:.3f})")

    ax.set_xlabel("Mean decode confidence")
    ax.set_ylabel("Clinical state F1 (weighted)")
    ax.set_title("Decode confidence vs clinical state F1")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Legend: deduplicate by keeping state+command unique
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc="lower right", fontsize=7, ncol=2)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig3_confidence_vs_f1"))
    print("[FIG3] Saved confidence vs F1 scatter.")


def fig4_latency_cdf(commands: pd.DataFrame, output_dir: str) -> None:
    """Figure 4: Latency CDF per command type with SLA badge."""
    if commands.empty or "latency_ms" not in commands.columns:
        print("[FIG4] No command_log data — skipping latency CDF.")
        return

    commands = commands.dropna(subset=["latency_ms"])
    if commands.empty:
        print("[FIG4] No valid latency_ms values — skipping.")
        return

    cmd_types = sorted(commands["command"].dropna().unique())
    COMMAND_COLORS = {
        "FULL_ECG":           "#1f77b4",
        "FULL_ECG_PRIORITY":  "#17becf",
        "DOWNSAMPLED_ECG":    "#ff7f0e",
        "SEMANTIC_ALERT":     "#e377c2",
        "SEMANTIC_CRITICAL":  "#d62728",
        "SEMANTIC_SUMMARY":   "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(7.5, 6))

    sla_compliance_critical: Optional[float] = None

    for cmd in cmd_types:
        subset = commands[commands["command"] == cmd]["latency_ms"].values
        if len(subset) == 0:
            continue
        sorted_vals = np.sort(subset)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        color = COMMAND_COLORS.get(cmd, "#888888")
        ax.step(sorted_vals, cdf, where="post", label=cmd, color=color, linewidth=2.0)

        if cmd == "SEMANTIC_CRITICAL":
            within_sla = float(np.mean(subset <= SLA_MS)) * 100.0
            sla_compliance_critical = within_sla

    # Calculate overall SLA compliance
    within_sla_all = float(np.mean(commands["latency_ms"] <= SLA_MS)) * 100.0

    # Add a styled badge for the SLA in the empty top-left region
    if sla_compliance_critical is not None:
        badge_text = (
            f"SLA Target: {SLA_MS:.0f} ms\n"
            f"Overall Compliance: {within_sla_all:.1f}%\n"
            f"Critical Compliance: {sla_compliance_critical:.1f}%"
        )
    else:
        badge_text = (
            f"SLA Target: {SLA_MS:.0f} ms\n"
            f"Overall Compliance: {within_sla_all:.1f}%"
        )

    props = dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#cfd8dc', alpha=0.95, zorder=5)
    ax.text(0.04, 0.96, badge_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=props, weight='bold', color='#2c3e50')

    ax.set_xlabel("Command latency (ms)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Command Latency CDF by Command Type")
    
    # Zoom x-axis to the actual data range dynamically (plus a small margin)
    max_lat = commands["latency_ms"].max()
    ax.set_xlim(0, max(80.0, float(max_lat) * 1.15))
    ax.set_ylim(0, 1.05)
    
    # Clean grid and spines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    if 'top' in ax.spines:
        ax.spines['top'].set_visible(False)
    if 'right' in ax.spines:
        ax.spines['right'].set_visible(False)

    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "fig4_latency_cdf"))
    _save(fig, os.path.join(output_dir, "fig_latency_cdf"))
    print("[FIG4] Saved latency CDF as fig4_latency_cdf and fig_latency_cdf.")


# ---------------------------------------------------------------------------
# STEP 5 — Summary report
# ---------------------------------------------------------------------------

def print_and_save_summary(
    cells: pd.DataFrame,
    df_all: pd.DataFrame,
    commands: pd.DataFrame,
    output_dir: str,
) -> None:
    lines = []

    def emit(line: str = "") -> None:
        print(line)
        lines.append(line)

    emit("=" * 62)
    emit("  Semantic Fidelity Summary Report")
    emit("=" * 62)

    # Overall F1 — encoded packets only, aggregated at 250ms-window level
    enc_all = df_all[df_all["encoded"] == True] if "encoded" in df_all.columns else df_all
    if not enc_all.empty and "label" in enc_all.columns and "decoded_label" in enc_all.columns:
        df_win = enc_all.copy()
        df_win["_win_key"] = (df_win["timestamp"] // 0.25).astype("int64")
        win_agg_all = (
            df_win.groupby("_win_key")
            .agg(
                true_label=("label", _worst_label),
                dec_label=("decoded_label", _worst_label),
            )
        )
        true_all = win_agg_all["true_label"].tolist()
        dec_all  = win_agg_all["dec_label"].tolist()
        try:
            overall_f1 = f1_score(
                true_all, dec_all,
                labels=["NORMAL", "ALERT", "CRITICAL"],
                average="weighted",
                zero_division=0,
            )
        except Exception:
            overall_f1 = float("nan")
        n_win_total = len(win_agg_all)
    else:
        overall_f1  = float("nan")
        n_win_total = 0

    emit(f"Overall clinical state F1 (all modes, all conditions): {overall_f1:.3f}"
         f"  [{n_win_total} windows @ 0.25 s]"
         if not np.isnan(overall_f1) else "Overall clinical state F1: N/A")

    # Best/worst under Critical network
    crit_cells = cells[cells["network_condition"] == "Critical"].dropna(
        subset=["clinical_state_f1"]
    )
    if not crit_cells.empty:
        best  = crit_cells.loc[crit_cells["clinical_state_f1"].idxmax()]
        worst = crit_cells.loc[crit_cells["clinical_state_f1"].idxmin()]
        emit(f"Best mode under critical network:  {best['command']}"
             f"  —  F1={best['clinical_state_f1']:.3f}"
             f",  BW reduction={best['bandwidth_reduction']*100:.1f}%")
        emit(f"Worst mode under critical network: {worst['command']}"
             f"  —  F1={worst['clinical_state_f1']:.3f}")
    else:
        emit("Best/worst under critical network: no data")

    # SLA compliance for SEMANTIC_CRITICAL
    if not commands.empty and "latency_ms" in commands.columns:
        sc = commands[commands["command"] == "SEMANTIC_CRITICAL"]["latency_ms"].dropna()
        if len(sc) > 0:
            sla_pct = float(np.mean(sc <= SLA_MS)) * 100.0
            emit(f"SLA compliance (SEMANTIC_CRITICAL within {SLA_MS:.0f} ms): {sla_pct:.1f}%")
        else:
            emit("SLA compliance (SEMANTIC_CRITICAL): no events")
    else:
        emit("SLA compliance (SEMANTIC_CRITICAL): no command_log data")

    # Mean decode confidence
    if not cells.empty:
        conf_vals = cells["decode_confidence_mean"].dropna()
        if len(conf_vals) > 0:
            emit(f"Mean decode confidence (encoded cells): {conf_vals.mean():.3f}")
        else:
            emit("Mean decode confidence: no encoded packets found")
    else:
        emit("Mean decode confidence: N/A")

    emit("=" * 62)

    txt_path = os.path.join(output_dir, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[REPORT] Saved to {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic fidelity evaluation — no PyTorch required."
    )
    parser.add_argument(
        "--logs-dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "logs",
        ),
        help="Directory containing sender_log_dev*.csv, network_telemetry.csv, command_log.csv",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "plots", "semantic_fidelity",
        ),
        help="Directory to write figures and summary_report.txt",
    )
    parser.add_argument(
        "--enforce-sla", action="store_true",
        help="Fail with non-zero exit code when SEMANTIC_CRITICAL SLA is below threshold.",
    )
    parser.add_argument(
        "--min-sla-pct", type=float, default=95.0,
        help="Minimum required SEMANTIC_CRITICAL SLA compliance percentage for --enforce-sla.",
    )
    args = parser.parse_args()

    logs_dir   = args.logs_dir
    output_dir = _ensure(args.output_dir)

    print(f"[INFO] Logs dir   : {logs_dir}")
    print(f"[INFO] Output dir : {output_dir}")

    # --- Load ---
    print("\n[STEP 1] Loading data...")
    df_sender  = load_sender_logs(logs_dir)
    df_tele    = load_telemetry(logs_dir)
    df_cmds    = load_command_log(logs_dir)

    if df_sender.empty:
        sys.exit("[ERROR] No sender log files found. Cannot continue.")

    print(f"  Sender packets : {len(df_sender):,}")
    print(f"  Telemetry rows : {len(df_tele):,}")
    print(f"  Command events : {len(df_cmds):,}")

    # --- Align ---
    print("\n[STEP 2] Aligning packets...")
    df_merged = align_packets(df_sender, df_tele, df_cmds)
    df_merged = assign_decoded_labels(df_merged, df_tele)
    print(f"  Merged rows    : {len(df_merged):,}")
    print(f"  Encoded packets: {int(df_merged['encoded'].sum()):,}")

    # --- Cell metrics ---
    print("\n[STEP 3] Computing per-cell metrics...")
    cells = compute_cell_metrics(df_merged)
    if cells.empty:
        print("[WARN] No cell metrics computed — check logs.")
    else:
        cols = ["network_condition", "command", "clinical_state_f1",
                "bandwidth_reduction", "decode_confidence_mean", "n_windows"]
        print(cells[[c for c in cols if c in cells.columns]].to_string(index=False))

    # --- Figures ---
    print("\n[STEP 4] Generating figures...")
    fig1_heatmap(cells, output_dir)
    fig2_bandwidth(cells, df_merged, output_dir)
    fig3_confidence_vs_f1(cells, output_dir)
    fig4_latency_cdf(df_cmds, output_dir)

    # --- Summary ---
    print("\n[STEP 5] Summary:")
    print_and_save_summary(cells, df_merged, df_cmds, output_dir)

    if args.enforce_sla:
        sla_pct = semantic_critical_sla_pct(df_cmds)
        if sla_pct is None:
            print("[SLA ENFORCEMENT] FAIL: no SEMANTIC_CRITICAL latency samples found.")
            raise SystemExit(2)
        if sla_pct < args.min_sla_pct:
            print(
                "[SLA ENFORCEMENT] FAIL: "
                f"SEMANTIC_CRITICAL SLA {sla_pct:.1f}% < required {args.min_sla_pct:.1f}%"
            )
            raise SystemExit(2)
        print(
            "[SLA ENFORCEMENT] PASS: "
            f"SEMANTIC_CRITICAL SLA {sla_pct:.1f}% >= required {args.min_sla_pct:.1f}%"
        )


if __name__ == "__main__":
    main()
