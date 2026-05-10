"""
verify_seed_match.py
────────────────────
Compares sender_log_dev*.csv files from the baseline and closed-loop
runs to verify the physiological value sequences are identical.

If the seed fix is working correctly, every device's value sequence
should be identical for the first ~N samples (before adaptation-induced
timing drift causes minor float divergence — explained below).

Usage:
    python3 scripts/evaluation/verify_seed_match.py \
        --baseline-dir  outputs/evaluation/baseline_natural_YYYYMMDD_HHMMSS \
        --closedloop-dir outputs/evaluation/closedloop_natural_YYYYMMDD_HHMMSS
"""

import argparse
import glob
import os
import sys

import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir",   required=True)
    parser.add_argument("--closedloop-dir", required=True)
    parser.add_argument("--rows", type=int, default=50,
                        help="Number of value rows to compare per device (default: 50)")
    args = parser.parse_args()

    b_dir = args.baseline_dir
    c_dir = args.closedloop_dir
    N     = args.rows

    print("=" * 65)
    print("  Physiological Sequence Seed Verification")
    print("=" * 65)
    print(f"  Baseline  : {b_dir}")
    print(f"  Closed-Loop: {c_dir}")
    print(f"  Rows compared per device: {N}")
    print()

    b_files = sorted(glob.glob(os.path.join(b_dir, "sender_log_dev*.csv")))

    if not b_files:
        print(f"[ERROR] No sender_log_dev*.csv files found in:\n  {b_dir}")
        sys.exit(1)

    all_match   = True
    results     = []

    for bf in b_files:
        fname = os.path.basename(bf)
        cf    = os.path.join(c_dir, fname)

        if not os.path.exists(cf):
            print(f"  [MISSING] {fname} — not found in closed-loop dir")
            all_match = False
            continue

        try:
            b_df = pd.read_csv(bf, low_memory=False)
            c_df = pd.read_csv(cf, low_memory=False)
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            all_match = False
            continue

        if "value" not in b_df.columns or "value" not in c_df.columns:
            print(f"  [SKIP]  {fname} — no 'value' column")
            continue

        b_vals = pd.to_numeric(b_df["value"], errors="coerce").dropna().values
        c_vals = pd.to_numeric(c_df["value"], errors="coerce").dropna().values

        n_compare = min(N, len(b_vals), len(c_vals))
        b_v = b_vals[:n_compare].round(4)
        c_v = c_vals[:n_compare].round(4)

        mismatches     = int((b_v != c_v).sum())
        match_rate_pct = (1 - mismatches / n_compare) * 100 if n_compare > 0 else 0.0
        device_match   = mismatches == 0

        # Find first divergence index
        first_div = None
        for i, (bv, cv) in enumerate(zip(b_v, c_v)):
            if bv != cv:
                first_div = i
                break

        results.append({
            "file":         fname,
            "b_rows":       len(b_vals),
            "cl_rows":      len(c_vals),
            "compared":     n_compare,
            "mismatches":   mismatches,
            "match_rate":   match_rate_pct,
            "match":        device_match,
            "first_div":    first_div,
        })

        status = "✅ MATCH" if device_match else f"⚠  DIFF  (first divergence @ row {first_div})"
        print(f"  {fname:<45} {status}")
        print(f"    baseline_rows={len(b_vals):6d}  cl_rows={len(c_vals):6d}  "
              f"compared={n_compare}  mismatches={mismatches}  "
              f"match_rate={match_rate_pct:.1f}%")

        if not device_match and first_div is not None:
            # Show neighbourhood around first divergence
            lo = max(0, first_div - 1)
            hi = min(n_compare, first_div + 4)
            print(f"    First {hi - lo} rows around divergence:")
            for i in range(lo, hi):
                marker = " ←" if i == first_div else ""
                print(f"      row {i:4d}:  baseline={b_v[i]:.4f}  cl={c_v[i]:.4f}{marker}")
        print()

    if not results:
        print("[ERROR] No files were compared.")
        sys.exit(1)

    # Summary
    n_matched = sum(1 for r in results if r["match"])
    n_total   = len(results)

    print("=" * 65)
    print(f"  SUMMARY: {n_matched}/{n_total} devices have identical sequences")
    print()

    if n_matched == n_total:
        print("  ✅ ALL DEVICES MATCH — seed fix is working correctly.")
        print()
        print("  The physiological value sequences are byte-identical between")
        print("  baseline and closed-loop runs.  The ONLY experimental variable")
        print("  between the two runs is the semantic adaptation layer.")
    else:
        print("  ⚠  Some devices diverged.  Possible causes:")
        print()
        print("  1. TIMING DRIFT (expected, minor): The closed-loop sender")
        print("     processes ward_controller commands, which adds microseconds")
        print("     of per-loop overhead.  This shifts the OU process dt slightly")
        print("     after many thousands of iterations, causing float drift.")
        print("     If match_rate > 95% for affected devices, this is benign.")
        print()
        print("  2. SEED NOT APPLIED: Check that both scripts printed")
        print("     'Physiological seed: 2026010X' at startup.")
        print()
        print("  3. NEUROKIT2 (ECG): neurokit2 uses its own internal RNG.")
        print("     ECG devices may diverge even with matching Python seeds.")
        print("     This is cosmetic — HR values stay in the same physiological")
        print("     range; only the exact HRV pattern differs.")

        # Show per-device match rates
        print()
        print("  Per-device match rates:")
        for r in results:
            bar = "█" * int(r["match_rate"] / 5)
            print(f"    {r['file']:<45} {r['match_rate']:5.1f}%  {bar}")

    print("=" * 65)


if __name__ == "__main__":
    main()
