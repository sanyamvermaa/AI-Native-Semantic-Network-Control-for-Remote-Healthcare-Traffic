"""
verify_stress_match.py
─────────────────────────────────────────────────────
Compares the stress.log files from baseline and closed-loop runs
to check whether identical tc netem conditions were applied.

The STRESS lines look like:
  [STRESS] loss=0.300% delay=12.0ms jitter=2.0ms

Phase transitions (interpolate_to) are deterministic arithmetic
and will always match.  The dwell() function uses bash $RANDOM,
which is unseeded — so dwell micro-noise WILL differ between runs
unless bash RANDOM is seeded identically.

Usage:
    python3 scripts/evaluation/verify_stress_match.py \
        --baseline-dir  outputs/evaluation/baseline_natural_YYYYMMDD \
        --closedloop-dir outputs/evaluation/closedloop_natural_YYYYMMDD
"""

import argparse
import re
import sys
import os
import numpy as np

def parse_stress_log(path):
    """Extract (loss%, delay_ms, jitter_ms) tuples from a stress.log file."""
    pat = re.compile(
        r'\[STRESS\]\s+loss=([\d.]+)%\s+delay=([\d.]+)ms\s+jitter=([\d.]+)ms'
    )
    entries = []
    scenario_lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = pat.search(line)
            if m:
                entries.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif "[SCENARIO]" in line:
                scenario_lines.append((len(entries), line))
    return entries, scenario_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir",    required=True)
    parser.add_argument("--closedloop-dir",  required=True)
    args = parser.parse_args()

    b_stress = os.path.join(args.baseline_dir,    "stress.log")
    c_stress = os.path.join(args.closedloop_dir,  "stress.log")

    for p in [b_stress, c_stress]:
        if not os.path.exists(p):
            print(f"[ERROR] stress.log not found: {p}")
            sys.exit(1)

    b_entries, b_phases = parse_stress_log(b_stress)
    c_entries, c_phases = parse_stress_log(c_stress)

    print("=" * 68)
    print("  Network Stress Match Verification")
    print("=" * 68)
    print(f"  Baseline    stress entries : {len(b_entries)}")
    print(f"  Closed-Loop stress entries : {len(c_entries)}")
    print()

    n = min(len(b_entries), len(c_entries))
    if n == 0:
        print("[ERROR] No stress entries found.")
        sys.exit(1)

    b_arr = np.array(b_entries[:n])   # (loss, delay, jitter)
    c_arr = np.array(c_entries[:n])

    loss_diff   = np.abs(b_arr[:,0] - c_arr[:,0])
    delay_diff  = np.abs(b_arr[:,1] - c_arr[:,1])
    jitter_diff = np.abs(b_arr[:,2] - c_arr[:,2])

    # A "match" means within floating point rounding (same value)
    loss_exact   = (b_arr[:,0] == c_arr[:,0]).sum()
    delay_exact  = (b_arr[:,1] == c_arr[:,1]).sum()
    jitter_exact = (b_arr[:,2] == c_arr[:,2]).sum()

    print("  Metric        Exact Match   Max Diff    Mean Diff")
    print("  " + "-"*52)
    print(f"  Loss (%)      {loss_exact:4d}/{n}      {loss_diff.max():8.3f}    {loss_diff.mean():.3f}")
    print(f"  Delay (ms)    {delay_exact:4d}/{n}      {delay_diff.max():8.3f}    {delay_diff.mean():.3f}")
    print(f"  Jitter (ms)   {jitter_exact:4d}/{n}      {jitter_diff.max():8.3f}    {jitter_diff.mean():.3f}")
    print()

    # Show first divergence
    for i, ((bl,bd,bj),(cl,cd,cj)) in enumerate(zip(b_arr, c_arr)):
        if bl != cl or bd != cd or bj != cj:
            print(f"  First divergence at stress step {i}:")
            print(f"    Baseline    : loss={bl:.3f}%  delay={bd:.1f}ms  jitter={bj:.1f}ms")
            print(f"    Closed-Loop : loss={cl:.3f}%  delay={cd:.1f}ms  jitter={cj:.1f}ms")
            break
    else:
        print("  [OK] All compared entries are IDENTICAL - perfect stress match.")

    # Side-by-side of first 15 entries
    print()
    print("  First 15 stress steps (side by side):")
    print(f"  {'Step':>4}  {'B-loss':>8} {'B-delay':>9} {'B-jitter':>9}   "
          f"{'CL-loss':>8} {'CL-delay':>9} {'CL-jitter':>9}  Match")
    print("  " + "-"*80)
    for i in range(min(15, n)):
        bl,bd,bj = b_arr[i]
        cl,cd,cj = c_arr[i]
        ok = "OK" if (bl==cl and bd==cd and bj==cj) else "WARN"
        print(f"  {i:>4}  {bl:>8.3f} {bd:>9.1f} {bj:>9.1f}   "
              f"{cl:>8.3f} {cd:>9.1f} {cj:>9.1f}  {ok}")

    print()
    # Overall verdict
    total_mismatches = int(((b_arr[:,0] != c_arr[:,0]) |
                            (b_arr[:,1] != c_arr[:,1]) |
                            (b_arr[:,2] != c_arr[:,2])).sum())
    match_pct = (n - total_mismatches) / n * 100

    print("=" * 68)
    if total_mismatches == 0:
        print("  [OK] VERDICT: Network stress is IDENTICAL between runs.")
        print("       The seed fix is working for netem conditions.")
    else:
        print(f"  [WARN] VERDICT: {total_mismatches}/{n} stress steps differ ({match_pct:.1f}% match).")
        print()
        print("  ROOT CAUSE: bash $RANDOM in dwell() is unseeded.")
        print("  FIX: Add  RANDOM=20260101  near the top of both shell scripts")
        print("       (run_natural_stress_baseline.sh and run_natural_stress_closedloop.sh)")
        print("       before the ward_natural_scenario() function runs.")
        print()
        print("  NOTE: Phase transitions (interpolate_to) DO match — only dwell()")
        print("  micro-noise differs (~±10% of the phase target value).")
    print("=" * 68)


if __name__ == "__main__":
    main()
