#!/usr/bin/env python3
"""
generate_stress_manifest.py
───────────────────────────────────────────────────────────────
Pre-generates the complete ward_natural_scenario tc netem sequence
as a CSV file.  Both run_natural_stress_baseline.sh and
run_natural_stress_closedloop.sh replay this file instead of
computing random dwell noise at runtime.

This guarantees byte-identical network conditions across both runs
regardless of bash version, PID, or process startup timing.

Output: scripts/evaluation/scenario_manifest.csv
  Columns: step, phase, loss_pct, delay_ms, jitter_ms, sleep_s

Usage:
    python3 scripts/evaluation/generate_stress_manifest.py
    python3 scripts/evaluation/generate_stress_manifest.py --seed 20260101
    python3 scripts/evaluation/generate_stress_manifest.py --preview
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# ── Scenario definition ───────────────────────────────────────────────────────
# Mirrors ward_natural_scenario() exactly — change here, change in both scripts.

DWELL_STEP = 8          # seconds between each dwell noise application
DWELL_NOISE_LO = 0.92   # multiplier lower bound (loss)
DWELL_NOISE_HI = 0.08   # +range (0.92 + r * 0.16) → [0.92, 1.08]

# Each phase: (name, to_loss, to_delay, to_jitter, interp_steps, interp_sleep, dwell_sec)
PHASES = [
    ("Phase 1 — Morning stable (quiet ward)",           0.3,  12.0,  2.0,  3, 5,  65),
    ("Phase 2 — Gradual degradation (shift change)",    5.0,  70.0, 14.0, 10, 4,  40),
    ("Phase 3 — Brief partial recovery",                2.0,  35.0,  7.0,  5, 4,  20),
    ("Phase 4 — Escalation to critical (busy hour)",   14.0, 200.0, 55.0, 10, 5,  30),
    ("Phase 5 — Sustained critical (peak congestion)", 18.0, 260.0, 70.0,  4, 4,  64),
    ("Phase 6 — Slow recovery to Unstable",             4.0,  55.0, 11.0, 10, 5,  20),
    ("Phase 7 — Stable recovery window",                0.5,  15.0,  3.0,  6, 4,  46),
    ("Phase 8 — Sharp interference burst (equipment)", 20.0, 220.0, 72.0,  3, 3,  21),
    ("Phase 9 — Fast recovery post-burst",              1.0,  20.0,  4.0,  5, 4,  20),
    ("Phase 10 — Stable finish",                        0.2,  10.0,  2.0,  4, 4,  44),
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_manifest(seed: int) -> list:
    """Returns list of (step, phase_name, loss, delay, jitter, sleep_s)."""
    rng = random.Random(seed)
    rows = []
    step = 0

    cur_loss   = 0.3
    cur_delay  = 12.0
    cur_jitter = 2.0

    for phase_name, to_loss, to_delay, to_jitter, n_steps, step_sleep, dwell_sec in PHASES:

        # ── interpolate_to ───────────────────────────────────────────────────
        for i in range(1, n_steps + 1):
            frac      = i / n_steps
            new_loss  = clamp(cur_loss  + (to_loss  - cur_loss)  * frac, 0.001, 30.0)
            new_delay = clamp(cur_delay + (to_delay - cur_delay) * frac, 1.0, 1000.0)
            new_jit   = clamp(cur_jitter + (to_jitter - cur_jitter) * frac, 0.0, 90.0)
            rows.append((step, phase_name,
                         round(new_loss, 3), round(new_delay, 1), round(new_jit, 1),
                         step_sleep))
            step += 1

        cur_loss, cur_delay, cur_jitter = to_loss, to_delay, to_jitter

        # ── dwell ────────────────────────────────────────────────────────────
        elapsed = 0
        while elapsed + DWELL_STEP <= dwell_sec:
            # Same noise formula as the shell scripts
            rl = cur_loss   * (0.92 + rng.random() * 0.16)
            dl = cur_delay  * (0.93 + rng.random() * 0.14)
            jl = cur_jitter * (0.90 + rng.random() * 0.20)

            rl = clamp(rl, 0.001, 30.0)
            dl = max(dl, 1.0)
            jl = 0.0 if jl < 0.5 else clamp(jl, 0.0, 90.0)

            rows.append((step, phase_name,
                         round(rl, 3), round(dl, 1), round(jl, 1),
                         DWELL_STEP))
            step += 1
            elapsed += DWELL_STEP

        remaining = dwell_sec - elapsed
        if remaining > 0:
            # silent sleep — no stress change, not a row
            pass

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic stress scenario manifest for controlled experiments."
    )
    parser.add_argument("--seed",    type=int, default=20260101,
                        help="Random seed for dwell noise (default: 20260101)")
    parser.add_argument("--preview", action="store_true",
                        help="Print all steps to stdout instead of writing the file")
    parser.add_argument("--out",     type=str, default="",
                        help="Output path (default: scripts/evaluation/scenario_manifest.csv)")
    args = parser.parse_args()

    rows = generate_manifest(args.seed)

    if args.preview:
        print(f"{'Step':>5}  {'Loss%':>8}  {'Delay':>8}  {'Jitter':>8}  {'Sleep':>6}  Phase")
        print("-" * 80)
        for step, phase, loss, delay, jitter, slp in rows:
            print(f"{step:>5}  {loss:>8.3f}  {delay:>8.1f}  {jitter:>8.1f}  {slp:>6}  {phase}")
        print(f"\nTotal steps: {len(rows)}")
        return

    # Default output path: same directory as this script
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).parent / "scenario_manifest.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "phase", "loss_pct", "delay_ms", "jitter_ms", "sleep_s"])
        for step, phase, loss, delay, jitter, slp in rows:
            writer.writerow([step, phase, loss, delay, jitter, slp])

    print(f"[OK] Wrote {len(rows)} stress steps -> {out_path}")
    print(f"     Seed: {args.seed}")
    print(f"     Both baseline and closedloop scripts will replay this file.")
    print(f"     Run verify_stress_match.py after paired runs — should be 100% match.")


if __name__ == "__main__":
    main()
