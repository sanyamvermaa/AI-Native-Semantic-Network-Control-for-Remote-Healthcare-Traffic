#!/usr/bin/env python3
"""
test_live_integration.py — End-to-end smoke test for the AI-native semantic codec
closed loop.

Verifies that:
  1. A real health_sender (ECG, --initial-command SEMANTIC_CRITICAL) pushes
     25-sample windows through the trained SemanticEncoder and transmits
     encoded JSON packets via UDP.
  2. A real health_receiver on the same loopback address decodes those packets
     with SemanticEncoder.decode() and writes a row to semantic_drain.csv.

No network namespaces or WSL veth interfaces are required — runs entirely on
127.0.0.1.  Ward-controller commands are not needed; the sender starts
immediately in SEMANTIC_CRITICAL mode.

Usage (project root, WSL):
    python3 scripts/evaluation/test_live_integration.py [--duration 20]

Exit codes:
    0  PASS — drain log has at least one decoded entry
    1  FAIL — see printed reason
"""

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
SENDER_PY  = str(ROOT / "scripts" / "closed_loop" / "health_sender.py")
RECEIVER_PY = str(ROOT / "scripts" / "closed_loop" / "health_receiver.py")

# Use an isolated temp directory so we never stomp on real run data.
TEST_DIR   = str(ROOT / "data" / "integration_test")
DRAIN_LOG  = os.path.join(TEST_DIR, "semantic_drain.csv")

# Port isolated from the live system (default 9000).
LISTEN_PORT = 9001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kill(proc: subprocess.Popen) -> None:
    """Terminate gracefully, then SIGKILL if still alive after 3 s."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _tail_output(proc: subprocess.Popen, label: str, n_lines: int = 8) -> None:
    """Print the last n_lines from a subprocess's stdout pipe."""
    if proc.stdout is None:
        return
    try:
        raw = proc.stdout.read()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        print(f"\n--- {label} (last {n_lines} lines) ---")
        for line in lines[-n_lines:]:
            print(" ", line)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live sender-receiver integration smoke test"
    )
    parser.add_argument(
        "--duration", type=int, default=20,
        help="Seconds to run the sender-receiver pair (default 20)"
    )
    parser.add_argument(
        "--device-type", type=str, default="ECG",
        choices=["ECG", "BloodPressure"],
        help="Device type to simulate (default ECG)"
    )
    parser.add_argument(
        "--command", type=str, default="SEMANTIC_CRITICAL",
        choices=["SEMANTIC_ALERT", "SEMANTIC_CRITICAL", "SEMANTIC_SUMMARY"],
        help="Initial transmission command (default SEMANTIC_CRITICAL)"
    )
    parser.add_argument(
        "--keep-data", action="store_true",
        help="Do not delete the test data directory after the run"
    )
    args = parser.parse_args()

    # --- Prepare test directory ---
    if os.path.isdir(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)

    env = {
        **os.environ,
        "HEALTH_DATA_BASE_DIR": TEST_DIR,
        "RECEIVER_PORT":        str(LISTEN_PORT),
        "SEMANTIC_DRAIN_PATH":  DRAIN_LOG,
        # Suppress ward-controller connection errors (no controller in test)
        "WARD_CONTROLLER_IP":   "127.0.0.1",
        "WARD_CONTROLLER_PORT": "15006",
    }

    python = sys.executable   # same interpreter that launched this script

    # --- Start receiver ---
    print(f"[TEST] Starting receiver on 127.0.0.1:{LISTEN_PORT} ...")
    recv_proc = subprocess.Popen(
        [python, RECEIVER_PY],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    # Give the receiver time to bind before the sender starts sending
    time.sleep(1.5)

    if recv_proc.poll() is not None:
        _tail_output(recv_proc, "RECEIVER crashed at startup")
        print("\n[FAIL] Receiver exited before sender started.")
        sys.exit(1)

    # --- Start sender ---
    print(f"[TEST] Starting sender  (dev_id=10, type={args.device_type}, "
          f"command={args.command}) ...")
    send_proc = subprocess.Popen(
        [
            python, SENDER_PY,
            "--device-id",       "10",
            "--device-type",     args.device_type,
            "--receiver-ip",     "127.0.0.1",
            "--receiver-port",   str(LISTEN_PORT),
            "--base-dir",        TEST_DIR,
            "--initial-command", args.command,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    print(f"[TEST] Running for {args.duration} s — waiting for encoded packets ...")
    deadline = time.time() + args.duration

    # Poll drain log every second for early-success detection
    passed_early = False
    while time.time() < deadline:
        time.sleep(1.0)
        if os.path.isfile(DRAIN_LOG):
            with open(DRAIN_LOG, newline="", encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1   # subtract header row
            if n >= 5:
                print(f"[TEST] Early-success: {n} rows already in drain log.")
                passed_early = True
                break

        # abort if either subprocess died
        if recv_proc.poll() is not None:
            _tail_output(recv_proc, "RECEIVER")
            print("\n[FAIL] Receiver process died during the test.")
            _kill(send_proc)
            sys.exit(1)
        if send_proc.poll() is not None:
            _tail_output(send_proc, "SENDER")
            print("\n[FAIL] Sender process died during the test.")
            _kill(recv_proc)
            sys.exit(1)

    # --- Shut down both processes ---
    print("[TEST] Stopping sender ...")
    _kill(send_proc)
    time.sleep(0.5)
    print("[TEST] Stopping receiver ...")
    _kill(recv_proc)
    time.sleep(0.5)

    if not passed_early:
        _tail_output(send_proc,  "SENDER  (last lines)")
        _tail_output(recv_proc, "RECEIVER (last lines)")

    # --- Inspect drain log ---
    if not os.path.isfile(DRAIN_LOG):
        print(f"\n[FAIL] semantic_drain.csv not created at {DRAIN_LOG}")
        sys.exit(1)

    rows = []
    with open(DRAIN_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    if n == 0:
        print("\n[FAIL] semantic_drain.csv exists but contains 0 decoded entries.")
        print("       Check that models/semantic/ contains trained .pt files.")
        sys.exit(1)

    valid_labels = {"NORMAL", "ALERT", "CRITICAL"}
    bad = [r for r in rows if r.get("decoded_label", "") not in valid_labels]
    if bad:
        print(f"\n[FAIL] {len(bad)} rows have invalid decoded_label: "
              f"{[r['decoded_label'] for r in bad[:3]]}")
        sys.exit(1)

    # --- Print results ---
    counts  = Counter(r["decoded_label"] for r in rows)
    n_dims_vals = [int(r["n_dims"]) for r in rows if r.get("n_dims", "").isdigit()]
    mean_conf   = (
        sum(float(r["confidence"]) for r in rows if r.get("confidence"))
        / max(1, sum(1 for r in rows if r.get("confidence")))
    )

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  PASS — Live semantic closed-loop integration confirmed  ║
╚══════════════════════════════════════════════════════════╝
  Decoded entries in drain log : {n}
  Label breakdown              : {dict(counts)}
  Latent dims transmitted      : {set(n_dims_vals)} (command={args.command})
  Mean decode confidence       : {mean_conf:.3f}
  Drain log path               : {DRAIN_LOG}

  First row : {rows[0]}
  Last  row : {rows[-1]}
""")

    if not args.keep_data:
        shutil.rmtree(TEST_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
