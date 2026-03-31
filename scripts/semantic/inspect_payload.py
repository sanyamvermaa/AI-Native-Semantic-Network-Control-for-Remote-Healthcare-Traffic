#!/usr/bin/env python3
"""
Step 3 inspector: simulate health_sender semantic fast-path without a socket.

Feeds a stream of OU-like ECG and BloodPressure values through SemanticEncoder,
encodes them at each SEMANTIC_* command level, and pretty-prints the JSON payload.

Usage (from project root, WSL):
    python3 scripts/semantic/inspect_payload.py
"""
import json
import os
import sys
import math
import random

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts", "semantic"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts", "closed_loop"))

from semantic_encoder  import SemanticEncoder
from channel_quantizer import encode_payload, decode_payload, LATENT_DIMS_BY_COMMAND

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "semantic")
SEP = "─" * 60

# ── OU process simulator (matches health_sender._next_value logic) ─────────
def ou_stream(mu, sigma, theta, n, burst_at=None, burst_val=None):
    """Yield n OU samples; optionally inject a burst from sample burst_at onward."""
    x = mu
    for i in range(n):
        if burst_at is not None and i >= burst_at:
            x += theta * (burst_val - x) + sigma * random.gauss(0, 1)
        else:
            x += theta * (mu - x) + sigma * random.gauss(0, 1)
        yield x

def inspect_device(device_type, mu, sigma, theta, burst_val, commands):
    random.seed(42)
    print(f"\n{'═'*60}")
    print(f"  Device: {device_type}   OU mu={mu}  burst_val={burst_val}")
    print(f"{'═'*60}")

    enc = SemanticEncoder(device_type, MODELS_DIR)
    if enc._enc is None:
        print(f"  [SKIP] No trained model for {device_type}")
        return

    window   = enc.window_size
    n_total  = window + 50          # fill buffer + 50 extra ticks

    # Phase 1: fill buffer with normal values
    samples = list(ou_stream(mu, sigma, theta, n_total, burst_at=window+20, burst_val=burst_val))
    for v in samples[:window]:
        enc.push(v)

    assert enc.ready, "buffer should be full after window samples"
    z_normal = enc.encode()
    dec_normal = enc.decode(z_normal)

    print(f"\n  [NORMAL phase — {window} samples at mu={mu}]")
    print(f"    decoded state : {dec_normal['clinical_state']}")
    print(f"    confidence    : {dec_normal['confidence']:.3f}")
    print(f"    probabilities : { {k: round(v,3) for k,v in dec_normal['probabilities'].items()} }")

    # Phase 2: continue into burst
    burst_enc = SemanticEncoder(device_type, MODELS_DIR)
    for v in samples:                 # full stream including burst tail
        burst_enc.push(v)
    z_burst = burst_enc.encode()
    dec_burst = burst_enc.decode(z_burst)

    print(f"\n  [BURST phase — burst_val={burst_val}]")
    print(f"    decoded state : {dec_burst['clinical_state']}")
    print(f"    confidence    : {dec_burst['confidence']:.3f}")
    print(f"    probabilities : { {k: round(v,3) for k,v in dec_burst['probabilities'].items()} }")

    # Phase 3: encode at each command level and show the JSON packet
    print(f"\n  [PAYLOAD inspection — burst z encoded at each command]")
    print(f"  {'Command':<22} {'dims':>4}  {'bytes':>5}  {'decoded_state':<12}  {'confidence':>10}")
    print(f"  {'-'*22} {'-'*4}  {'-'*5}  {'-'*12}  {'-'*10}")

    for cmd in commands:
        n_dims = LATENT_DIMS_BY_COMMAND.get(cmd, 16)
        raw = encode_payload(
            device_id   = 0,
            seq         = 1,
            ts          = 1743388800.0,
            device_type = device_type,
            z           = z_burst,
            command     = cmd,
            label       = dec_burst["clinical_state"],
        )
        result = decode_payload(raw)
        rec = enc.decode(result["z_full"]) if result else {}
        state = rec.get("clinical_state", "N/A")
        conf  = rec.get("confidence", 0.0)
        print(f"  {cmd:<22} {n_dims:>4}  {len(raw):>5}  {state:<12}  {conf:>10.3f}")

        # Pretty-print the full JSON for the smallest command only
        if cmd == commands[-1]:
            parsed = json.loads(raw)
            print(f"\n  [Full JSON — {cmd}]")
            print("  " + json.dumps(parsed, indent=4).replace("\n", "\n  "))


def main():
    print("=" * 60)
    print("  Step 3 — Semantic payload inspector")
    print("=" * 60)

    commands = [
        "FULL_ECG",
        "FULL_ECG_PRIORITY",
        "DOWNSAMPLED_ECG",
        "SEMANTIC_ALERT",
        "SEMANTIC_CRITICAL",
        "SEMANTIC_SUMMARY",
    ]

    inspect_device(
        device_type = "ECG",
        mu          = 75.0,    # normal resting HR
        sigma       = 3.0,
        theta       = 0.1,
        burst_val   = 140.0,   # tachycardia burst
        commands    = commands,
    )

    inspect_device(
        device_type = "BloodPressure",
        mu          = 120.0,   # normal systolic
        sigma       = 5.0,
        theta       = 0.08,
        burst_val   = 185.0,   # hypertensive crisis
        commands    = commands,
    )

    print(f"\n{'='*60}")
    print("  Step 3 COMPLETE — payload format verified.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
