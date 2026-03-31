#!/usr/bin/env python3
"""Step 2 unit test: encoder -> quantize -> decode round-trip."""
import os, sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts", "semantic"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts", "closed_loop"))

from semantic_encoder import SemanticEncoder
from channel_quantizer import encode_payload, decode_payload

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "semantic")

def check(condition, msg):
    if not condition:
        print(f"FAIL: {msg}")
        sys.exit(1)

# ── ECG CRITICAL (high HR = 140 bpm, window=200) ─────────────────────────────
print("=== ECG CRITICAL window ===")
enc = SemanticEncoder("ECG", MODELS_DIR)
check(enc._enc is not None, "SemanticEncoder must load encoder model")
print(f"  model loaded, window_size={enc.window_size}")

for _ in range(200):
    enc.push(140.0)

check(enc.ready, "SemanticEncoder.ready must be True after filling buffer")

z = enc.encode()
check(z is not None, "encode() returned None")
check(len(z) == 16, f"z should have 16 dims, got {len(z)}")
print(f"  z[:4] = {[round(v,4) for v in z[:4]]}")

# quantize to SEMANTIC_CRITICAL (4 dims)
payload = encode_payload(0, 1, 1234567890.0, "ECG", z, "SEMANTIC_CRITICAL", "CRITICAL")
print(f"  payload size: {len(payload)} bytes")

result = decode_payload(payload)
check(result is not None, "decode_payload returned None")
z_full = result["z_full"]
check(len(z_full) == 16, f"z_full should be 16 dims, got {len(z_full)}")

dec = enc.decode(z_full)
print(f"  decoded state: {dec['clinical_state']}  confidence: {dec['confidence']:.3f}")
check(dec["clinical_state"] in ("CRITICAL", "ALERT"),
      f"Expected CRITICAL or ALERT, got {dec['clinical_state']}")

# ── ECG NORMAL (low normal HR = 72 bpm, window=200) ──────────────────────────
print("\n=== ECG NORMAL window ===")
enc2 = SemanticEncoder("ECG", MODELS_DIR)
for _ in range(200):
    enc2.push(72.0)
z2 = enc2.encode()
dec2 = enc2.decode(z2)
print(f"  decoded state: {dec2['clinical_state']}  confidence: {dec2['confidence']:.3f}")

# ── BloodPressure NORMAL (120 mmHg, window=200) ───────────────────────────────
print("\n=== BloodPressure NORMAL window ===")
enc_bp = SemanticEncoder("BloodPressure", MODELS_DIR)
for _ in range(200):
    enc_bp.push(120.0)
z_bp = enc_bp.encode()
dec_bp = enc_bp.decode(z_bp)
print(f"  decoded state: {dec_bp['clinical_state']}  confidence: {dec_bp['confidence']:.3f}")

# ── BloodPressure CRITICAL (180 mmHg, window=200) ────────────────────────────
print("\n=== BloodPressure CRITICAL window ===")
enc_bp2 = SemanticEncoder("BloodPressure", MODELS_DIR)
for _ in range(200):
    enc_bp2.push(185.0)
z_bp2 = enc_bp2.encode()
dec_bp2 = enc_bp2.decode(z_bp2)
print(f"  decoded state: {dec_bp2['clinical_state']}  confidence: {dec_bp2['confidence']:.3f}")
check(dec_bp2["clinical_state"] in ("CRITICAL", "ALERT"),
      f"Expected CRITICAL or ALERT for 185 mmHg, got {dec_bp2['clinical_state']}")

print("\n✓ ALL STEP 2 CHECKS PASSED")
