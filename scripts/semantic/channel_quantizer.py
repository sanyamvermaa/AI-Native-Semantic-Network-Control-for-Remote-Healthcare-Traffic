#!/usr/bin/env python3
"""
Channel quantizer for variable-fidelity semantic payloads.

Converts a full 16-dim latent vector ``z`` (produced by SemanticEncoder.encode())
into a compact indexed representation sized according to the current network
command, then serialises / deserialises the result as JSON bytes.

Key functions
-------------
quantize(z, command)
    → List[Tuple[int, float]]  (indexed sparse representation)

dequantize(indexed_z, full_dim)
    → List[float]              (full-length vector, zeros in missing dims)

encode_payload(...)
    → bytes                    (JSON-encoded semantic packet)

decode_payload(data)
    → Optional[Dict]           (parsed packet with "z_full" inserted)

Run the module directly for a self-test:
    python3 scripts/semantic/channel_quantizer.py
"""

import json
import math
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Command → number of latent dims to transmit
# ---------------------------------------------------------------------------
# Mirrors the COMMAND_SEMANTIC_MAP strings used in health_sender.py.
# All commands with fewer than 16 dims transmit only the top-|n| components
# selected by absolute value (most informative dims first).

LATENT_DIMS_BY_COMMAND: Dict[str, int] = {
    "FULL_ECG":           16,
    "FULL_ECG_PRIORITY":  16,
    "DOWNSAMPLED_ECG":    8,
    "SEMANTIC_ALERT":     8,
    "SEMANTIC_CRITICAL":  4,
    "SEMANTIC_SUMMARY":   2,
}

_DEFAULT_LATENT_DIM = 16
_DEFAULT_N_DIMS     = 16  # fallback if command not in the dict


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def quantize(z: List[float], command: str) -> List[Tuple[int, float]]:
    """
    Select the top-``n`` components of ``z`` by absolute value and return
    them as ``(original_index, value)`` pairs sorted by descending |value|.

    Parameters
    ----------
    z : List[float]
        Full latent vector (typically length 16).
    command : str
        Network command string.  Determines how many dims to keep.

    Returns
    -------
    List[Tuple[int, float]]
        At most ``n`` ``(index, value)`` pairs.  May be fewer if ``len(z) < n``.
    """
    n_dims = LATENT_DIMS_BY_COMMAND.get(command, _DEFAULT_N_DIMS)
    n_dims = min(n_dims, len(z))

    # Sort indices by descending |value|, take top-n
    ranked = sorted(range(len(z)), key=lambda i: abs(z[i]), reverse=True)
    top    = ranked[:n_dims]

    # Return in ascending index order so the receiver can reconstruct cheaply
    top.sort()
    return [(i, float(z[i])) for i in top]


def dequantize(
    indexed_z: List[Tuple[int, float]],
    full_dim: int = _DEFAULT_LATENT_DIM,
) -> List[float]:
    """
    Reconstruct a full-length vector from a sparse indexed representation.

    Parameters
    ----------
    indexed_z : List[Tuple[int, float]]
        ``(index, value)`` pairs as returned by ``quantize()``.
    full_dim : int
        Length of the output vector (default 16).

    Returns
    -------
    List[float]
        Length ``full_dim`` vector; positions not present in ``indexed_z``
        are zero.
    """
    result = [0.0] * full_dim
    for idx, val in indexed_z:
        if 0 <= idx < full_dim:
            result[idx] = float(val)
    return result


# ---------------------------------------------------------------------------
# Payload serialisation
# ---------------------------------------------------------------------------

def encode_payload(
    device_id:   int,
    seq:         int,
    ts:          float,
    device_type: str,
    z:           List[float],
    command:     str,
    label:       str,
) -> bytes:
    """
    Serialise a semantic packet as UTF-8 JSON bytes.

    The latent vector is quantised according to ``command`` before encoding.
    The payload format is intentionally compact: only required fields are
    included.

    Packet schema
    -------------
    {
      "device_id":   int,
      "seq":         int,
      "ts":          float,
      "device_type": str,
      "encoded":     true,
      "z":           [[index, value], ...],   ← sparse indexed representation
      "n_dims":      int,                     ← number of dims actually sent
      "label":       str,                     ← current clinical label
      "command":     str
    }

    Parameters
    ----------
    device_id : int
    seq : int
        Sequence number.
    ts : float
        Epoch timestamp (seconds).
    device_type : str
    z : List[float]
        Full latent vector from SemanticEncoder.encode().
    command : str
    label : str
        Clinical label for this window ("NORMAL" / "ALERT" / "CRITICAL").

    Returns
    -------
    bytes
        UTF-8 encoded JSON.
    """
    indexed_z = quantize(z, command)
    payload = {
        "device_id":   device_id,
        "seq":         seq,
        "ts":          round(ts, 6),
        "device_type": device_type,
        "encoded":     True,
        "z":           [[i, round(v, 6)] for i, v in indexed_z],
        "n_dims":      len(indexed_z),
        "label":       label,
        "command":     command,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_payload(data: bytes) -> Optional[Dict]:
    """
    Deserialise a payload produced by ``encode_payload()``.

    If the packet has ``"encoded": true``, the sparse ``z`` list is dequantised
    and stored in ``"z_full"`` (a full-length ``List[float]``).

    If the packet does not have ``"encoded": true`` (i.e. it is a legacy RAW
    or DELTA packet), returns ``None`` so the caller can fall back to its
    normal processing path.

    Parameters
    ----------
    data : bytes
        Raw bytes received from the network.

    Returns
    -------
    Optional[Dict]
        Parsed packet dict with ``"z_full"`` added, or ``None`` for legacy
        packets or parse errors.
    """
    try:
        packet = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(packet, dict) or not packet.get("encoded"):
        return None

    raw_z = packet.get("z", [])
    try:
        indexed_z: List[Tuple[int, float]] = [
            (int(pair[0]), float(pair[1])) for pair in raw_z
        ]
    except (TypeError, IndexError, ValueError):
        return None

    packet["z_full"] = dequantize(indexed_z, full_dim=_DEFAULT_LATENT_DIM)
    return packet


# ---------------------------------------------------------------------------
# __main__ — self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(0)

    print("=== channel_quantizer self-test ===\n")

    # Build a fake 16-dim latent vector
    fake_z: List[float] = [random.uniform(-1.0, 1.0) for _ in range(16)]
    print(f"Original z ({len(fake_z)} dims): {[round(v, 4) for v in fake_z]}")

    # --- quantize to 4 dims (SEMANTIC_CRITICAL)
    command  = "SEMANTIC_CRITICAL"
    n_keep   = LATENT_DIMS_BY_COMMAND[command]
    q        = quantize(fake_z, command)
    print(f"\nquantize(z, '{command}') → {n_keep} dims:")
    for idx, val in q:
        print(f"  [{idx}] = {val:.6f}")

    # --- encode payload
    raw_bytes = encode_payload(
        device_id=0, seq=1, ts=1743388800.0,
        device_type="ECG", z=fake_z,
        command=command, label="CRITICAL",
    )
    print(f"\nencode_payload() → {len(raw_bytes)} bytes")
    print(f"  {raw_bytes.decode('utf-8')}")

    # --- decode payload
    parsed = decode_payload(raw_bytes)
    assert parsed is not None, "FAIL: decode_payload returned None"

    z_full = parsed["z_full"]
    assert len(z_full) == 16, f"FAIL: z_full length {len(z_full)} != 16"

    # Verify that the n_keep transmitted positions match the original values
    non_zero_positions = [i for i, v in enumerate(z_full) if v != 0.0]
    assert len(non_zero_positions) == n_keep, (
        f"FAIL: expected {n_keep} non-zero dims, got {len(non_zero_positions)}"
    )

    transmitted_indices = {i for i, _ in q}
    recovered_indices   = set(non_zero_positions)
    assert transmitted_indices == recovered_indices, (
        f"FAIL: index mismatch — transmitted {transmitted_indices}, "
        f"recovered {recovered_indices}"
    )

    for idx, val in q:
        assert math.isclose(z_full[idx], val, rel_tol=1e-5), (
            f"FAIL: z_full[{idx}]={z_full[idx]:.6f} != original {val:.6f}"
        )

    # --- dequantize — use the JSON-parsed (rounded) indexed_z so values match z_full
    indexed_z_from_parsed = [
        (int(pair[0]), float(pair[1])) for pair in parsed["z"]
    ]
    z_recovered = dequantize(indexed_z_from_parsed, full_dim=16)
    assert z_recovered == z_full, "FAIL: dequantize result differs from z_full"

    # --- decode_payload on a legacy (non-encoded) packet → None
    legacy  = json.dumps({"device_id": 0, "value": 72, "label": "NORMAL"}).encode()
    result  = decode_payload(legacy)
    assert result is None, "FAIL: legacy packet should return None"

    # --- decode_payload on corrupt bytes → None
    corrupt = b"not-json"
    assert decode_payload(corrupt) is None, "FAIL: corrupt bytes should return None"

    print(f"\nz_full (recovered): {[round(v, 4) for v in z_full]}")
    print(f"\nAll assertions passed.")
    print("PASS")
