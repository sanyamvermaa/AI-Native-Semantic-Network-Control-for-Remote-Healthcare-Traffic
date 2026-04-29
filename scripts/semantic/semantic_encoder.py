#!/usr/bin/env python3
"""
Runtime semantic encoder/decoder wrapper for a single device type.

Loads the TorchScript encoder and decoder saved by train_semantic_codec.py
and exposes a simple push-then-encode/decode interface to health_sender.py.

Usage example:
    enc = SemanticEncoder("ECG", "/path/to/models/semantic")
    for sample in stream:
        enc.push(sample)
        if enc.ready:
            z = enc.encode()          # List[float] of length 16, or None
            result = enc.decode(z)    # Dict with clinical_state, confidence, …
"""

import json
import os
import sys
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: import CLINICAL_THRESHOLDS from health_sender for arithmetic
# fallback when torch is unavailable or the buffer is not full yet.
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CLOSED_LOOP  = os.path.join(_PROJECT_ROOT, "scripts", "closed_loop")

if _CLOSED_LOOP not in sys.path:
    sys.path.insert(0, _CLOSED_LOOP)

_CLINICAL_THRESHOLDS_FALLBACK: Dict[str, Dict] = {}
try:
    from health_sender import CLINICAL_THRESHOLDS as _CT  # type: ignore
    _CLINICAL_THRESHOLDS_FALLBACK = _CT
except ImportError:
    pass  # gracefully degrade; fallback will return 0.5 if thresholds missing

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch  # type: ignore  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# SemanticEncoder
# ---------------------------------------------------------------------------

class SemanticEncoder:
    """
    Per-device-type semantic codec wrapper.

    Parameters
    ----------
    device_type : str
        One of "ECG", "SpO2", "BloodPressure", "Temperature", "Respiration".
    models_dir : str
        Directory that contains enc_{device_type}.pt, dec_{device_type}.pt,
        and metadata.json (written by train_semantic_codec.py).
    """

    def __init__(self, device_type: str, models_dir: str) -> None:
        self.device_type      = device_type
        self.models_dir       = models_dir
        self.torch_available  = _TORCH_AVAILABLE

        # Defaults — overwritten on successful load
        self.window_size:   int         = 10
        self.latent_dim:    int         = 16
        self.label_classes: List[str]   = ["NORMAL", "ALERT", "CRITICAL"]
        self.normal_lo:     float       = 0.0
        self.normal_hi:     float       = 1.0

        self._enc = None
        self._dec = None

        self._load_metadata(models_dir, device_type)

        self.buffer: Deque[float] = deque(maxlen=self.window_size)
        self.ready: bool = False

        if self.torch_available:
            self._load_models(models_dir, device_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_metadata(self, models_dir: str, device_type: str) -> None:
        """Read metadata.json and populate window_size / normal_range."""
        meta_path = os.path.join(models_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            entry = meta.get(device_type, {})
            if "window_size" in entry:
                self.window_size = int(entry["window_size"])
                # Re-size buffer with correct maxlen
                self.buffer = deque(maxlen=self.window_size)
            if "latent_dim" in entry:
                self.latent_dim = int(entry["latent_dim"])
            if "label_classes" in entry:
                self.label_classes = list(entry["label_classes"])
            if "normal_range" in entry:
                nr = entry["normal_range"]
                self.normal_lo = float(nr[0])
                self.normal_hi = float(nr[1])
        except Exception as exc:  # pylint: disable=broad-except
            # Non-fatal: use defaults
            print(f"[SemanticEncoder] WARN: could not read metadata.json: {exc}")

    def _load_models(self, models_dir: str, device_type: str) -> None:
        """Load TorchScript encoder and decoder via torch.jit.load."""
        import torch  # type: ignore

        enc_path = os.path.join(models_dir, f"enc_{device_type}.pt")
        dec_path = os.path.join(models_dir, f"dec_{device_type}.pt")

        if not os.path.isfile(enc_path) or not os.path.isfile(dec_path):
            print(
                f"[SemanticEncoder] WARN: model files not found for {device_type}. "
                "Run train_semantic_codec.py first."
            )
            self.torch_available = False
            return

        try:
            self._enc = torch.jit.load(enc_path, map_location="cpu")
            self._dec = torch.jit.load(dec_path, map_location="cpu")
            self._enc.eval()
            self._dec.eval()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[SemanticEncoder] WARN: could not load models: {exc}")
            self.torch_available = False

    def _normalise(self, value: float) -> float:
        """Map a raw sensor value to [0, 1] using the trained normal range."""
        denom = self.normal_hi - self.normal_lo
        if denom == 0.0:
            return 0.5
        return (value - self.normal_lo) / denom

    def _denormalise(self, x: float) -> float:
        """Invert _normalise."""
        return x * (self.normal_hi - self.normal_lo) + self.normal_lo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, value: float) -> None:
        """
        Append one normalised sample to the rolling buffer.

        Sets ``self.ready = True`` once the buffer is completely filled
        for the first time (i.e. len == window_size).
        """
        self.buffer.append(self._normalise(value))
        if len(self.buffer) == self.window_size:
            self.ready = True

    def encode(self) -> Optional[List[float]]:
        """
        Run the encoder on the current buffer contents.

        Returns
        -------
        List[float] of length ``latent_dim`` (values in [-1, 1]), or
        ``None`` if the buffer is not full or torch is unavailable.
        """
        if not self.ready or not self.torch_available or self._enc is None:
            return None

        import torch  # type: ignore

        x = torch.tensor(
            list(self.buffer), dtype=torch.float32
        ).unsqueeze(0)  # [1, window_size]

        with torch.no_grad():
            z = self._enc(x)  # [1, latent_dim]

        return z.squeeze(0).tolist()

    def decode(self, z_partial: List[float]) -> Dict:
        """
        Run the decoder on a (possibly partial / sparse) latent vector.

        ``z_partial`` may be shorter than ``latent_dim``; missing dims are
        zero-padded.  This mirrors what channel_quantizer.dequantize() does.

        Returns
        -------
        Dict with keys:
          clinical_state  : str  — argmax label ("NORMAL" / "ALERT" / "CRITICAL")
          confidence      : float — max softmax probability
          probabilities   : Dict[str, float] — one entry per label_class
          reconstructed   : Dict[str, float] — {"min","max","mean","std"}
                            in *denormalised* (original sensor) units
        """
        # Build a full-length vector, padding or truncating as needed
        full_z = [0.0] * self.latent_dim
        for i, v in enumerate(z_partial):
            if i >= self.latent_dim:
                break
            full_z[i] = v

        if not self.torch_available or self._dec is None:
            return self._fallback_decode(full_z)

        import torch  # type: ignore

        z_tensor = torch.tensor(full_z, dtype=torch.float32).unsqueeze(0)  # [1,16]

        with torch.no_grad():
            head_a, head_b = self._dec(z_tensor)  # [1,3], [1,4]

        probs  = torch.softmax(head_a, dim=-1).squeeze(0).tolist()
        feats  = head_b.squeeze(0).tolist()  # [min, max, mean, std] normalised

        best_idx    = int(max(range(len(probs)), key=lambda i: probs[i]))
        state       = self.label_classes[best_idx] if best_idx < len(self.label_classes) else "NORMAL"
        confidence  = float(probs[best_idx])

        prob_dict = {
            cls: float(probs[i])
            for i, cls in enumerate(self.label_classes)
            if i < len(probs)
        }

        # Denormalise feature head outputs
        reconstructed = {
            "min":  self._denormalise(feats[0]) if len(feats) > 0 else 0.0,
            "max":  self._denormalise(feats[1]) if len(feats) > 1 else 0.0,
            "mean": self._denormalise(feats[2]) if len(feats) > 2 else 0.0,
            "std":  float(feats[3]) * (self.normal_hi - self.normal_lo) if len(feats) > 3 else 0.0,
        }

        return {
            "clinical_state": state,
            "confidence":     confidence,
            "probabilities":  prob_dict,
            "reconstructed":  reconstructed,
        }

    def clinical_importance_learned(self, value: float) -> float:
        """
        Return a clinical importance score in [0.0, 1.0].

        If the buffer is full and torch is available, creates a temporary
        snapshot of the buffer with ``value`` appended, encodes it and
        returns max(P_ALERT, P_CRITICAL).

        **Important**: this method is READ-ONLY — it does NOT mutate
        ``self.buffer``.  The caller (health_sender main loop) is
        responsible for calling ``self.push(value)`` separately.

        Otherwise falls back to the threshold-based arithmetic from
        health_sender.SemanticBuffer.clinical_importance().

        Parameters
        ----------
        value : float
            Latest raw sensor reading (not normalised).
        """
        if self.torch_available and self.ready and self._enc is not None:
            import torch  # type: ignore

            # Build a temporary window: current buffer + new value (normalised).
            # We do NOT call self.push() — the buffer must stay untouched.
            tmp = list(self.buffer)
            tmp.append(self._normalise(value))
            # Keep only the last window_size elements (same as deque maxlen)
            if len(tmp) > self.window_size:
                tmp = tmp[-self.window_size:]

            x = torch.tensor(tmp, dtype=torch.float32).unsqueeze(0)  # [1, W]
            with torch.no_grad():
                z_t = self._enc(x)  # [1, latent_dim]
            z = z_t.squeeze(0).tolist()

            result = self.decode(z)
            probs  = result.get("probabilities", {})
            return max(
                probs.get("ALERT",    0.0),
                probs.get("CRITICAL", 0.0),
            )
        # ---- arithmetic fallback ----------------------------------------
        return self._threshold_importance(value)

    # ------------------------------------------------------------------
    # Fallback helpers (no-torch or buffer-not-ready)
    # ------------------------------------------------------------------

    def _threshold_importance(self, value: float) -> float:
        """Replicate SemanticBuffer.clinical_importance() arithmetic."""
        t = _CLINICAL_THRESHOLDS_FALLBACK.get(self.device_type, {})
        hi_crit = t.get("critical")
        hi_warn = t.get("warn")
        lo_crit = t.get("low_critical")
        lo_warn = t.get("low_warn")

        if hi_crit is not None and value >= hi_crit:
            return 1.0
        if lo_crit is not None and value <= lo_crit:
            return 1.0
        if hi_warn is not None and value >= hi_warn:
            return 0.7
        if lo_warn is not None and value <= lo_warn:
            return 0.7
        return 0.2

    def _fallback_decode(self, full_z: List[float]) -> Dict:
        """Return a minimal decode result when torch is unavailable."""
        return {
            "clinical_state": "NORMAL",
            "confidence":     0.0,
            "probabilities":  {c: 0.0 for c in self.label_classes},
            "reconstructed":  {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
        }


# ---------------------------------------------------------------------------
# __main__ — quick smoke test (no model files required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[smoke] SemanticEncoder — torch_available:", _TORCH_AVAILABLE)

    # Test with a dummy models dir (no .pt files → graceful degrade)
    enc = SemanticEncoder("ECG", os.path.join(_PROJECT_ROOT, "models", "semantic"))
    print(f"  window_size  : {enc.window_size}")
    print(f"  normal_range : [{enc.normal_lo}, {enc.normal_hi}]")
    print(f"  ready        : {enc.ready}")

    # Feed enough samples to fill the buffer
    import random as _rnd
    for _ in range(enc.window_size + 5):
        enc.push(_rnd.uniform(60, 100))

    print(f"  ready (after push) : {enc.ready}")

    z = enc.encode()
    print(f"  encode() -> {z}")

    if z is not None:
        result = enc.decode(z)
        print(f"  decode() -> {result}")

    score = enc.clinical_importance_learned(125.0)
    print(f"  clinical_importance_learned(125.0) -> {score:.3f}")
    print("[smoke] PASS")
