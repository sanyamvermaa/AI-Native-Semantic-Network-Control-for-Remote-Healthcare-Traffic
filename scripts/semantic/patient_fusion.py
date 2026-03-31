#!/usr/bin/env python3
"""
Runtime patient-level fusion inference wrapper.

Loads the TorchScript PatientFusionEncoder saved by train_patient_fusion.py
and maintains a rolling per-device latent-vector store. Calling infer()
produces a joint patient state and deterioration probability from all
recently observed device representations.

Usage example:
    pfe = PatientFusionEncoder("/path/to/models/semantic")
    # In the receive loop, after SemanticEncoder.encode():
    pfe.update(device_id=0, device_type="ECG", z=z, timestamp=time.time())
    result = pfe.infer()
    # result["joint_state"]        → "NORMAL" / "ALERT" / "CRITICAL"
    # result["deterioration_prob"] → 0.0–1.0
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (must match train_patient_fusion.py)
# ---------------------------------------------------------------------------
LABEL_CLASSES: List[str] = ["NORMAL", "ALERT", "CRITICAL"]
LATENT_DIM                = 16
N_DEVICE_SLOTS            = 8
DEVICE_TYPE_INDEX: Dict[str, int] = {
    "ECG":           0,
    "SpO2":          1,
    "BloodPressure": 2,
    "Temperature":   3,
    "Respiration":   4,
}
DEVICE_ENTRY_TTL = 2.0   # seconds before a stale entry is evicted

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
# PatientFusionEncoder
# ---------------------------------------------------------------------------

class PatientFusionEncoder:
    """
    Cross-modal patient-level fusion encoder wrapper.

    Parameters
    ----------
    models_dir : str
        Directory containing patient_fusion.pt (written by
        train_patient_fusion.py).
    """

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self.available  = False
        self._model     = None

        # {device_id: {"z": List[float], "device_type": str, "ts": float}}
        self._store: Dict[int, Dict] = {}

        if _TORCH_AVAILABLE:
            self._load_model(models_dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, models_dir: str) -> None:
        """Load the TorchScript fusion model."""
        import torch  # type: ignore

        model_path = os.path.join(models_dir, "patient_fusion.pt")
        if not os.path.isfile(model_path):
            print(
                f"[PatientFusionEncoder] WARN: model not found at {model_path}. "
                "Run train_patient_fusion.py first."
            )
            return

        try:
            self._model   = torch.jit.load(model_path, map_location="cpu")
            self._model.eval()
            self.available = True
        except Exception as exc:
            print(f"[PatientFusionEncoder] WARN: could not load model: {exc}")

    def _evict_stale(self, now: float) -> None:
        """Remove entries older than DEVICE_ENTRY_TTL seconds."""
        stale = [k for k, v in self._store.items() if now - v["ts"] > DEVICE_ENTRY_TTL]
        for k in stale:
            del self._store[k]

    def _build_input(self) -> Optional["torch.Tensor"]:
        """
        Assemble the [1, N_DEVICE_SLOTS, 17] input tensor from the store.
        Returns None if torch is unavailable.
        """
        import torch  # type: ignore
        import numpy as np  # type: ignore (numpy always present in WSL env)

        matrix = np.zeros((N_DEVICE_SLOTS, LATENT_DIM + 1), dtype=np.float32)

        for dev_id, entry in self._store.items():
            slot     = min(int(dev_id), N_DEVICE_SLOTS - 1)
            z        = entry["z"]
            dev_type = entry["device_type"]
            type_idx = DEVICE_TYPE_INDEX.get(dev_type, 0)

            z_arr = [float(v) for v in z]
            # Pad or truncate z to exactly LATENT_DIM
            z_padded = (z_arr + [0.0] * LATENT_DIM)[:LATENT_DIM]

            matrix[slot, :LATENT_DIM] = z_padded
            matrix[slot, LATENT_DIM]  = float(type_idx)

        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # [1, 8, 17]
        return tensor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        device_id:   int,
        device_type: str,
        z:           List[float],
        timestamp:   float,
    ) -> None:
        """
        Store the latest latent vector for a device, replacing any prior entry.

        Parameters
        ----------
        device_id   : int
        device_type : str
        z           : List[float]   — latent vector from SemanticEncoder.encode()
        timestamp   : float         — epoch timestamp; used for TTL eviction
        """
        self._evict_stale(timestamp)
        self._store[device_id] = {
            "z":           list(z),
            "device_type": device_type,
            "ts":          timestamp,
        }

    def infer(self) -> Dict:
        """
        Run the fusion model on the current device store.

        Returns
        -------
        Dict with keys:
          joint_state        : str    — "NORMAL" / "ALERT" / "CRITICAL" / "UNKNOWN"
          deterioration_prob : Optional[float] — 0.0–1.0, or None
          active_devices     : int
          confidence         : float
        """
        now = time.time()
        self._evict_stale(now)

        active = len(self._store)

        if active < 2:
            return {
                "joint_state":        "UNKNOWN",
                "deterioration_prob": None,
                "active_devices":     active,
                "confidence":         0.0,
            }

        if not self.available or self._model is None or not _TORCH_AVAILABLE:
            return {
                "joint_state":        "UNKNOWN",
                "deterioration_prob": None,
                "active_devices":     active,
                "confidence":         0.0,
            }

        import torch  # type: ignore

        tensor = self._build_input()
        if tensor is None:
            return {
                "joint_state":        "UNKNOWN",
                "deterioration_prob": None,
                "active_devices":     active,
                "confidence":         0.0,
            }

        with torch.no_grad():
            logits, det_logit = self._model(tensor)           # [1,3], [1,1]
            probs             = torch.softmax(logits, dim=-1) # [1,3]
            det_prob          = torch.sigmoid(det_logit)      # [1,1]

        probs_list = probs.squeeze(0).tolist()
        best_idx   = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))
        confidence = float(probs_list[best_idx])
        joint_state = (
            LABEL_CLASSES[best_idx]
            if best_idx < len(LABEL_CLASSES)
            else "NORMAL"
        )

        return {
            "joint_state":        joint_state,
            "deterioration_prob": float(det_prob.squeeze()),
            "active_devices":     active,
            "confidence":         confidence,
        }


# ---------------------------------------------------------------------------
# __main__ — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _HERE         = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
    _models_dir   = os.path.join(_PROJECT_ROOT, "models", "semantic")

    print("[smoke] PatientFusionEncoder — torch_available:", _TORCH_AVAILABLE)

    pfe = PatientFusionEncoder(_models_dir)
    print(f"  available: {pfe.available}")

    # Push fake latent vectors for 5 devices
    import random as _rnd
    for dev_id in range(5):
        z = [_rnd.uniform(-1, 1) for _ in range(16)]
        dtype = ["ECG", "SpO2", "BloodPressure", "Temperature", "Respiration"][dev_id]
        pfe.update(dev_id, dtype, z, time.time())

    result = pfe.infer()
    print(f"  infer() -> {result}")

    # Test TTL eviction path
    pfe._store[99] = {"z": [0.0]*16, "device_type": "ECG", "ts": time.time() - 10.0}
    pfe._evict_stale(time.time())
    assert 99 not in pfe._store, "FAIL: stale entry not evicted"

    # Test <2 devices path
    pfe2 = PatientFusionEncoder(_models_dir)
    pfe2.update(0, "ECG", [0.0]*16, time.time())
    r2 = pfe2.infer()
    assert r2["joint_state"] == "UNKNOWN", f"FAIL: expected UNKNOWN got {r2['joint_state']}"

    print("[smoke] PASS")
