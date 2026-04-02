"""Quick smoke test: can the receiver's model load path work under sudo python3?"""
import importlib, pathlib, sys

try:
    jl = importlib.import_module("joblib")
    print("[TEST] joblib imported OK:", jl.__version__)
except Exception as e:
    print("[FAIL] joblib import failed:", e); sys.exit(1)

repo_root = pathlib.Path(__file__).resolve().parents[2]
model_path = repo_root / "models" / "xgboost_network_model.pkl"
print("[TEST] Looking for model at:", model_path)
if not model_path.exists():
    print("[FAIL] Model file not found"); sys.exit(1)

try:
    model = jl.load(model_path)
    print("[TEST] Model loaded:", type(model).__name__)
    feat = list(getattr(model, "feature_names_in_", []))
    print("[TEST] Feature count:", len(feat))
    dummy = [[0.0] * max(len(feat), 24)]
    pred = model.predict(dummy)
    print("[TEST] Predict OK -> class:", pred[0])
except Exception as e:
    print("[FAIL] Model load/predict failed:", e); sys.exit(1)

print("[TEST] PASS — receiver will use XGBoost live")
