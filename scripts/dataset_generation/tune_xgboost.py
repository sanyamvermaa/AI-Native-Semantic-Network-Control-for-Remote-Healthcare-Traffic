"""
tune_xgboost.py — XGBoost Hyperparameter Tuning
─────────────────────────────────────────────────────────────────
Runs RandomizedSearchCV on XGBoost using TimeSeriesSplit CV.
Does NOT touch train_model.py or any existing models.
If a better model is found, saves as xgboost_tuned_model.pkl
"""
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

print("── XGBoost Hyperparameter Tuning ────────────────────────────────")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(DATASETS_DIR / "realistic_network_dataset.csv")
    print(f"[+] Loaded {len(df)} rows.")
except FileNotFoundError:
    print("[!] Dataset not found. Run dynamic_traffic_generator.py first.")
    raise SystemExit(1)

# ── 2. Sort by time ───────────────────────────────────────────────────────────
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 3. Feature Engineering (identical to train_model.py) ─────────────────────
print("[*] Generating features ...")
W, SLOPE_W = 10, 8
xs = np.arange(SLOPE_W)

df["rolling_loss_mean"]       = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"]        = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"]     = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"]      = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"]  = df["throughput_bps"].rolling(W).std()
df["loss_delta"]              = df["packet_loss_rate"].diff()
df["jitter_delta"]            = df["jitter"].diff()
df["delay_delta"]             = df["avg_delay"].diff()
df["loss_accel"]              = df["loss_delta"].diff()
df["loss_trend_3"]            = (df["packet_loss_rate"].rolling(3).mean()
                                 - df["packet_loss_rate"].rolling(8).mean())
df["throughput_trend_3"]      = (df["throughput_bps"].rolling(3).mean()
                                 - df["throughput_bps"].rolling(8).mean())
df["delay_trend_3"]           = (df["avg_delay"].rolling(3).mean()
                                 - df["avg_delay"].rolling(8).mean())
df["delay_slope"]             = df["avg_delay"].rolling(SLOPE_W).apply(
                                    lambda y: np.polyfit(xs, y, 1)[0], raw=True)
df["loss_slope"]              = df["packet_loss_rate"].rolling(SLOPE_W).apply(
                                    lambda y: np.polyfit(xs, y, 1)[0], raw=True)
df["loss_x_delay"]            = df["packet_loss_rate"] * df["avg_delay"]
df["loss_x_jitter"]           = df["packet_loss_rate"] * df["jitter"]

df = df.dropna().reset_index(drop=True)
print(f"    Rows after NaN drop: {len(df)}")

FEATURE_COLS = [
    "bandwidth_usage_bps", "throughput_bps", "packet_loss_rate", "jitter",
    "avg_delay", "rolling_loss_mean", "rolling_loss_std", "rolling_jitter_mean",
    "rolling_delay_mean", "rolling_throughput_mean", "rolling_throughput_std",
    "loss_delta", "jitter_delta", "delay_delta", "loss_accel",
    "loss_trend_3", "throughput_trend_3", "delay_trend_3",
    "delay_slope", "loss_slope", "active_devices", "packets_per_window",
    "loss_x_delay", "loss_x_jitter",
]

X = df[FEATURE_COLS]
y = df["network_condition"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
all_classes = sorted(y.unique())

# ── 4. Train/test split ───────────────────────────────────────────────────────
split    = int(len(df) * 0.8)
X_train, X_test   = X.iloc[:split],  X.iloc[split:]
ye_train, ye_test = y_enc[:split],   y_enc[split:]
y_test_labels     = y.iloc[split:]

# ── 5. Baseline XGBoost (your current model params) ───────────────────────────
print("\n── Baseline XGBoost (current params) ───────────────────────────")
baseline_xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1
)
baseline_xgb.fit(X_train, ye_train)
baseline_pred  = le.inverse_transform(baseline_xgb.predict(X_test))
baseline_f1    = f1_score(y_test_labels, baseline_pred, average="macro", zero_division=0)
print(classification_report(y_test_labels, baseline_pred,
                             labels=all_classes, zero_division=0))
print(f"  Baseline held-out macro F1 : {baseline_f1:.3f}")

# ── 6. Search Space ───────────────────────────────────────────────────────────
param_dist = {
    "n_estimators":      [300, 500, 700, 1000],
    "max_depth":         [4, 5, 6, 7, 8],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 3, 5, 7],
    "gamma":             [0, 0.1, 0.2, 0.5],
    "reg_alpha":         [0, 0.01, 0.1, 1.0],    # L1 regularisation
    "reg_lambda":        [1, 1.5, 2.0, 5.0],     # L2 regularisation
}

# ── 7. RandomizedSearchCV ─────────────────────────────────────────────────────
print("\n── RandomizedSearchCV (n_iter=40, 5-fold TimeSeriesSplit) ───────")
print("    This will take several minutes ...")

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=param_dist,
    n_iter=40,                  # try 40 random combinations
    scoring="f1_macro",
    cv=tscv,
    n_jobs=1,                   # parallelism handled inside XGBoost
    verbose=2,
    random_state=42,
    refit=True                  # refit best params on full X_train
)

search.fit(X_train, ye_train)

print(f"\n  Best CV macro F1 : {search.best_score_:.3f}")
print(f"  Best params:")
for k, v in search.best_params_.items():
    print(f"    {k:<22} = {v}")

# ── 8. Evaluate best model on held-out test ────────────────────────────────────
print("\n── Tuned XGBoost — Held-Out Test ────────────────────────────────")
tuned_pred   = le.inverse_transform(search.best_estimator_.predict(X_test))
tuned_f1     = f1_score(y_test_labels, tuned_pred, average="macro", zero_division=0)

print(classification_report(y_test_labels, tuned_pred,
                             labels=all_classes, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test_labels, tuned_pred, labels=all_classes))
print(f"Labels: {all_classes}")

# ── 9. Compare and save ───────────────────────────────────────────────────────
print("\n── Comparison ───────────────────────────────────────────────────")
print(f"  {'Model':<25} {'Held-out F1'}")
print(f"  {'-'*38}")
print(f"  {'XGBoost (current)':<25} {baseline_f1:.3f}")
print(f"  {'XGBoost (tuned)':<25} {tuned_f1:.3f}  "
      f"{'✅ improved' if tuned_f1 > baseline_f1 else '⚠️  no improvement'}")

if tuned_f1 > baseline_f1:
    # Retrain on full dataset with best params
    print("\n[*] Retraining on full dataset with best params ...")
    full_tuned = XGBClassifier(
        **search.best_params_,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    full_tuned.fit(X, y_enc)
    joblib.dump(full_tuned, MODELS_DIR / "xgboost_tuned_model.pkl")
    print(f"[✓] Tuned model saved → '{MODELS_DIR / 'xgboost_tuned_model.pkl'}'")
    print(f"    To use as best model, copy/rename into '{MODELS_DIR / 'best_network_model.pkl'}'")
else:
    print("\n[~] Current XGBoost params are already near-optimal for this dataset.")
    print("    No file saved — existing xgboost_network_model.pkl unchanged.")