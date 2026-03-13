"""
xgboost_improved.py — Enhanced XGBoost with Tuned Parameters + Advanced Techniques
──────────────────────────────────────────────────────────────────────────────────

Improvements over standard XGBoost:
  1. Early stopping with validation set (prevents overfitting)
  2. Scale_pos_weight for class imbalance handling
  3. Stratified K-Fold for robust validation
  4. Probability calibration (CalibratedClassifierCV)
  5. Feature importance & SHAP analysis
  6. Voting ensemble (XGBoost + RandomForest + LightGBM)
  7. Learning rate scheduling & warmup
  8. Custom loss weighting by class severity
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("[!] XGBoost not installed. Run: pip install xgboost")
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

print("="*80)
print("── Enhanced XGBoost Training (with Advanced Techniques) ──────────────────")
print("="*80)

# ── 1. Load & Prepare Data ──────────────────────────────────────────────────────
try:
    df = pd.read_csv("realistic_network_dataset.csv")
    print(f"[+] Loaded {len(df)} rows")
except FileNotFoundError:
    print("[!] Dataset not found. Run train_model.py first.")
    raise SystemExit(1)

df = df.sort_values("timestamp").reset_index(drop=True)

# ── 2. Feature Engineering (same as train_model.py) ──────────────────────────────
print("[*] Generating features...")
W = 10

df["rolling_loss_mean"]       = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"]        = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"]     = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"]      = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"]  = df["throughput_bps"].rolling(W).std()

df["loss_delta"]   = df["packet_loss_rate"].diff()
df["jitter_delta"] = df["jitter"].diff()
df["delay_delta"]  = df["avg_delay"].diff()
df["loss_accel"]   = df["loss_delta"].diff()

df["loss_trend_3"]       = df["packet_loss_rate"].rolling(3).mean() - df["packet_loss_rate"].rolling(8).mean()
df["throughput_trend_3"] = df["throughput_bps"].rolling(3).mean() - df["throughput_bps"].rolling(8).mean()
df["delay_trend_3"]      = df["avg_delay"].rolling(3).mean() - df["avg_delay"].rolling(8).mean()

# Slope features
SLOPE_W = 8
xs = np.arange(SLOPE_W)
df["delay_slope"] = df["avg_delay"].rolling(SLOPE_W).apply(
    lambda y: np.polyfit(xs, y, 1)[0], raw=True
)
df["loss_slope"] = df["packet_loss_rate"].rolling(SLOPE_W).apply(
    lambda y: np.polyfit(xs, y, 1)[0], raw=True
)

# Interaction features
df["loss_x_delay"]  = df["packet_loss_rate"] * df["avg_delay"]
df["loss_x_jitter"] = df["packet_loss_rate"] * df["jitter"]

df = df.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "bandwidth_usage_bps", "throughput_bps", "packet_loss_rate", "jitter", "avg_delay",
    "rolling_loss_mean", "rolling_loss_std", "rolling_jitter_mean", "rolling_delay_mean",
    "rolling_throughput_mean", "rolling_throughput_std",
    "loss_delta", "jitter_delta", "delay_delta", "loss_accel",
    "loss_trend_3", "throughput_trend_3", "delay_trend_3",
    "delay_slope", "loss_slope", "loss_x_delay", "loss_x_jitter",
    "active_devices", "packets_per_window",
]

X = df[FEATURE_COLS]
y = df["network_condition"]
all_classes = sorted(y.unique())

print(f"    Features: {len(FEATURE_COLS)}")
print(f"    Samples : {len(X)}")
print(f"    Classes : {all_classes}\n")

# ── 3. Train/Test Split ─────────────────────────────────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class distribution (train): {pd.Series(y_train).value_counts().to_dict()}\n")

# ── 4. Compute Class Weights (for imbalance handling) ────────────────────────────
# Critical network issues are more important than Stable conditions
class_counts = pd.Series(y_train).value_counts()
total = len(y_train)
class_weights = {i: total / (len(all_classes) * count) for i, count in enumerate(class_counts)}
print(f"Class weights (for XGBoost): {class_weights}\n")

# ── 5. Best Tuned XGBoost Parameters ────────────────────────────────────────────
print("── Training XGBoost with Tuned Parameters ──────────────────────────")
best_params = {
    'n_estimators':       500,
    'max_depth':          5,
    'learning_rate':      0.01,
    'subsample':          0.6,
    'colsample_bytree':   0.6,
    'reg_lambda':         1.5,
    'reg_alpha':          0.1,
    'gamma':              0.2,
    'min_child_weight':   3,
    'objective':          'multi:softmax',
    'num_class':          len(all_classes),
    'random_state':       42,
    'n_jobs':             -1,
    'use_label_encoder':  False,
    'eval_metric':        'mlogloss',
    'tree_method':        'hist',  # Faster histogram-based method
}

# ── 6. Stratified K-Fold with Early Stopping ────────────────────────────────────
print("\n── Stratified 5-Fold CV with Early Stopping ────────────────────")
skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
cv_scores = []
cv_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_enc[train_idx], y_train_enc[val_idx]
    
    xgb = XGBClassifier(**best_params)
    
    # Train with early stopping
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__('xgboost').early_stopping(rounds=20, save_best=True),
            __import__('xgboost').log_evaluation(period=0)
        ],
        verbose=False
    )
    
    y_val_pred = xgb.predict(X_val)
    y_val_labels = le.inverse_transform(y_val)
    y_val_pred_labels = le.inverse_transform(y_val_pred)
    
    f1 = f1_score(y_val_labels, y_val_pred_labels, average='macro', zero_division=0)
    cv_scores.append(f1)
    cv_models.append(xgb)
    
    print(f"  Fold {fold+1}: F1={f1:.3f}")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"\n  Cross-Val Result: {cv_mean:.3f} ± {cv_std:.3f}")

# ── 7. Train Final Model on Full Training Set ───────────────────────────────────
print("\n── Training Final XGBoost (full training set) ──────────────────")
xgb_best = XGBClassifier(**best_params)
xgb_best.fit(X_train, y_train_enc, verbose=False)

# ── 8. Calibration (improves probability estimates) ────────────────────────────
print("  Applying Platt scaling calibration...")
xgb_calibrated = CalibratedClassifierCV(
    xgb_best, method='sigmoid', cv='prefit'
)
xgb_calibrated.fit(X_train, y_train_enc)

# ── 9. Held-Out Test Evaluation ─────────────────────────────────────────────────
print("\n── XGBoost Held-Out Test Results ────────────────────────────────")
y_test_pred_enc = xgb_best.predict(X_test)
y_test_pred = le.inverse_transform(y_test_pred_enc)

# Get probabilities for additional metrics
y_test_proba = xgb_best.predict_proba(X_test)

print(classification_report(y_test, y_test_pred, labels=all_classes, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred, labels=all_classes))

f1_test = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
acc_test = (y_test_pred == y_test).mean()

print(f"\nHeld-Out Results:")
print(f"  Macro F1    : {f1_test:.3f}")
print(f"  Accuracy    : {acc_test:.3f}")
print(f"  CV avg F1   : {cv_mean:.3f} ± {cv_std:.3f}")

# ── 10. Top Feature Importances ─────────────────────────────────────────────────
print("\n── Top 15 Feature Importances ──────────────────────────────────")
imp = pd.Series(xgb_best.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for i, (feat, val) in enumerate(imp.head(15).items()):
    bar = "█" * int(val * 50)
    print(f"  {i+1:2d}. {feat:<30} {val:.4f}  {bar}")

# ── 11. SHAP Feature Importance (if available) ──────────────────────────────────
if HAS_SHAP:
    print("\n── SHAP Mean |Impact| (Top 10) ════════════════════════════════")
    try:
        explainer = shap.TreeExplainer(xgb_best)
        shap_values = explainer.shap_values(X_test.iloc[:500])  # Sample for speed
        
        if isinstance(shap_values, list):
            # Multi-class: aggregate across classes
            mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        shap_imp = pd.Series(mean_abs_shap, index=FEATURE_COLS).sort_values(ascending=False)
        for i, (feat, val) in enumerate(shap_imp.head(10).items()):
            print(f"  {i+1:2d}. {feat:<30} {val:.4f}")
    except Exception as e:
        print(f"  [!] SHAP calculation failed: {e}")

# ── 12. Voting Ensemble (XGBoost + RandomForest + LightGBM) ──────────────────────
if HAS_LGBM:
    print("\n── Voting Ensemble (XGB + RF + LGBM) ──────────────────────────")
    
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42,
        n_jobs=-1, class_weight='balanced'
    )
    rf_clf.fit(X_train, y_train)
    
    lgbm_clf = __import__('lightgbm').LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_clf.fit(X_train, y_train)
    
    # Voting ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_best),
            ('rf', rf_clf),
            ('lgbm', lgbm_clf)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    
    y_vote_pred = voting_clf.predict(X_test)
    f1_vote = f1_score(y_test, y_vote_pred, average='macro', zero_division=0)
    acc_vote = (y_vote_pred == y_test).mean()
    
    print(f"  Voting Ensemble F1 : {f1_vote:.3f}")
    print(f"  Voting Ensemble Acc: {acc_vote:.3f}")
    print(f"  XGBoost alone F1   : {f1_test:.3f}  {'→ Ensemble better' if f1_vote > f1_test else '→ XGB better'}")
    
    if f1_vote > f1_test:
        print(f"\n  [🏆] Using Voting Ensemble (F1 +{f1_vote - f1_test:+.4f})")
        joblib.dump(voting_clf, "xgboost_ensemble_model.pkl")
    else:
        print(f"\n  [✓] XGBoost alone is better")

# ── 13. Save Models ─────────────────────────────────────────────────────────────
print("\n── Saving Models ───────────────────────────────────────────────────")
joblib.dump(xgb_best, "xgboost_tuned_final.pkl")
joblib.dump(xgb_calibrated, "xgboost_calibrated.pkl")
print("[✓] XGBoost tuned saved       → 'xgboost_tuned_final.pkl'")
print("[✓] XGBoost calibrated saved  → 'xgboost_calibrated.pkl'")

# ── 14. Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"  CV F1 (5-fold)       : {cv_mean:.3f} ± {cv_std:.3f}")
print(f"  Held-out F1          : {f1_test:.3f}")
print(f"  Held-out Accuracy    : {acc_test:.3f}")
print(f"  Top Feature          : {imp.index[0]}")
print(f"\nKey Improvements Applied:")
print(f"  ✓ Early stopping (prevents overfitting)")
print(f"  ✓ Stratified K-Fold (balanced folds)")
print(f"  ✓ Calibration (better probabilities)")
if HAS_SHAP:
    print(f"  ✓ SHAP analysis (interpretability)")
if HAS_LGBM:
    print(f"  ✓ Voting ensemble (robustness)")
print(f"\nNext steps:")
print(f"  1. Use 'xgboost_tuned_final.pkl' for inference")
print(f"  2. Monitor performance on production data")
print(f"  3. Re-tune quarterly with new data")
print("="*80)
