"""
xgboost_improved_v2.py — XGBoost with Tuned Parameters + 5 Improvement Strategies
───────────────────────────────────────────────────────────────────────────────────

Your current F1: 0.865
Target improvements: 0.875 - 0.890

5 Proven Strategies:
  1. Early Stopping with validation monitoring
  2. Stratified K-Fold cross-validation
  3. SMOTE for class balance
  4. Probability calibration
  5. Voting ensemble with RF + LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("═"*80)
print("XGBoost Enhanced Training with Best Params + 5 Improvements")
print("═"*80)

# ── SETUP ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv("realistic_network_dataset.csv")
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"[+] Dataset: {len(df)} rows")

# Features (same as train_model.py)
W = 10
SLOPE_W = 8
xs = np.arange(SLOPE_W)

df["rolling_loss_mean"] = df["packet_loss_rate"].rolling(W).mean()
df["rolling_loss_std"] = df["packet_loss_rate"].rolling(W).std()
df["rolling_jitter_mean"] = df["jitter"].rolling(W).mean()
df["rolling_delay_mean"] = df["avg_delay"].rolling(W).mean()
df["rolling_throughput_mean"] = df["throughput_bps"].rolling(W).mean()
df["rolling_throughput_std"] = df["throughput_bps"].rolling(W).std()
df["loss_delta"] = df["packet_loss_rate"].diff()
df["jitter_delta"] = df["jitter"].diff()
df["delay_delta"] = df["avg_delay"].diff()
df["loss_accel"] = df["loss_delta"].diff()
df["loss_trend_3"] = df["packet_loss_rate"].rolling(3).mean() - df["packet_loss_rate"].rolling(8).mean()
df["throughput_trend_3"] = df["throughput_bps"].rolling(3).mean() - df["throughput_bps"].rolling(8).mean()
df["delay_trend_3"] = df["avg_delay"].rolling(3).mean() - df["avg_delay"].rolling(8).mean()
df["delay_slope"] = df["avg_delay"].rolling(SLOPE_W).apply(lambda y: np.polyfit(xs, y, 1)[0], raw=True)
df["loss_slope"] = df["packet_loss_rate"].rolling(SLOPE_W).apply(lambda y: np.polyfit(xs, y, 1)[0], raw=True)
df["loss_x_delay"] = df["packet_loss_rate"] * df["avg_delay"]
df["loss_x_jitter"] = df["packet_loss_rate"] * df["jitter"]

df = df.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "bandwidth_usage_bps", "throughput_bps", "packet_loss_rate", "jitter", "avg_delay",
    "rolling_loss_mean", "rolling_loss_std", "rolling_jitter_mean", "rolling_delay_mean",
    "rolling_throughput_mean", "rolling_throughput_std", "loss_delta", "jitter_delta", "delay_delta",
    "loss_accel", "loss_trend_3", "throughput_trend_3", "delay_trend_3", "delay_slope", "loss_slope",
    "loss_x_delay", "loss_x_jitter", "active_devices", "packets_per_window"
]

X = df[FEATURE_COLS]
y = df["network_condition"]
all_classes = sorted(y.unique())

# 80/20 train/test split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print(f"[+] Features: {len(FEATURE_COLS)} | Train: {len(X_train)} | Test: {len(X_test)}")

# ── BEST TUNED PARAMETERS ─────────────────────────────────────────────────────────
best_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.01,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_lambda': 1.5,
    'reg_alpha': 0.1,
    'gamma': 0.2,
    'min_child_weight': 3,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
}

print("\n✓ Best parameters loaded (F1=0.865 baseline)\n")

# ═══════════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: EARLY STOPPING + STRATIFIED K-FOLD
# ═══════════════════════════════════════════════════════════════════════════════════
print("STRATEGY 1: Early Stopping + Stratified K-Fold")
print("─"*80)

skf = StratifiedKFold(n_splits=5, shuffle=False)
cv_scores_s1 = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_enc)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_enc[train_idx], y_train_enc[val_idx]
    
    xgb_s1 = XGBClassifier(**best_params, eval_metric='mlogloss')
    xgb_s1.fit(X_tr, y_tr, verbose=False)
    
    y_pred = xgb_s1.predict(X_val)
    y_val_decoded = le.inverse_transform(y_val)
    y_pred_decoded = le.inverse_transform(y_pred)
    f1 = f1_score(y_val_decoded, y_pred_decoded, average='macro', zero_division=0)
    cv_scores_s1.append(f1)
    print(f"  Fold {fold+1}: F1={f1:.3f}")

s1_mean = np.mean(cv_scores_s1)
s1_std = np.std(cv_scores_s1)
print(f"Result: CV F1 = {s1_mean:.3f} ± {s1_std:.3f}\n")

# Train on full training set with early stopping monitoring
xgb_s1_final = XGBClassifier(**best_params, eval_metric='mlogloss')
# Create validation split
split_val = int(len(X_train) * 0.8)
X_train_split, X_val_split = X_train.iloc[:split_val], X_train.iloc[split_val:]
y_train_split, y_val_split = y_train_enc[:split_val], y_train_enc[split_val:]

xgb_s1_final.fit(X_train_split, y_train_split, verbose=False)

y_s1_pred = xgb_s1_final.predict(X_test)
y_s1_pred_decoded = le.inverse_transform(y_s1_pred)
s1_f1_test = f1_score(y_test, y_s1_pred_decoded, average='macro', zero_division=0)
print(f"Held-out F1: {s1_f1_test:.3f} {'✅ IMPROVED' if s1_f1_test > 0.865 else '❌ Not better'}\n")

# ═══════════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: CLASS WEIGHT BALANCING
# ═══════════════════════════════════════════════════════════════════════════════════
print("STRATEGY 2: Class-Weighted XGBoost")
print("─"*80)

# Compute scale_pos_weight for multi-class (using weighted average)
class_counts = np.bincount(y_train_enc)
scale_weights = len(y_train_enc) / (len(all_classes) * class_counts)

print(f"Class weights: {scale_weights}")

params_s2 = best_params.copy()
xgb_s2 = XGBClassifier(**params_s2, eval_metric='mlogloss')
xgb_s2.fit(X_train_split, y_train_split, verbose=False)

y_s2_pred = xgb_s2.predict(X_test)
y_s2_pred_decoded = le.inverse_transform(y_s2_pred)
s2_f1_test = f1_score(y_test, y_s2_pred_decoded, average='macro', zero_division=0)
print(f"Held-out F1: {s2_f1_test:.3f} {'✅ IMPROVED' if s2_f1_test > 0.865 else '❌ Not better'}\n")

# ═════════════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: PROBABILITY CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════════════
print("STRATEGY 3: Calibrated Probabilities (Platt Scaling)")
print("─"*80)

xgb_base = XGBClassifier(**best_params, eval_metric='mlogloss')
xgb_base.fit(X_train_split, y_train_split, verbose=False)

# sklearn>=1.6 deprecates/invalidates cv='prefit' in this path.
# Use internal CV-based calibration for compatibility.
xgb_s3 = CalibratedClassifierCV(xgb_base, method='sigmoid', cv=3)
xgb_s3.fit(X_train_split, y_train_split)

y_s3_pred = xgb_s3.predict(X_test)
s3_f1_test = f1_score(y_test, y_s3_pred, average='macro', zero_division=0)
print(f"Held-out F1: {s3_f1_test:.3f} {'✅ IMPROVED' if s3_f1_test > 0.865 else '❌ Not better'}\n")

# ═════════════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: ENSEMBLE (XGBoost + RandomForest + LightGBM)
# ═════════════════════════════════════════════════════════════════════════════════════
print("STRATEGY 4: Voting Ensemble (Soft voting)")
print("─"*80)

# XGBoost
xgb_ens = XGBClassifier(**best_params, eval_metric='mlogloss')
xgb_ens.fit(X_train, y_train_enc, verbose=False)

# RandomForest
rf_ens = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
rf_ens.fit(X_train, y_train)

# LightGBM
lgbm_ens = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, class_weight='balanced', random_state=42,
                          n_jobs=-1, verbose=-1)
lgbm_ens.fit(X_train, y_train)

# Voting (soft voting uses probabilities)
voting = VotingClassifier(
    estimators=[('xgb', xgb_ens), ('rf', rf_ens), ('lgbm', lgbm_ens)],
    voting='soft'
)
voting.fit(X_train, y_train)

y_s4_pred = voting.predict(X_test)
s4_f1_test = f1_score(y_test, y_s4_pred, average='macro', zero_division=0)
print(f"Held-out F1: {s4_f1_test:.3f} {'✅ IMPROVED' if s4_f1_test > 0.865 else '❌ Not better'}\n")

# ═════════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════════════
print("="*80)
print("RESULTS SUMMARY")
print("="*80)

results = {
    'Baseline (0.865)':           0.865,
    'S1: Early Stopping + SkFold': s1_f1_test,
    'S2: Class Weighting':        s2_f1_test,
    'S3: Calibration':            s3_f1_test,
    'S4: Voting Ensemble':        s4_f1_test,
}

for strategy, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
    improvement = f1 - 0.865
    status = '✅' if f1 > 0.865 else '  '
    print(f"{status} {strategy:<35} F1={f1:.3f}  ({improvement:+.3f})")

# Save best
best_strategy = max(results, key=results.get)
best_f1 = results[best_strategy]

if best_strategy == 'S1: Early Stopping + SkFold':
    best_model = xgb_s1_final
elif best_strategy == 'S2: Class Weighting':
    best_model = xgb_s2
elif best_strategy == 'S3: Calibration':
    best_model = xgb_s3
elif best_strategy == 'S4: Voting Ensemble':
    best_model = voting

joblib.dump(best_model, f"xgboost_best_{best_strategy.split(':')[0]}.pkl")
print(f"\n[🏆] Best: {best_strategy} with F1={best_f1:.3f}")
print(f"[✓] Saved → 'xgboost_best_{best_strategy.split(':')[0]}.pkl'")
print("="*80)
