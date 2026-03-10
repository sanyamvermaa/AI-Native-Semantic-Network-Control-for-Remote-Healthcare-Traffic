import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("--- Starting Model Training Pipeline ---")

# 1. Load Data
# Ensure 'realistic_network_dataset.csv' is in the same folder
try:
    df = pd.read_csv('realistic_network_dataset.csv')
    print(f"[+] Loaded dataset with {len(df)} rows.")
except FileNotFoundError:
    print("[!] Error: 'realistic_network_dataset.csv' not found.")
    exit(1)

# 2. Feature Engineering (Must match receiver logic!)
# Rolling windows help the model see trends (e.g., Is packet loss increasing?)
print("[*] Generating rolling window features...")
window_size = 5
df['rolling_loss_mean'] = df['packet_loss_rate'].rolling(window=window_size).mean()
df['rolling_jitter_mean'] = df['jitter'].rolling(window=window_size).mean()
df['rolling_throughput_mean'] = df['throughput_bps'].rolling(window=window_size).mean()

# Drop initial rows with NaN values caused by rolling window
df = df.dropna()

# 3. Define Features and Target
feature_cols = [
    'bandwidth_usage_bps', 
    'throughput_bps', 
    'packet_loss_rate', 
    'jitter', 
    'queue_length', 
    'rolling_loss_mean', 
    'rolling_jitter_mean', 
    'rolling_throughput_mean'
]

X = df[feature_cols]
y = df['network_condition']

# 4. Split Data (80% Train, 20% Test)
# Stratify ensures balanced classes in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[+] Training Set: {len(X_train)} samples")
print(f"[+] Test Set:     {len(X_test)} samples")

# 5. Train Model
print("[*] Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
print("\n--- Evaluation Results (Test Set) ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save Weights
model_filename = 'robust_network_model.pkl'
joblib.dump(model, model_filename)
print(f"\n[SUCCESS] Model saved to '{model_filename}'")