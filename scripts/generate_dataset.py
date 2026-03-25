import subprocess
import pandas as pd
import time
import os
from pathlib import Path

# Define scenarios: (Loss %, Delay ms, Duration s, Network Label)
scenarios = [
    (0, 0, 60, "Stable"),      # Good Network
    (5, 50, 60, "Unstable"),   # Moderate Issues
    (20, 200, 60, "Critical")  # Heavy Congestion
]

telemetry_files = []
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
LOGS_DIR = PROJECT_ROOT / "data" / "logs"

BASE_DIR = str(LOGS_DIR)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

print("Starting Data Generation Pipeline...")

# 1. Setup Namespaces (Run once)
subprocess.run(["sudo", str(SCRIPTS_DIR / "setup_namespaces.sh")], check=True)

for loss, delay, duration, label in scenarios:
    print(f"\n--- Running Scenario: {label} (Loss={loss}%, Delay={delay}ms) ---")
    
    # Run the experiment script
    subprocess.run(["sudo", str(SCRIPTS_DIR / "run_experiment.sh"), str(loss), str(delay), str(duration)], check=True)
    
    # Load the generated telemetry
    telemetry_path = os.path.join(BASE_DIR, "network_telemetry.csv")
    if os.path.exists(telemetry_path):
        df = pd.read_csv(telemetry_path)
        df['network_condition'] = label  # Add the ground truth label
        
        # Save to a temporary file
        temp_filename = DATASETS_DIR / f"telemetry_{label}.csv"
        df.to_csv(temp_filename, index=False)
        telemetry_files.append(str(temp_filename))
        
        print(f"Captured {len(df)} rows for {label}")
    else:
        print(f"Error: No data found for {label}")

# 2. Merge all datasets
if telemetry_files:
    print("\nMerging datasets...")
    full_df = pd.concat([pd.read_csv(f) for f in telemetry_files])
    
    # Save final dataset
    final_path = DATASETS_DIR / "final_network_dataset.csv"
    full_df.to_csv(final_path, index=False)
    print(f"Success! Dataset saved to '{final_path}'")
    print(full_df['network_condition'].value_counts())
else:
    print("No data was generated.")