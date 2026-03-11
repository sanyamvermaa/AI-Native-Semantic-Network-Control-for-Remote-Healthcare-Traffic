import subprocess
import pandas as pd
import time
import os

# Define scenarios: (Loss %, Delay ms, Duration s, Network Label)
scenarios = [
    (0, 0, 60, "Stable"),      # Good Network
    (5, 50, 60, "Unstable"),   # Moderate Issues
    (20, 200, 60, "Critical")  # Heavy Congestion
]

telemetry_files = []
BASE_DIR = "/home/ayhm23/health_data/csv"

print("Starting Data Generation Pipeline...")

# 1. Setup Namespaces (Run once)
subprocess.run(["sudo", "./setup_namespaces.sh"])

for loss, delay, duration, label in scenarios:
    print(f"\n--- Running Scenario: {label} (Loss={loss}%, Delay={delay}ms) ---")
    
    # Run the experiment script
    subprocess.run(["sudo", "./run_experiment.sh", str(loss), str(delay), str(duration)])
    
    # Load the generated telemetry
    telemetry_path = os.path.join(BASE_DIR, "network_telemetry.csv")
    if os.path.exists(telemetry_path):
        df = pd.read_csv(telemetry_path)
        df['network_condition'] = label  # Add the ground truth label
        
        # Save to a temporary file
        temp_filename = f"telemetry_{label}.csv"
        df.to_csv(temp_filename, index=False)
        telemetry_files.append(temp_filename)
        
        print(f"Captured {len(df)} rows for {label}")
    else:
        print(f"Error: No data found for {label}")

# 2. Merge all datasets
if telemetry_files:
    print("\nMerging datasets...")
    full_df = pd.concat([pd.read_csv(f) for f in telemetry_files])
    
    # Save final dataset
    full_df.to_csv("final_network_dataset.csv", index=False)
    print("Success! Dataset saved to 'final_network_dataset.csv'")
    print(full_df['network_condition'].value_counts())
else:
    print("No data was generated.")