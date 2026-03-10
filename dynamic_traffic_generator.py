import subprocess
import time
import random
import csv
import os
import pandas as pd
import signal

# Configuration
DURATION_SECONDS = 300  # 5 Minutes
BASE_DIR = "/home/ayhm23/health_data/csv"
os.makedirs(BASE_DIR, exist_ok=True)
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth_log.csv")

# Profiles: (min, max)
PROFILES = {
    "Stable":   {"loss": (0, 0.05),   "delay": (5, 15),     "jitter": (0, 3)},
    "Unstable": {"loss": (0.5, 3.0),  "delay": (20, 80),    "jitter": (5, 20)},
    "Critical": {"loss": (5.0, 15.0), "delay": (100, 300),  "jitter": (30, 80)}
}

def run_command(cmd):
    """Executes a shell command."""
    # print(f"DEBUG Executing: {cmd}")  # Uncomment for debugging
    subprocess.run(cmd, shell=True, check=True)

def apply_network_condition(loss, delay, jitter):
    """Applies traffic control rules with safety checks for 'tc' syntax."""
    
    # FIX: 'distribution normal' fails if jitter is 0. 
    # logic: Only use distribution if jitter is significant (> 1ms)
    if jitter < 1.0:
        cmd = f"sudo ip netns exec sender_ns tc qdisc replace dev veth_s root netem loss {loss:.3f}% delay {delay:.2f}ms"
    else:
        cmd = f"sudo ip netns exec sender_ns tc qdisc replace dev veth_s root netem loss {loss:.3f}% delay {delay:.2f}ms {jitter:.2f}ms distribution normal"

    try:
        run_command(cmd)
    except subprocess.CalledProcessError:
        # If 'replace' fails (first run), try 'add'
        cmd = cmd.replace("replace", "add")
        run_command(cmd)

def generate_random_params(profile_name):
    ranges = PROFILES[profile_name]
    loss = random.uniform(*ranges["loss"])
    delay = random.uniform(*ranges["delay"])
    jitter = random.uniform(*ranges["jitter"])
    return loss, delay, jitter

def merge_data():
    telemetry_path = os.path.join(BASE_DIR, "network_telemetry.csv")
    
    if not os.path.exists(telemetry_path):
        print("Error: No telemetry found. Did the receiver run?")
        return

    print("Loading data...")
    telem_df = pd.read_csv(telemetry_path)
    if os.path.exists(GROUND_TRUTH_FILE):
        truth_df = pd.read_csv(GROUND_TRUTH_FILE)
    else:
        print("Error: No ground truth log found.")
        return
    
    # Sort
    telem_df = telem_df.sort_values("timestamp")
    truth_df = truth_df.sort_values("start_time")

    def get_label_for_ts(ts):
        # Find matching time window
        match = truth_df[(truth_df["start_time"] <= ts) & (truth_df["end_time"] >= ts)]
        if not match.empty:
            return match.iloc[0]["label"]
        return "Transition"

    print("Merging labels...")
    telem_df["network_condition"] = telem_df["timestamp"].apply(get_label_for_ts)
    telem_df = telem_df[telem_df["network_condition"] != "Transition"]
    
    output_file = "realistic_network_dataset.csv"
    telem_df.to_csv(output_file, index=False)
    
    print(f"\nSUCCESS! Generated {len(telem_df)} rows.")
    print(f"Saved to: {output_file}")
    print(telem_df["network_condition"].value_counts())

def main():
    print("--- Starting Dynamic Network Simulation (Chaos Mode) ---")
    
    print("[*] Setting up network namespaces...")
    # Clean run ensure
    subprocess.run(["sudo", "./setup_namespaces.sh"], check=True)
    
    print("[*] Starting Health Receiver and Sender...")
    # Using absolute paths to ensure sudo finds the correct python
    PYTHON_EXEC = "/home/ayhm23/miniconda3/bin/python3"
    
    rx_cmd = ["sudo", "ip", "netns", "exec", "receiver_ns", PYTHON_EXEC, "health_receiver.py"]
    tx_cmd = ["sudo", "ip", "netns", "exec", "sender_ns",   PYTHON_EXEC, "health_sender.py"]
    
    rx_proc = subprocess.Popen(rx_cmd)
    time.sleep(2) 
    tx_proc = subprocess.Popen(tx_cmd)
    
    start_time = time.time()
    print(f"[*] Running simulation for {DURATION_SECONDS} seconds...")
    
    try:
        with open(GROUND_TRUTH_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "label", "applied_loss", "applied_delay", "applied_jitter"])
            
            while time.time() - start_time < DURATION_SECONDS:
                # Weighted random choice
                state = random.choices(["Stable", "Unstable", "Critical"], weights=[0.4, 0.35, 0.25])[0]
                
                loss, delay, jitter = generate_random_params(state)
                
                print(f" -> State: {state:8} | Loss: {loss:.2f}% | Delay: {delay:.0f}ms | Jitter: {jitter:.0f}ms")
                apply_network_condition(loss, delay, jitter)
                
                # Random duration for this state
                step_start = time.time()
                duration = random.uniform(5, 10)
                time.sleep(duration)
                step_end = time.time()
                
                writer.writerow([step_start, step_end, state, loss, delay, jitter])
                f.flush()
                
    except KeyboardInterrupt:
        print("\n[!] User interrupted simulation.")
        
    finally:
        print("\n[*] Stopping processes...")
        tx_proc.terminate()
        rx_proc.terminate()
        # Reset network to clean state
        try:
            run_command("sudo ip netns exec sender_ns tc qdisc del dev veth_s root")
        except:
            pass
        
        # Give file handles a moment to close
        time.sleep(1)
        merge_data()

if __name__ == "__main__":
    main()