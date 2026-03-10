"""
dynamic_traffic_generator.py
─────────────────────────────
Simulates a hospital ward network:
  - N patient monitoring devices (senders) in sender_ns
  - 1 central gateway receiver in receiver_ns
  - tc netem injects real loss / delay / jitter on the shared link
  - Profiles cycle randomly with realistic weighted transitions
  - Ground-truth labels are written for later merging with telemetry

Balancing strategy:
  Contiguous block sampling — preserves local time continuity within each
  sampled chunk so rolling features remain meaningful after balancing.
  Random row undersampling is intentionally avoided (breaks time series).
"""

import subprocess
import time
import random
import csv
import os
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
DURATION_SECONDS  = 1800
BASE_DIR          = "/home/ayhm23/health_data/csv"
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth_log.csv")
os.makedirs(BASE_DIR, exist_ok=True)

PYTHON_EXEC = "/home/ayhm23/miniconda3/bin/python3"

# ── Device Fleet ──────────────────────────────────────────────────────────────
DEVICES = [
    {"id": 0, "type": "ECG"},
    {"id": 1, "type": "ECG"},
    {"id": 2, "type": "SpO2"},
    {"id": 3, "type": "SpO2"},
    {"id": 4, "type": "BloodPressure"},
    {"id": 5, "type": "BloodPressure"},
    {"id": 6, "type": "Temperature"},
    {"id": 7, "type": "Respiration"},
]

# ── Network Profiles ──────────────────────────────────────────────────────────
PROFILES = {
    "Stable": {
        "loss":     (0.0,  0.8),
        "delay":    (5,    25),
        "jitter":   (0,    4),
        "duration": (18,   24),
    },
    "Unstable": {
        "loss":     (3.0,  7.0),
        "delay":    (40,   120),
        "jitter":   (8,    25),
        "duration": (22,   30),
    },
    "Critical": {
        "loss":     (8.0,  22.0),
        "delay":    (130,  380),
        "jitter":   (35,   90),
        "duration": (26,   34),
    },
}

STATE_WEIGHTS = {"Stable": 0.33, "Unstable": 0.34, "Critical": 0.33}

# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, **kwargs):
    subprocess.run(cmd, shell=True, check=True, **kwargs)

def apply_netem(loss, delay, jitter):
    if jitter < 1.0:
        cmd = (f"sudo ip netns exec sender_ns tc qdisc replace dev veth_s "
               f"root netem loss {loss:.3f}% delay {delay:.1f}ms")
    else:
        cmd = (f"sudo ip netns exec sender_ns tc qdisc replace dev veth_s "
               f"root netem loss {loss:.3f}% delay {delay:.1f}ms "
               f"{jitter:.1f}ms distribution normal")
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        run(cmd.replace("replace", "add"))

def sample_profile(name):
    p = PROFILES[name]
    return (
        random.uniform(*p["loss"]),
        random.uniform(*p["delay"]),
        random.uniform(*p["jitter"]),
        random.uniform(*p["duration"]),
    )


# ── Merging ───────────────────────────────────────────────────────────────────
def merge_data():
    telemetry_path = os.path.join(BASE_DIR, "network_telemetry.csv")

    if not os.path.exists(telemetry_path):
        print("[!] No telemetry found. Did the receiver run?")
        return
    if not os.path.exists(GROUND_TRUTH_FILE):
        print("[!] No ground-truth log found.")
        return

    print("[*] Loading and merging data ...")
    telem_df = pd.read_csv(telemetry_path).sort_values("timestamp")
    truth_df = pd.read_csv(GROUND_TRUTH_FILE).sort_values("start_time")

    print(f"    Raw telemetry rows  : {len(telem_df)}")
    print(f"    Telemetry range     : "
          f"{telem_df['timestamp'].min():.1f} -> {telem_df['timestamp'].max():.1f}")
    print(f"    Ground truth range  : "
          f"{truth_df['start_time'].min():.1f} -> {truth_df['end_time'].max():.1f}")

    # ── Label assignment ──────────────────────────────────────────────────────
    TRANSITION_BUFFER = 0.1     # discard 0.1s on each edge of state transitions

    def get_label(ts):
        match = truth_df[
            (truth_df["start_time"] + TRANSITION_BUFFER <= ts) &
            (truth_df["end_time"]   - TRANSITION_BUFFER >= ts)
        ]
        return match.iloc[0]["label"] if not match.empty else "Transition"

    telem_df["network_condition"] = telem_df["timestamp"].apply(get_label)

    dropped = (telem_df["network_condition"] == "Transition").sum()
    print(f"    Rows dropped as Transition: {dropped}")
    telem_df = telem_df[telem_df["network_condition"] != "Transition"]

    print(f"\n    Pre-balance class distribution:")
    print(telem_df["network_condition"].value_counts().to_string())

    # ── Contiguous block balancing ────────────────────────────────────────────
    # Preserves time continuity within blocks — safe for rolling features.
    # Falls back to keeping all data if already balanced (±10%).
    

    

    # ── Save ──────────────────────────────────────────────────────────────────
    output_file = "realistic_network_dataset.csv"
    telem_df.to_csv(output_file, index=False)

    print(f"\n✅  Dataset saved -> {output_file}")
    print(f"    Final rows : {len(telem_df)}")
    print(f"    Class distribution:\n"
          f"{telem_df['network_condition'].value_counts().to_string()}")
    if "packets_per_window" in telem_df.columns:
        print(f"\n    Avg packets/window : {telem_df['packets_per_window'].mean():.1f}")
        print(f"    Min packets/window : {telem_df['packets_per_window'].min()}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("══════════════════════════════════════════════════")
    print(" Hospital Ward Network Simulator — Multi-Device  ")
    print(f" Devices: {len(DEVICES)}  |  Duration: {DURATION_SECONDS}s")
    print("══════════════════════════════════════════════════")

    print("\n[1/4] Setting up network namespaces ...")
    subprocess.run(["sudo", "./setup_namespaces.sh"], check=True)

    print("[2/4] Starting central receiver ...")
    rx_cmd = [
        "sudo", "ip", "netns", "exec", "receiver_ns",
        PYTHON_EXEC, "health_receiver.py"
    ]
    rx_proc = subprocess.Popen(rx_cmd)
    time.sleep(2)

    print(f"[3/4] Starting {len(DEVICES)} device senders ...")
    tx_procs = []
    for dev in DEVICES:
        tx_cmd = [
            "sudo", "ip", "netns", "exec", "sender_ns",
            PYTHON_EXEC, "health_sender.py",
            "--device-id",   str(dev["id"]),
            "--device-type", dev["type"],
        ]
        proc = subprocess.Popen(tx_cmd)
        tx_procs.append(proc)
        time.sleep(0.1)

    print(f"[4/4] Running simulation for {DURATION_SECONDS}s ...\n")
    start_time = time.time()

    try:
        with open(GROUND_TRUTH_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "label",
                             "applied_loss", "applied_delay", "applied_jitter"])

            while time.time() - start_time < DURATION_SECONDS:
                state = random.choices(
                    list(STATE_WEIGHTS.keys()),
                    weights=list(STATE_WEIGHTS.values())
                )[0]

                loss, delay, jitter, duration = sample_profile(state)
                print(f"  [{state:8}]  loss={loss:5.2f}%  "
                      f"delay={delay:5.0f}ms  jitter={jitter:4.0f}ms  "
                      f"for {duration:.0f}s")

                apply_netem(loss, delay, jitter)

                step_start = time.time()
                time.sleep(duration)
                step_end = time.time()

                writer.writerow([step_start, step_end, state,
                                 loss, delay, jitter])
                f.flush()

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")

    finally:
        print("\n[*] Stopping all processes ...")
        for p in tx_procs:
            p.terminate()
        rx_proc.terminate()

        try:
            run("sudo ip netns exec sender_ns tc qdisc del dev veth_s root")
        except Exception:
            pass

        time.sleep(1)
        merge_data()

if __name__ == "__main__":
    main()