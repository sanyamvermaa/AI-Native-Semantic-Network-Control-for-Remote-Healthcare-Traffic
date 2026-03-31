import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load telemetry data
df = pd.read_csv(LOGS_DIR / "network_telemetry.csv")

# Convert timestamps to relative time (seconds)
t0 = df["timestamp"].iloc[0]
df["time_sec"] = df["timestamp"] - t0


# -------------------------------
# 1. Bandwidth vs Throughput
# -------------------------------
plt.figure()
plt.plot(df["time_sec"], df["bandwidth_usage_bps"], label="Bandwidth Usage")
plt.plot(df["time_sec"], df["throughput_bps"], label="Throughput")
plt.xlabel("Time (seconds)")
plt.ylabel("Bytes per second")
plt.title("Bandwidth Usage vs Throughput")
plt.legend()
plt.grid(True)
plt.savefig(PLOTS_DIR / "bandwidth_vs_throughput.png", dpi=300)
plt.close()


# -------------------------------
# 2. Packet Loss Rate
# -------------------------------
plt.figure()
plt.plot(df["time_sec"], df["packet_loss_rate"])
plt.xlabel("Time (seconds)")
plt.ylabel("Packet Loss Rate")
plt.title("Packet Loss Over Time")
plt.grid(True)
plt.savefig(PLOTS_DIR / "packet_loss.png", dpi=300)
plt.close()


# -------------------------------
# 3. Jitter
# -------------------------------
plt.figure()
plt.plot(df["time_sec"], df["jitter"])
plt.xlabel("Time (seconds)")
plt.ylabel("Jitter (seconds)")
plt.title("Jitter Over Time")
plt.grid(True)
plt.savefig(PLOTS_DIR / "jitter.png", dpi=300)
plt.close()


# -------------------------------
# 4. Queue Length
# -------------------------------
plt.figure()
plt.plot(df["time_sec"], df["queue_length"])
plt.xlabel("Time (seconds)")
plt.ylabel("Queue Length")
plt.title("Queue Length Over Time")
plt.grid(True)
plt.savefig(PLOTS_DIR / "queue_length.png", dpi=300)
plt.close()

print(f"Plots saved successfully in {PLOTS_DIR}.")
