"""
ward_controller.py — Centralized command listener for sender fleet.

Receives ONE UDP command stream from receiver and publishes latest fleet mode
to a shared state file consumed by all sender processes.
"""

import argparse
import json
import os
import socket
import time


# ── CLOSED LOOP ADDITION ─────────────────────────────────────
DEFAULT_BASE_DIR = os.environ.get("HEALTH_DATA_BASE_DIR", "/home/ayhm23/health_data/csv")
COMMAND_PORT = 5006
STATE_FILE_NAME = "ward_mode_state.json"
LOG_FILE_NAME = "ward_controller_log.csv"


def ensure_base_dir(base_dir):
    try:
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    except Exception as exc:
        fallback = os.path.join(os.getcwd(), "csv")
        os.makedirs(fallback, exist_ok=True)
        print(f"[WardController][WARN] BASE_DIR '{base_dir}' unavailable ({exc}). Using '{fallback}'.")
        return fallback


def atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-ip", type=str, default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=COMMAND_PORT)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    base_dir = ensure_base_dir(args.base_dir)
    state_path = os.path.join(base_dir, STATE_FILE_NAME)
    log_path = os.path.join(base_dir, LOG_FILE_NAME)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.bind_port))

    # Initialize state so senders have a deterministic startup mode.
    initial_payload = {
        "command": "FULL_ECG",
        "network_state": "Stable",
        "health_state": "NORMAL",
        "timestamp": time.time(),
        "consecutive_windows": 1,
    }
    atomic_write_json(state_path, initial_payload)

    with open(log_path, "w", newline="", encoding="utf-8") as logf:
        logf.write("recv_timestamp,command,network_state,health_state,source_ip,source_port\n")
        logf.flush()

        print(f"[WardController] Listening on {args.bind_ip}:{args.bind_port}")
        print(f"[WardController] State file: {state_path}")

        while True:
            try:
                data, addr = sock.recvfrom(4096)
                recv_ts = time.time()

                try:
                    payload = json.loads(data.decode("utf-8").strip())
                except Exception as exc:
                    print(f"[WardController][WARN] Invalid command packet: {exc}")
                    continue

                command = str(payload.get("command", "FULL_ECG"))
                net_state = str(payload.get("network_state", "N/A"))
                health_state = str(payload.get("health_state", "N/A"))
                cmd_ts = float(payload.get("timestamp", recv_ts))
                consecutive = int(payload.get("consecutive_windows", 1))

                state_payload = {
                    "command": command,
                    "network_state": net_state,
                    "health_state": health_state,
                    "timestamp": cmd_ts,
                    "consecutive_windows": consecutive,
                }
                atomic_write_json(state_path, state_payload)

                logf.write(
                    f"{recv_ts:.6f},{command},{net_state},{health_state},{addr[0]},{addr[1]}\n"
                )
                logf.flush()

                print(
                    f"[WARD CMD] {command}  (net={net_state}, health={health_state}, "
                    f"consecutive={consecutive})"
                )
            except KeyboardInterrupt:
                print("\n[WardController] Shutting down.")
                break
            except Exception as exc:
                print(f"[WardController][WARN] Runtime error: {exc}")


if __name__ == "__main__":
    main()
