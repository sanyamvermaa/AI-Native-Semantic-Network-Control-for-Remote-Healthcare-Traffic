import argparse
import csv
import json
import os
import socket
import time
from pathlib import Path

from common import DEVICE_LAYOUT, default_base_dir, normalize_health_state, normalize_network_state, policy_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-ip", type=str, default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=5006)
    parser.add_argument("--base-dir", type=str, default=default_base_dir())
    parser.add_argument("--window-timeout", type=float, default=3.0)
    parser.add_argument("--broadcast-ip", type=str, default="127.0.0.1")
    parser.add_argument("--sender-control-base-port", type=int, default=6000)
    args = parser.parse_args()

    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)

    state_path = os.path.join(base_dir, "ward_mode_state.json")
    command_log_path = os.path.join(base_dir, "command_log.csv")

    if not os.path.exists(command_log_path):
        with open(command_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "network_state",
                "health_state",
                "command",
                "latency_ms",
                "consecutive_windows",
            ])

    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind((args.listen_ip, args.listen_port))
    in_sock.settimeout(0.5)

    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[WARD] Listening on {args.listen_ip}:{args.listen_port}")

    last_command = "N/A"
    last_state = "UNKNOWN"
    consecutive = 0
    last_packet_ts = 0.0

    try:
        while True:
            now = time.time()
            try:
                data, _ = in_sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8", errors="replace"))
                if not isinstance(payload, dict):
                    continue
            except socket.timeout:
                if now - last_packet_ts > args.window_timeout and last_command != "SEMANTIC_SUMMARY":
                    network_state = "Critical"
                    health_state = "UNKNOWN"
                    command = "SEMANTIC_SUMMARY"
                    consecutive = 1 if last_state != network_state else consecutive + 1
                    last_state = network_state
                    last_command = command
                    snapshot = {
                        "command": command,
                        "network_state": network_state,
                        "health_state": health_state,
                        "timestamp": now,
                        "consecutive_windows": consecutive,
                    }
                    try:
                        with open(state_path, "w", encoding="utf-8") as f:
                            json.dump(snapshot, f)
                    except Exception:
                        pass
                continue
            except Exception:
                continue

            recv_ts = time.time()
            last_packet_ts = recv_ts

            network_state = normalize_network_state(payload.get("network_state"))
            health_state = normalize_health_state(payload.get("health_state"))
            source_ts = float(payload.get("timestamp") or recv_ts)
            latency_ms = max(0.0, (recv_ts - source_ts) * 1000.0)

            if network_state == "UNKNOWN":
                network_state = "Critical"

            command = policy_command(network_state, health_state)

            if network_state == last_state:
                consecutive += 1
            else:
                consecutive = 1
                last_state = network_state

            snapshot = {
                "command": command,
                "network_state": network_state,
                "health_state": health_state,
                "timestamp": recv_ts,
                "consecutive_windows": consecutive,
            }

            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f)
            except Exception:
                pass

            try:
                with open(command_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            round(recv_ts, 6),
                            network_state,
                            health_state,
                            command,
                            round(latency_ms, 3),
                            consecutive,
                        ]
                    )
            except Exception:
                pass

            control = {
                "command": command,
                "network_state": network_state,
                "health_state": health_state,
                "timestamp": recv_ts,
                "consecutive_windows": consecutive,
            }

            encoded = json.dumps(control).encode("utf-8")
            for dev_id, _ in DEVICE_LAYOUT:
                try:
                    out_sock.sendto(encoded, (args.broadcast_ip, args.sender_control_base_port + dev_id))
                except Exception:
                    pass

            if command != last_command:
                print(
                    f"[WARD CMD] {network_state}/{health_state} -> {command} "
                    f"(latency={latency_ms:.1f}ms, windows={consecutive})"
                )
                last_command = command

    except KeyboardInterrupt:
        print("\n[WARD] Shutdown requested.")
    finally:
        in_sock.close()
        out_sock.close()


if __name__ == "__main__":
    main()
