import socket
import time
import csv

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 9000))

print("Healthcare receiver listening...")

with open("receiver_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seq", "send_time", "recv_time", "heart_rate", "label", "delay"])

telemetry_file = open("network_telemetry.csv", "w", newline="")
telemetry_writer = csv.writer(telemetry_file)
telemetry_writer.writerow([
    "timestamp",
    "bandwidth_usage_bps",
    "throughput_bps",
    "packet_loss_rate",
    "jitter",
    "queue_length"
])

bytes_received = 0
bytes_attempted = 0

received_packets = 0
lost_packets = 0
expected_seq = 1

# Jitter tracking
prev_delay = None
jitter_sum = 0.0
jitter_count = 0

# Modeled queue length
queue_length = 0

telemetry_interval = 1.0
last_telemetry_time = time.time()

while True:
    # print("Waiting for data...")
    data, addr = sock.recvfrom(1024)
    # print("Data received:", data)
    # print("Data:", data)

    recv_time = time.time()

    message = data.decode()
    # print("RAW MESSAGE RECEIVED:", repr(message))
    seq, send_time, hr, label = message.split(",")
    # print("SPLIT PARTS:", parts)

    seq = int(seq)
    send_time = float(send_time)

    delay = recv_time - send_time

    # Jitter update
    if prev_delay is not None:
        jitter_sum += abs(delay - prev_delay)
        jitter_count += 1
    prev_delay = delay

    # Packet loss detection
    if seq > expected_seq:
        lost_packets += (seq - expected_seq)
    expected_seq = seq + 1
    received_packets += 1

    with open("receiver_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seq, send_time, recv_time, hr, label, delay])

    packet_size = len(data)
    bytes_received += packet_size
    bytes_attempted += packet_size

    current_time = time.time()

    if current_time - last_telemetry_time >= telemetry_interval:
        throughput_bps = bytes_received / telemetry_interval
        bandwidth_usage_bps = bytes_attempted / telemetry_interval

        total_attempted = received_packets + lost_packets
        if total_attempted > 0:
            packet_loss_rate = lost_packets / total_attempted
        else:
            packet_loss_rate = 0.0

        if jitter_count > 0:
            avg_jitter = jitter_sum / jitter_count
        else:
            avg_jitter = 0.0

        queue_length += lost_packets

        telemetry_writer.writerow([
            int(current_time),
            round(bandwidth_usage_bps, 2),
            round(throughput_bps, 2),
            round(packet_loss_rate, 3),
            round(avg_jitter, 6),
            queue_length
        ])

        # Reset window counters
        bytes_received = 0
        bytes_attempted = 0
        received_packets = 0
        lost_packets = 0
        jitter_sum = 0.0
        jitter_count = 0
        last_telemetry_time = current_time
