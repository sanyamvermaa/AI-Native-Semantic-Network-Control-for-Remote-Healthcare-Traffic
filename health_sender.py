import socket
import time
import random
import csv

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

SERVER_IP = "127.0.0.1"
SERVER_PORT = 9000
server = (SERVER_IP, SERVER_PORT)

SEND_INTERVAL = 0.01  # seconds

# Sequence number
seq_num = 0

# CSV file
with open("csv/sender_log.csv", "w", newline="") as f:  
    writer = csv.writer(f)
    writer.writerow(["seq", "timestamp", "heart_rate", "label"])

print("Healthcare sender started...")

burst_mode = False
burst_end_time = 0

while True:
    now = time.time()

    # Occasionally trigger abnormal burst
    if not burst_mode and random.random() < 0.05:
        burst_mode = True
        burst_end_time = now + 5  # 5 seconds burst

    if burst_mode:
        hr = random.randint(110, 130)
        if now > burst_end_time:
            burst_mode = False
    else:
        hr = random.randint(60, 90)

    # Label logic
    if hr > 120:
        label = "CRITICAL"
    elif hr > 100:
        label = "ALERT"
    else:
        label = "NORMAL"

    # Increment sequence number
    seq_num += 1

    # NEW message format
    message = f"{seq_num},{now},{hr},{label}"
    sock.sendto(message.encode(), server)

    # Log to CSV
    with open("csv/sender_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seq_num, now, hr, label])

    print("Sent:", message)
    time.sleep(SEND_INTERVAL)
