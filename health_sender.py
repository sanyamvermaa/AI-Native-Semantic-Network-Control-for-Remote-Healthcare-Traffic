import socket
import time
import random
import csv
import os

# FIXED: Use a robust path that works for both header and loop
BASE_DIR = "/home/ayhm23/health_data/csv"
os.makedirs(BASE_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "sender_log.csv")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server = ("10.0.0.2", 9000)
SEND_INTERVAL = 0.01

# Write Header
with open(LOG_FILE, "w", newline="") as f:  
    writer = csv.writer(f)
    writer.writerow(["seq", "timestamp", "heart_rate", "label"])

print("Healthcare sender started...")

burst_mode = False
burst_end_time = 0
seq_num = 0

while True:
    now = time.time()

    # FIXED: Lower probability from 0.05 to 0.001 to allow more NORMAL data
    if not burst_mode and random.random() < 0.001:
        burst_mode = True
        burst_end_time = now + 5 

    if burst_mode:
        hr = random.randint(110, 130)
        if now > burst_end_time:
            burst_mode = False
    else:
        hr = random.randint(60, 90)

    # Patient Label (NOTE: This is Patient Health, NOT Network Health)
    if hr > 120: label = "CRITICAL"
    elif hr > 100: label = "ALERT"
    else: label = "NORMAL"

    seq_num += 1
    message = f"{seq_num},{now},{hr},{label}"
    sock.sendto(message.encode(), server)

    # FIXED: Use the correct absolute path variable
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seq_num, now, hr, label])

    time.sleep(SEND_INTERVAL)