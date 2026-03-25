#!/bin/bash
set -e

# Configuration
SENDER_NS=sender_ns
RECEIVER_NS=receiver_ns
SENDER_IP=10.0.0.1
RECEIVER_IP=10.0.0.2

echo "[*] Aggressive Cleanup..."
sudo pkill -f health_sender.py || true
sudo pkill -f health_receiver.py || true
sudo ip netns del $SENDER_NS 2>/dev/null || true
sudo ip netns del $RECEIVER_NS 2>/dev/null || true
# Delete veth links explicitly just in case
sudo ip link del veth_s 2>/dev/null || true
sudo ip link del veth_r 2>/dev/null || true

echo "[*] Creating namespaces..."
sudo ip netns add $SENDER_NS
sudo ip netns add $RECEIVER_NS

echo "[*] Creating veth pair..."
sudo ip link add veth_s type veth peer name veth_r
sudo ip link set veth_s netns $SENDER_NS
sudo ip link set veth_r netns $RECEIVER_NS

echo "[*] Assigning IP addresses..."
# The /24 CIDR automatically creates the necessary kernel routes
sudo ip netns exec $SENDER_NS ip addr add $SENDER_IP/24 dev veth_s
sudo ip netns exec $RECEIVER_NS ip addr add $RECEIVER_IP/24 dev veth_r

echo "[*] Bringing interfaces up..."
sudo ip netns exec $SENDER_NS ip link set veth_s up
sudo ip netns exec $RECEIVER_NS ip link set veth_r up
sudo ip netns exec $SENDER_NS ip link set lo up
sudo ip netns exec $RECEIVER_NS ip link set lo up

# REMOVED: Explicit route addition (This was causing the crash)

echo "[*] Initializing Traffic Control..."
# Initialize with 0 loss/delay so we can modify it later
sudo ip netns exec $SENDER_NS tc qdisc add dev veth_s root netem loss 0% delay 0ms

echo "[*] Testing connectivity..."
sleep 1
sudo ip netns exec $SENDER_NS ping -c 1 $RECEIVER_IP > /dev/null

echo "[âœ“] Network namespaces are READY"