#!/bin/bash
set -e

SENDER_NS=sender_ns
RECEIVER_NS=receiver_ns
SENDER_IP=10.0.0.1
RECEIVER_IP=10.0.0.2
NET=10.0.0.0/24
LOSS=10
DELAY=50
JITTER=15

echo "[*] Cleaning old namespaces (if any)..."
sudo ip netns del $SENDER_NS 2>/dev/null || true
sudo ip netns del $RECEIVER_NS 2>/dev/null || true

echo "[*] Creating namespaces..."
sudo ip netns add $SENDER_NS
sudo ip netns add $RECEIVER_NS

echo "[*] Creating veth pair..."
sudo ip link add veth_s type veth peer name veth_r
sudo ip link set veth_s netns $SENDER_NS
sudo ip link set veth_r netns $RECEIVER_NS

echo "[*] Assigning IP addresses..."
sudo ip netns exec $SENDER_NS ip addr add $SENDER_IP/24 dev veth_s
sudo ip netns exec $RECEIVER_NS ip addr add $RECEIVER_IP/24 dev veth_r

echo "[*] Bringing interfaces up..."
sudo ip netns exec $SENDER_NS ip link set veth_s up
sudo ip netns exec $RECEIVER_NS ip link set veth_r up
sudo ip netns exec $SENDER_NS ip link set lo up
sudo ip netns exec $RECEIVER_NS ip link set lo up

echo "[*] Adding routes..."
sudo ip netns exec $SENDER_NS ip route add $NET dev veth_s
sudo ip netns exec $RECEIVER_NS ip route add $NET dev veth_r

echo "[*] Applying packet loss and delay..."
sudo ip netns exec $SENDER_NS tc qdisc add dev veth_s root netem \
    loss ${LOSS}% delay ${DELAY}ms ${JITTER}ms

echo "[*] Testing connectivity..."
sudo ip netns exec $SENDER_NS ping -c 2 $RECEIVER_IP

echo "[✓] Network namespaces are READY"
echo "    Sender   : $SENDER_NS ($SENDER_IP)"
echo "    Receiver : $RECEIVER_NS ($RECEIVER_IP)"
echo "    Loss     : ${LOSS}%"
echo "    Delay    : ${DELAY}ms ± ${JITTER}ms"
