#!/bin/bash

LOSS=$1
DELAY=$2
DURATION=$3



sudo ip netns exec sender_ns tc qdisc replace dev veth_s root netem loss ${LOSS}% delay ${DELAY}ms

sudo ip netns exec receiver_ns /usr/bin/python3 health_receiver.py &
R_PID=$!

sleep 2

sudo ip netns exec sender_ns /usr/bin/python3 health_sender.py &
S_PID=$!

sleep $DURATION

kill $S_PID
kill $R_PID

sudo ip netns exec sender_ns tc qdisc del dev veth_s root


