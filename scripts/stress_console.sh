#!/usr/bin/env bash

set -euo pipefail

# Interactive and command-line stress controller for tc netem.
# Designed for sender_ns:veth_s by default.
#
# Non-interactive examples:
#   bash scripts/stress_console.sh preset stable
#   bash scripts/stress_console.sh preset unstable
#   bash scripts/stress_console.sh preset critical
#   bash scripts/stress_console.sh set --loss 5 --delay 80 --jitter 15
#   bash scripts/stress_console.sh clear
#   bash scripts/stress_console.sh status
#
# Interactive:
#   bash scripts/stress_console.sh

NS_NAME="${NS_NAME:-sender_ns}"
IFACE="${IFACE:-veth_s}"

apply_cmd() {
    sudo ip netns exec "${NS_NAME}" tc "$@"
}

apply_set() {
    local loss="$1"
    local delay="$2"
    local jitter="$3"

    if awk "BEGIN{exit !(${jitter} < 1.0)}"; then
        apply_cmd qdisc replace dev "${IFACE}" root netem loss "${loss}%" delay "${delay}ms"
    else
        apply_cmd qdisc replace dev "${IFACE}" root netem loss "${loss}%" delay "${delay}ms" "${jitter}ms" distribution normal
    fi

    echo "[STRESS] Applied loss=${loss}% delay=${delay}ms jitter=${jitter}ms on ${NS_NAME}:${IFACE}"
}

apply_preset() {
    local preset="$1"
    case "$preset" in
        stable)
            apply_set 0.2 12 2
            ;;
        unstable)
            apply_set 5 80 15
            ;;
        critical)
            apply_set 15 220 50
            ;;
        recover)
            apply_set 0 5 0
            ;;
        *)
            echo "Unknown preset: ${preset}"
            echo "Available: stable | unstable | critical | recover"
            exit 1
            ;;
    esac
}

show_status() {
    echo "[STRESS] Current qdisc on ${NS_NAME}:${IFACE}"
    apply_cmd qdisc show dev "${IFACE}" || true
}

clear_stress() {
    apply_cmd qdisc del dev "${IFACE}" root 2>/dev/null || true
    echo "[STRESS] Cleared netem on ${NS_NAME}:${IFACE}"
}

usage() {
    cat <<EOF
Usage:
  bash scripts/stress_console.sh                         # interactive mode
  bash scripts/stress_console.sh preset <name>
  bash scripts/stress_console.sh set --loss X --delay Y --jitter Z
  bash scripts/stress_console.sh clear
  bash scripts/stress_console.sh status

Environment overrides:
  NS_NAME=<namespace> IFACE=<interface>
EOF
}

interactive_menu() {
    echo "=== Stress Console (${NS_NAME}:${IFACE}) ==="
    echo "Keep this running in one terminal while watching logs in another."

    while true; do
        echo
        echo "Choose action:"
        echo "  1) preset stable"
        echo "  2) preset unstable"
        echo "  3) preset critical"
        echo "  4) preset recover"
        echo "  5) custom set"
        echo "  6) status"
        echo "  7) clear"
        echo "  q) quit"
        read -r -p "> " choice

        case "$choice" in
            1) apply_preset stable ;;
            2) apply_preset unstable ;;
            3) apply_preset critical ;;
            4) apply_preset recover ;;
            5)
                read -r -p "loss (%)   : " loss
                read -r -p "delay (ms) : " delay
                read -r -p "jitter(ms) : " jitter
                apply_set "$loss" "$delay" "$jitter"
                ;;
            6) show_status ;;
            7) clear_stress ;;
            q|Q)
                echo "Exiting stress console."
                break
                ;;
            *)
                echo "Invalid choice."
                ;;
        esac
    done
}

if [[ "$#" -eq 0 ]]; then
    interactive_menu
    exit 0
fi

case "$1" in
    preset)
        [[ "$#" -eq 2 ]] || { usage; exit 1; }
        apply_preset "$2"
        ;;
    set)
        shift
        LOSS=""
        DELAY=""
        JITTER=""
        while [[ "$#" -gt 0 ]]; do
            case "$1" in
                --loss)
                    LOSS="$2"
                    shift 2
                    ;;
                --delay)
                    DELAY="$2"
                    shift 2
                    ;;
                --jitter)
                    JITTER="$2"
                    shift 2
                    ;;
                *)
                    usage
                    exit 1
                    ;;
            esac
        done

        [[ -n "$LOSS" && -n "$DELAY" && -n "$JITTER" ]] || { usage; exit 1; }
        apply_set "$LOSS" "$DELAY" "$JITTER"
        ;;
    clear)
        clear_stress
        ;;
    status)
        show_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
