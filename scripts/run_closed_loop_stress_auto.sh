#!/usr/bin/env bash

set -euo pipefail

# Automatic stress test for closed-loop adaptation in namespace topology.
#
# Usage:
#   bash scripts/run_closed_loop_stress_auto.sh [duration_seconds] [stage_seconds]
# Example:
#   bash scripts/run_closed_loop_stress_auto.sh 180 20

DURATION="${1:-180}"
STAGE_SECONDS="${2:-20}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "/home/ayhm23/miniconda3/bin/python3" ]]; then
    PYTHON_BIN="/home/ayhm23/miniconda3/bin/python3"
else
    PYTHON_BIN="python3"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/outputs/results/stress_auto_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${HEALTH_DATA_BASE_DIR:-${PROJECT_DIR}/csv}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}"

RECEIVER_LOG="${RUN_DIR}/receiver.log"
WARD_LOG="${RUN_DIR}/ward_controller.log"
DASHBOARD_LOG="${RUN_DIR}/dashboard.log"
SENDER_PID_FILE="${RUN_DIR}/sender_pids.txt"
STRESS_LOG="${RUN_DIR}/stress.log"

: > "${SENDER_PID_FILE}"
: > "${STRESS_LOG}"

# Profiles aligned with project context.
stable_loss=(0.0 0.8)
stable_delay=(5 25)
stable_jitter=(0 4)

unstable_loss=(3.0 7.0)
unstable_delay=(40 120)
unstable_jitter=(8 25)

critical_loss=(12.0 22.0)
critical_delay=(180 380)
critical_jitter=(50 90)

rand_float() {
    local min="$1"
    local max="$2"
    awk -v min="$min" -v max="$max" 'BEGIN{srand(); printf "%.3f", min+rand()*(max-min)}'
}

apply_netem() {
    local loss="$1"
    local delay="$2"
    local jitter="$3"

    local cmd=(sudo ip netns exec sender_ns tc qdisc replace dev veth_s root netem)

    if awk "BEGIN{exit !(${jitter} < 1.0)}"; then
        "${cmd[@]}" loss "${loss}%" delay "${delay}ms"
    else
        "${cmd[@]}" loss "${loss}%" delay "${delay}ms" "${jitter}ms" distribution normal
    fi
}

apply_profile() {
    local profile="$1"
    local loss delay jitter

    case "$profile" in
        Stable)
            loss="$(rand_float "${stable_loss[0]}" "${stable_loss[1]}")"
            delay="$(rand_float "${stable_delay[0]}" "${stable_delay[1]}")"
            jitter="$(rand_float "${stable_jitter[0]}" "${stable_jitter[1]}")"
            ;;
        Unstable)
            loss="$(rand_float "${unstable_loss[0]}" "${unstable_loss[1]}")"
            delay="$(rand_float "${unstable_delay[0]}" "${unstable_delay[1]}")"
            jitter="$(rand_float "${unstable_jitter[0]}" "${unstable_jitter[1]}")"
            ;;
        Critical)
            loss="$(rand_float "${critical_loss[0]}" "${critical_loss[1]}")"
            delay="$(rand_float "${critical_delay[0]}" "${critical_delay[1]}")"
            jitter="$(rand_float "${critical_jitter[0]}" "${critical_jitter[1]}")"
            ;;
        *)
            echo "[STRESS][WARN] Unknown profile: $profile" | tee -a "${STRESS_LOG}"
            return
            ;;
    esac

    apply_netem "$loss" "$delay" "$jitter"
    echo "[STRESS] profile=${profile} loss=${loss}% delay=${delay}ms jitter=${jitter}ms" | tee -a "${STRESS_LOG}"
}

cleanup() {
    set +e

    if [[ -n "${STRESS_PID:-}" ]] && kill -0 "${STRESS_PID}" 2>/dev/null; then
        kill "${STRESS_PID}" 2>/dev/null || true
    fi

    if [[ -f "${SENDER_PID_FILE}" ]]; then
        while read -r spid; do
            if [[ -n "$spid" ]] && kill -0 "$spid" 2>/dev/null; then
                kill "$spid" 2>/dev/null || true
            fi
        done < "${SENDER_PID_FILE}"
    fi

    if [[ -n "${RECEIVER_PID:-}" ]] && kill -0 "${RECEIVER_PID}" 2>/dev/null; then
        kill "${RECEIVER_PID}" 2>/dev/null || true
    fi

    if [[ -n "${WARD_PID:-}" ]] && kill -0 "${WARD_PID}" 2>/dev/null; then
        kill "${WARD_PID}" 2>/dev/null || true
    fi

    if [[ -n "${DASHBOARD_PID:-}" ]] && kill -0 "${DASHBOARD_PID}" 2>/dev/null; then
        kill "${DASHBOARD_PID}" 2>/dev/null || true
    fi

    sudo ip netns exec sender_ns tc qdisc del dev veth_s root 2>/dev/null || true
}

trap cleanup EXIT INT TERM

cd "${PROJECT_DIR}"

echo "[RUN] Project dir     : ${PROJECT_DIR}"
echo "[RUN] Duration (sec)  : ${DURATION}"
echo "[RUN] Stage (sec)     : ${STAGE_SECONDS}"
echo "[RUN] Python          : ${PYTHON_BIN}"
echo "[RUN] Output dir      : ${RUN_DIR}"
echo "[RUN] Data dir        : ${DATA_DIR}"

echo "[RUN] Setting up namespaces..."
sudo ./setup_namespaces.sh

echo "[RUN] Starting receiver in receiver_ns..."
sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" ip netns exec receiver_ns "${PYTHON_BIN}" -u health_receiver.py > "${RECEIVER_LOG}" 2>&1 &
RECEIVER_PID=$!

sleep 2

echo "[RUN] Starting ward controller in sender_ns..."
sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" ip netns exec sender_ns "${PYTHON_BIN}" -u ward_controller.py --base-dir "${DATA_DIR}" > "${WARD_LOG}" 2>&1 &
WARD_PID=$!

sleep 1

echo "[RUN] Starting dashboard..."
HEALTH_DATA_BASE_DIR="${DATA_DIR}" "${PYTHON_BIN}" -u dashboard.py > "${DASHBOARD_LOG}" 2>&1 &
DASHBOARD_PID=$!
echo "[RUN] Dashboard at http://localhost:5050"

declare -a DEVICES=(
    "0 ECG"
    "1 ECG"
    "2 SpO2"
    "3 SpO2"
    "4 BloodPressure"
    "5 BloodPressure"
    "6 Temperature"
    "7 Respiration"
)

echo "[RUN] Starting senders in sender_ns..."
for dev in "${DEVICES[@]}"; do
    dev_id="$(awk '{print $1}' <<< "${dev}")"
    dev_type="$(awk '{print $2}' <<< "${dev}")"

    sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" ip netns exec sender_ns "${PYTHON_BIN}" -u health_sender.py \
        --device-id "${dev_id}" \
        --device-type "${dev_type}" \
        --receiver-ip "10.0.0.2" \
        --receiver-port "9000" \
        --base-dir "${DATA_DIR}" \
        > "${RUN_DIR}/sender_${dev_id}_${dev_type}.log" 2>&1 &

    echo "$!" >> "${SENDER_PID_FILE}"
done

stress_loop() {
    local start_ts now elapsed
    start_ts="$(date +%s)"

    while true; do
        now="$(date +%s)"
        elapsed=$((now - start_ts))
        if (( elapsed >= DURATION )); then
            break
        fi

        # Weighted schedule to trigger clear transitions.
        apply_profile "Stable"
        sleep "${STAGE_SECONDS}"

        now="$(date +%s)"
        elapsed=$((now - start_ts))
        if (( elapsed >= DURATION )); then break; fi

        apply_profile "Unstable"
        sleep "${STAGE_SECONDS}"

        now="$(date +%s)"
        elapsed=$((now - start_ts))
        if (( elapsed >= DURATION )); then break; fi

        apply_profile "Critical"
        sleep "${STAGE_SECONDS}"
    done
}

echo "[RUN] Starting automatic stress timeline..."
stress_loop > /dev/null 2>&1 &
STRESS_PID=$!

sleep "${DURATION}"

echo "[CHECK] Finalizing..."
cleanup

sleep 1

COMMAND_LOG="${DATA_DIR}/command_log.csv"
PREDICT_COUNT="$(grep -c "\[PREDICT\]" "${RECEIVER_LOG}" || true)"
LATENCY_COUNT="$(grep -c "\[LATENCY\]" "${RECEIVER_LOG}" || true)"
WARN_COUNT="$(grep -c "\[WARN\]" "${RECEIVER_LOG}" || true)"
WARD_CMD_COUNT="$(grep -c "\[WARD CMD\]" "${WARD_LOG}" || true)"

TRANSITIONS="$(grep -h "\[MODE CHANGE\]" "${RUN_DIR}"/sender_*.log 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "==== Automatic Stress Test Summary ===="
echo "Run directory        : ${RUN_DIR}"
echo "Receiver log         : ${RECEIVER_LOG}"
echo "Stress log           : ${STRESS_LOG}"
echo "Ward log             : ${WARD_LOG}"
echo "Receiver PREDICT cnt : ${PREDICT_COUNT}"
echo "Receiver LATENCY cnt : ${LATENCY_COUNT}"
echo "Receiver WARN cnt    : ${WARN_COUNT}"
echo "Ward command cnt     : ${WARD_CMD_COUNT}"
echo "Sender mode changes  : ${TRANSITIONS}"

if [[ -f "${COMMAND_LOG}" ]]; then
    echo "Command CSV          : ${COMMAND_LOG}"
    total_cmd="$(awk 'NR>1{c++} END{print c+0}' "${COMMAND_LOG}")"
    echo "Commands sent        : ${total_cmd}"

    echo "Command distribution :"
    awk -F',' 'NR>1 {cmd[$4]++} END {for (k in cmd) printf("  %s: %d\n", k, cmd[k]);}' "${COMMAND_LOG}" | sort

    echo "Avg latency (ms)     :"
    awk -F',' 'NR>1 {s+=$5; c++} END {if (c>0) printf("  %.2f\n", s/c); else print "  0.00"}' "${COMMAND_LOG}"
else
    echo "Command CSV          : Not found at ${COMMAND_LOG}"
fi

echo
echo "Tip: run 'tail -f ${RECEIVER_LOG}' during execution to watch live predictions."
