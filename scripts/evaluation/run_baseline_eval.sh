#!/usr/bin/env bash

set -euo pipefail

# Baseline evaluation runner for paper metrics.
# Launches 8 static (non-adaptive) senders + receiver over identical cycling
# netem profiles as run_closedloop_eval.sh, with NO ward controller and NO
# semantic adaptation — all devices transmit every sample in RAW mode.
#
# Usage:
#   bash scripts/evaluation/run_baseline_eval.sh [duration_seconds] [stage_seconds]
# Example:
#   bash scripts/evaluation/run_baseline_eval.sh 180 20
#
# Output directory printed at end: outputs/evaluation/baseline_<timestamp>/

DURATION="${1:-180}"
STAGE_SECONDS="${2:-20}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
else
    PYTHON_BIN="python3"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${PROJECT_DIR}/scripts/evaluation"
CL_DIR="${PROJECT_DIR}/scripts/closed_loop"
SETUP_SCRIPT="${PROJECT_DIR}/scripts/setup_namespaces.sh"

RUN_DIR="${PROJECT_DIR}/outputs/evaluation/baseline_$(date +%Y%m%d_%H%M%S)"
# Each run writes its CSV data into its own directory for isolation
DATA_DIR="${RUN_DIR}"

mkdir -p "${RUN_DIR}"

RECEIVER_LOG="${RUN_DIR}/receiver.log"
SENDER_PID_FILE="${RUN_DIR}/sender_pids.txt"
STRESS_LOG="${RUN_DIR}/stress.log"

: > "${SENDER_PID_FILE}"
: > "${STRESS_LOG}"

# ---------------------------------------------------------------------------
# Netem profiles — IDENTICAL to run_closedloop_eval.sh for fair comparison
# ---------------------------------------------------------------------------

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
            loss="$(rand_float  "${stable_loss[0]}"    "${stable_loss[1]}")"
            delay="$(rand_float "${stable_delay[0]}"   "${stable_delay[1]}")"
            jitter="$(rand_float "${stable_jitter[0]}" "${stable_jitter[1]}")"
            ;;
        Unstable)
            loss="$(rand_float  "${unstable_loss[0]}"    "${unstable_loss[1]}")"
            delay="$(rand_float "${unstable_delay[0]}"   "${unstable_delay[1]}")"
            jitter="$(rand_float "${unstable_jitter[0]}" "${unstable_jitter[1]}")"
            ;;
        Critical)
            loss="$(rand_float  "${critical_loss[0]}"    "${critical_loss[1]}")"
            delay="$(rand_float "${critical_delay[0]}"   "${critical_delay[1]}")"
            jitter="$(rand_float "${critical_jitter[0]}" "${critical_jitter[1]}")"
            ;;
        *)
            echo "[STRESS][WARN] Unknown profile: $profile" | tee -a "${STRESS_LOG}"
            return
            ;;
    esac
    apply_netem "$loss" "$delay" "$jitter"
    echo "[STRESS] profile=${profile} loss=${loss}% delay=${delay}ms jitter=${jitter}ms" \
        | tee -a "${STRESS_LOG}"
}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup() {
    set +e
    if [[ -n "${STRESS_PID:-}" ]] && kill -0 "${STRESS_PID}" 2>/dev/null; then
        kill "${STRESS_PID}" 2>/dev/null || true
    fi
    if [[ -f "${SENDER_PID_FILE}" ]]; then
        while read -r spid; do
            [[ -n "$spid" ]] && kill -0 "$spid" 2>/dev/null && kill "$spid" 2>/dev/null || true
        done < "${SENDER_PID_FILE}"
    fi
    if [[ -n "${RECEIVER_PID:-}" ]] && kill -0 "${RECEIVER_PID}" 2>/dev/null; then
        kill "${RECEIVER_PID}" 2>/dev/null || true
    fi
    sudo ip netns exec sender_ns tc qdisc del dev veth_s root 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

cd "${PROJECT_DIR}"

echo "[BASELINE] Project dir   : ${PROJECT_DIR}"
echo "[BASELINE] Duration (sec): ${DURATION}"
echo "[BASELINE] Stage (sec)   : ${STAGE_SECONDS}"
echo "[BASELINE] Python        : ${PYTHON_BIN}"
echo "[BASELINE] Output dir    : ${RUN_DIR}"

echo "[BASELINE] Setting up namespaces..."
sudo "${SETUP_SCRIPT}"

echo "[BASELINE] Starting receiver in receiver_ns..."
umask 022
sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" \
    ip netns exec receiver_ns \
    "${PYTHON_BIN}" -u "${CL_DIR}/health_receiver.py" \
    > "${RECEIVER_LOG}" 2>&1 &
RECEIVER_PID=$!

sleep 2

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

echo "[BASELINE] Starting static senders in sender_ns (no adaptation)..."
for dev in "${DEVICES[@]}"; do
    read -r dev_id dev_type <<< "$dev"
    SENDER_LOG="${RUN_DIR}/sender_${dev_id}_${dev_type}.log"
    sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" \
        ip netns exec sender_ns \
        "${PYTHON_BIN}" -u "${EVAL_DIR}/baseline_sender.py" \
        --device-id   "${dev_id}" \
        --device-type "${dev_type}" \
        --receiver-ip 10.0.0.2 \
        --base-dir    "${DATA_DIR}" \
        > "${SENDER_LOG}" 2>&1 &
    echo "$!" >> "${SENDER_PID_FILE}"
done

sleep 2

# Profile sequence — same as closedloop script to guarantee identical stress exposure
declare -a PROFILE_SEQUENCE=("Stable" "Stable" "Unstable" "Unstable" "Critical" "Critical")

(
    while true; do
        for profile in "${PROFILE_SEQUENCE[@]}"; do
            apply_profile "$profile"
            sleep "${STAGE_SECONDS}"
        done
    done
) &
STRESS_PID=$!

echo "[BASELINE] Stress profiles cycling every ${STAGE_SECONDS}s for ${DURATION}s..."
sleep "${DURATION}"

echo ""
echo "[BASELINE] ─────────────────────────────────────────────────"
echo "[BASELINE] Run complete."
echo "[BASELINE] Output directory: ${RUN_DIR}"
echo "[BASELINE] ─────────────────────────────────────────────────"
echo ""
