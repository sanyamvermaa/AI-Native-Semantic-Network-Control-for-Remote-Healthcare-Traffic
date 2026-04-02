#!/usr/bin/env bash

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Natural-Stress Closed-Loop Evaluation
#
# Simulates a realistic hospital ward WiFi scenario instead of hard block
# jumps.  Network conditions drift organically between states using linear
# interpolation and micro-variation noise during dwell periods:
#
#   Phase 1  (  0- 80s) : Morning stable — quiet ward
#   Phase 2  ( 80-160s) : Gradual degradation — shift change, devices joining
#   Phase 3  (160-200s) : Brief partial recovery
#   Phase 4  (200-280s) : Escalation to critical — busy hour, congestion
#   Phase 5  (280-360s) : Sustained critical — peak stress
#   Phase 6  (360-430s) : Slow recovery to unstable zone
#   Phase 7  (430-500s) : Stable recovery window
#   Phase 8  (500-530s) : Sharp interference burst (equipment / microwave)
#   Phase 9  (530-570s) : Fast recovery after burst clears
#   Phase 10 (570-630s) : Stable finish
#
# Usage:
#   bash scripts/evaluation/run_natural_stress_closedloop.sh [duration_seconds]
# Example (run for full scenario + 30s buffer):
#   bash scripts/evaluation/run_natural_stress_closedloop.sh 660
#
# Output:
#   outputs/evaluation/closedloop_natural_<timestamp>/
#   Then pass that directory to analyze_results.py --closedloop-dir
# ─────────────────────────────────────────────────────────────────────────────

DURATION="${1:-660}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
else
    PYTHON_BIN="python3"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CL_DIR="${PROJECT_DIR}/scripts/closed_loop"
SETUP_SCRIPT="${PROJECT_DIR}/scripts/setup_namespaces.sh"

RUN_DIR="${PROJECT_DIR}/outputs/evaluation/closedloop_natural_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${RUN_DIR}"

mkdir -p "${RUN_DIR}"

RECEIVER_LOG="${RUN_DIR}/receiver.log"
WARD_LOG="${RUN_DIR}/ward_controller.log"
SENDER_PID_FILE="${RUN_DIR}/sender_pids.txt"
STRESS_LOG="${RUN_DIR}/stress.log"

: > "${SENDER_PID_FILE}"
: > "${STRESS_LOG}"

# ─────────────────────────────────────────────────────────────────────────────
# Netem helpers
# ─────────────────────────────────────────────────────────────────────────────

# Apply an exact (loss, delay, jitter) triple to the netem qdisc.
_apply_direct() {
    local loss="$1" delay="$2" jitter="$3"
    local cmd=(sudo ip netns exec sender_ns tc qdisc replace dev veth_s root netem)
    if awk "BEGIN{exit !(${jitter} < 1.0)}"; then
        "${cmd[@]}" loss "${loss}%" delay "${delay}ms"
    else
        "${cmd[@]}" loss "${loss}%" delay "${delay}ms" "${jitter}ms" distribution normal
    fi
    echo "[STRESS] loss=${loss}% delay=${delay}ms jitter=${jitter}ms" | tee -a "${STRESS_LOG}"
}

# ─────────────────────────────────────────────────────────────────────────────
# State tracking (module-level "current" netem values used by interpolate_to)
# ─────────────────────────────────────────────────────────────────────────────
_cur_loss="0.3"
_cur_delay="12"
_cur_jitter="2"

# ─────────────────────────────────────────────────────────────────────────────
# interpolate_to <to_loss> <to_delay> <to_jitter> [steps=8] [step_sleep=4]
#
#   Linearly interpolates from (_cur_*) to the target values over
#   (steps × step_sleep) seconds, updating netem at each step.
#   Updates _cur_* on completion.
# ─────────────────────────────────────────────────────────────────────────────
interpolate_to() {
    local to_loss="$1" to_delay="$2" to_jitter="$3"
    local steps="${4:-8}" step_sleep="${5:-4}"
    local i frac new_loss new_delay new_jitter

    for i in $(seq 1 "$steps"); do
        frac=$(awk "BEGIN{printf \"%.6f\", $i / $steps}")
        new_loss=$(awk  "BEGIN{v=$_cur_loss  + ($to_loss  - $_cur_loss)  * $frac; if(v<0.001)v=0.001; if(v>30)v=30; printf \"%.3f\",v}")
        new_delay=$(awk "BEGIN{v=$_cur_delay + ($to_delay - $_cur_delay) * $frac; if(v<1)v=1; printf \"%.1f\",v}")
        new_jitter=$(awk "BEGIN{v=$_cur_jitter+($to_jitter-$_cur_jitter)* $frac; if(v<0)v=0; if(v>90)v=90; printf \"%.1f\",v}")
        _apply_direct "$new_loss" "$new_delay" "$new_jitter"
        sleep "$step_sleep"
    done

    _cur_loss="$to_loss"
    _cur_delay="$to_delay"
    _cur_jitter="$to_jitter"
}

# ─────────────────────────────────────────────────────────────────────────────
# dwell <duration_seconds>
#
#   Holds at the current _cur_* values for <duration_seconds>, but applies
#   gentle ±10% random noise every 8 s to simulate natural micro-variation
#   (device contention, background radio interference, etc.).
# ─────────────────────────────────────────────────────────────────────────────
dwell() {
    local duration="$1"
    local step=8
    local elapsed=0
    local rl dl jl

    while (( elapsed + step <= duration )); do
        # Gaussian-like ±10% noise using $RANDOM (0-32767)
        rl=$(awk -v v="$_cur_loss"   -v r="$RANDOM" \
             'BEGIN{printf "%.3f", v*(0.92 + r/32767.0*0.16)}')
        dl=$(awk -v v="$_cur_delay"  -v r="$RANDOM" \
             'BEGIN{printf "%.1f", v*(0.93 + r/32767.0*0.14)}')
        jl=$(awk -v v="$_cur_jitter" -v r="$RANDOM" \
             'BEGIN{printf "%.1f", v*(0.90 + r/32767.0*0.20)}')
        # Clamp to safe ranges
        rl=$(awk -v v="$rl" 'BEGIN{if(v<0.001)v=0.001; if(v>30)v=30; printf "%.3f",v}')
        jl=$(awk -v v="$jl" 'BEGIN{if(v<0.5)v=0.0;    if(v>90)v=90; printf "%.1f",v}')
        _apply_direct "$rl" "$dl" "$jl"
        sleep "$step"
        elapsed=$(( elapsed + step ))
    done

    local remaining=$(( duration - elapsed ))
    if (( remaining > 0 )); then sleep "$remaining"; fi
}

# ─────────────────────────────────────────────────────────────────────────────
# ward_natural_scenario
#
#   Drives a ~630 s hospital ward WiFi degradation scenario with organic
#   transitions.  Designed to be launched as a background job.
# ─────────────────────────────────────────────────────────────────────────────
ward_natural_scenario() {
    # Phase 1: Morning stable — quiet ward (0–80 s)
    echo "[SCENARIO] Phase 1/10 — Morning stable (quiet ward)" | tee -a "${STRESS_LOG}"
    interpolate_to "0.3" "12" "2" 3 5      # 15 s gentle warm-up
    dwell 65

    # Phase 2: Gradual degradation — shift change, devices joining (80–160 s)
    echo "[SCENARIO] Phase 2/10 — Gradual degradation (shift change)" | tee -a "${STRESS_LOG}"
    interpolate_to "5.0" "70" "14" 10 4    # 40 s drift to Unstable
    dwell 40

    # Phase 3: Brief partial recovery (160–200 s)
    echo "[SCENARIO] Phase 3/10 — Brief partial recovery" | tee -a "${STRESS_LOG}"
    interpolate_to "2.0" "35" "7" 5 4      # 20 s partial ease
    dwell 20

    # Phase 4: Escalation toward critical — busy hour (200–280 s)
    echo "[SCENARIO] Phase 4/10 — Escalation to critical (busy hour)" | tee -a "${STRESS_LOG}"
    interpolate_to "14.0" "200" "55" 10 5  # 50 s gradual escalation
    dwell 30

    # Phase 5: Sustained critical — peak congestion (280–360 s)
    echo "[SCENARIO] Phase 5/10 — Sustained critical (peak congestion)" | tee -a "${STRESS_LOG}"
    interpolate_to "18.0" "260" "70" 4 4   # 16 s push to peak
    dwell 64

    # Phase 6: Slow recovery to Unstable zone (360–430 s)
    echo "[SCENARIO] Phase 6/10 — Slow recovery to Unstable" | tee -a "${STRESS_LOG}"
    interpolate_to "4.0" "55" "11" 10 5    # 50 s slow descent
    dwell 20

    # Phase 7: Stable recovery window (430–500 s)
    echo "[SCENARIO] Phase 7/10 — Stable recovery window" | tee -a "${STRESS_LOG}"
    interpolate_to "0.5" "15" "3" 6 4      # 24 s ease to Stable
    dwell 46

    # Phase 8: Sharp interference burst — medical equipment / microwave (500–530 s)
    echo "[SCENARIO] Phase 8/10 — Sharp interference burst (equipment)" | tee -a "${STRESS_LOG}"
    interpolate_to "20.0" "220" "72" 3 3   # 9 s spike (fast)
    dwell 21

    # Phase 9: Fast recovery after burst clears (530–570 s)
    echo "[SCENARIO] Phase 9/10 — Fast recovery post-burst" | tee -a "${STRESS_LOG}"
    interpolate_to "1.0" "20" "4" 5 4      # 20 s brisk recovery
    dwell 20

    # Phase 10: Stable finish (570–630 s)
    echo "[SCENARIO] Phase 10/10 — Stable finish" | tee -a "${STRESS_LOG}"
    interpolate_to "0.2" "10" "2" 4 4      # 16 s gentle settle
    dwell 44

    echo "[SCENARIO] Natural ward scenario complete (~630 s total)" | tee -a "${STRESS_LOG}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
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
    if [[ -n "${WARD_PID:-}" ]] && kill -0 "${WARD_PID}" 2>/dev/null; then
        kill "${WARD_PID}" 2>/dev/null || true
    fi
    sudo ip netns exec sender_ns tc qdisc del dev veth_s root 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

echo "[CLOSEDLOOP-NATURAL] Project dir  : ${PROJECT_DIR}"
echo "[CLOSEDLOOP-NATURAL] Duration (s) : ${DURATION}"
echo "[CLOSEDLOOP-NATURAL] Python       : ${PYTHON_BIN}"
echo "[CLOSEDLOOP-NATURAL] Output dir   : ${RUN_DIR}"

echo "[CLOSEDLOOP-NATURAL] Setting up namespaces..."
sudo "${SETUP_SCRIPT}"

echo "[CLOSEDLOOP-NATURAL] Starting receiver in receiver_ns..."
umask 022
sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" \
    HEALTH_NETWORK_MODEL_PATH="${PROJECT_DIR}/models/best_network_model.pkl" \
    ip netns exec receiver_ns \
    "${PYTHON_BIN}" -u "${CL_DIR}/health_receiver.py" \
    > "${RECEIVER_LOG}" 2>&1 &
RECEIVER_PID=$!

sleep 2

echo "[CLOSEDLOOP-NATURAL] Starting ward controller in sender_ns..."
sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" \
    ip netns exec sender_ns \
    "${PYTHON_BIN}" -u "${CL_DIR}/ward_controller.py" \
    --base-dir "${DATA_DIR}" \
    --broadcast-ip 10.0.0.1 \
    > "${WARD_LOG}" 2>&1 &
WARD_PID=$!

sleep 1

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

echo "[CLOSEDLOOP-NATURAL] Starting adaptive senders in sender_ns..."
for dev in "${DEVICES[@]}"; do
    read -r dev_id dev_type <<< "$dev"
    SENDER_LOG="${RUN_DIR}/sender_${dev_id}_${dev_type}.log"
    sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" \
        ip netns exec sender_ns \
        "${PYTHON_BIN}" -u "${CL_DIR}/health_sender.py" \
        --device-id   "${dev_id}" \
        --device-type "${dev_type}" \
        --receiver-ip 10.0.0.2 \
        --base-dir    "${DATA_DIR}" \
        > "${SENDER_LOG}" 2>&1 &
    echo "$!" >> "${SENDER_PID_FILE}"
done

sleep 2

echo "[CLOSEDLOOP-NATURAL] Launching natural ward stress scenario (~630 s)..."
ward_natural_scenario &
STRESS_PID=$!

sleep "${DURATION}"

echo ""
echo "[CLOSEDLOOP-NATURAL] ──────────────────────────────────────────────"
echo "[CLOSEDLOOP-NATURAL] Run complete."
echo "[CLOSEDLOOP-NATURAL] Output directory: ${RUN_DIR}"
echo "[CLOSEDLOOP-NATURAL] ──────────────────────────────────────────────"
echo ""
echo "To evaluate, run:"
echo "  python scripts/evaluation/analyze_results.py \\"
echo "    --baseline-dir  outputs/evaluation/baseline_natural_<timestamp> \\"
echo "    --closedloop-dir ${RUN_DIR} \\"
echo "    --output-dir    outputs/evaluation/figures_natural \\"
echo "    --model         models/xgboost_network_model.pkl"
echo ""
