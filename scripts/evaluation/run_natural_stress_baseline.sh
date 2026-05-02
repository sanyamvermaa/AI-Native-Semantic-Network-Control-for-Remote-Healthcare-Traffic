#!/usr/bin/env bash

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Natural-Stress Baseline Evaluation
#
# Identical hospital ward WiFi scenario as run_natural_stress_closedloop.sh
# but uses static (RAW) senders with no ward controller.  Run this script
# and the closedloop variant under the same scenario to get a fair comparison.
#
# Scenario phases (same as closedloop script):
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
#   bash scripts/evaluation/run_natural_stress_baseline.sh [duration_seconds]
# Example:
#   bash scripts/evaluation/run_natural_stress_baseline.sh 660
#
# Output:
#   outputs/evaluation/baseline_natural_<timestamp>/
#   Then pass that directory to analyze_results.py --baseline-dir
# ─────────────────────────────────────────────────────────────────────────────

DURATION="${1:-660}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
else
    PYTHON_BIN="python3"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${PROJECT_DIR}/scripts/evaluation"
SETUP_SCRIPT="${PROJECT_DIR}/scripts/setup_namespaces.sh"

RUN_DIR="${PROJECT_DIR}/outputs/evaluation/baseline_natural_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${RUN_DIR}"

mkdir -p "${RUN_DIR}"

RECEIVER_LOG="${RUN_DIR}/receiver.log"
SENDER_PID_FILE="${RUN_DIR}/sender_pids.txt"
STRESS_LOG="${RUN_DIR}/stress.log"

: > "${SENDER_PID_FILE}"
: > "${STRESS_LOG}"

# ─────────────────────────────────────────────────────────────────────────────
# Netem helpers  (identical to closedloop variant for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

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

_cur_loss="0.3"
_cur_delay="12"
_cur_jitter="2"

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

dwell() {
    local duration="$1"
    local step=8
    local elapsed=0
    local rl dl jl

    while (( elapsed + step <= duration )); do
        rl=$(awk -v v="$_cur_loss"   -v r="$RANDOM" \
             'BEGIN{printf "%.3f", v*(0.92 + r/32767.0*0.16)}')
        dl=$(awk -v v="$_cur_delay"  -v r="$RANDOM" \
             'BEGIN{printf "%.1f", v*(0.93 + r/32767.0*0.14)}')
        jl=$(awk -v v="$_cur_jitter" -v r="$RANDOM" \
             'BEGIN{printf "%.1f", v*(0.90 + r/32767.0*0.20)}')
        rl=$(awk -v v="$rl" 'BEGIN{if(v<0.001)v=0.001; if(v>30)v=30; printf "%.3f",v}')
        jl=$(awk -v v="$jl" 'BEGIN{if(v<0.5)v=0.0;    if(v>90)v=90; printf "%.1f",v}')
        _apply_direct "$rl" "$dl" "$jl"
        sleep "$step"
        elapsed=$(( elapsed + step ))
    done

    local remaining=$(( duration - elapsed ))
    if (( remaining > 0 )); then sleep "$remaining"; fi
}

ward_natural_scenario() {
    echo "[SCENARIO] Phase 1/10 — Morning stable (quiet ward)" | tee -a "${STRESS_LOG}"
    interpolate_to "0.3" "12" "2" 3 5
    dwell 65

    echo "[SCENARIO] Phase 2/10 — Gradual degradation (shift change)" | tee -a "${STRESS_LOG}"
    interpolate_to "5.0" "70" "14" 10 4
    dwell 40

    echo "[SCENARIO] Phase 3/10 — Brief partial recovery" | tee -a "${STRESS_LOG}"
    interpolate_to "2.0" "35" "7" 5 4
    dwell 20

    echo "[SCENARIO] Phase 4/10 — Escalation to critical (busy hour)" | tee -a "${STRESS_LOG}"
    interpolate_to "14.0" "200" "55" 10 5
    dwell 30

    echo "[SCENARIO] Phase 5/10 — Sustained critical (peak congestion)" | tee -a "${STRESS_LOG}"
    interpolate_to "18.0" "260" "70" 4 4
    dwell 64

    echo "[SCENARIO] Phase 6/10 — Slow recovery to Unstable" | tee -a "${STRESS_LOG}"
    interpolate_to "4.0" "55" "11" 10 5
    dwell 20

    echo "[SCENARIO] Phase 7/10 — Stable recovery window" | tee -a "${STRESS_LOG}"
    interpolate_to "0.5" "15" "3" 6 4
    dwell 46

    echo "[SCENARIO] Phase 8/10 — Sharp interference burst (equipment)" | tee -a "${STRESS_LOG}"
    interpolate_to "20.0" "220" "72" 3 3
    dwell 21

    echo "[SCENARIO] Phase 9/10 — Fast recovery post-burst" | tee -a "${STRESS_LOG}"
    interpolate_to "1.0" "20" "4" 5 4
    dwell 20

    echo "[SCENARIO] Phase 10/10 — Stable finish" | tee -a "${STRESS_LOG}"
    interpolate_to "0.2" "10" "2" 4 4
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
    sudo ip netns exec sender_ns tc qdisc del dev veth_s root 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

echo "[BASELINE-NATURAL] Project dir  : ${PROJECT_DIR}"
echo "[BASELINE-NATURAL] Duration (s) : ${DURATION}"
echo "[BASELINE-NATURAL] Python       : ${PYTHON_BIN}"
echo "[BASELINE-NATURAL] Output dir   : ${RUN_DIR}"

echo "[BASELINE-NATURAL] Setting up namespaces..."
sudo "${SETUP_SCRIPT}"

echo "[BASELINE-NATURAL] Starting receiver in receiver_ns..."
umask 022
sudo env HEALTH_DATA_BASE_DIR="${DATA_DIR}" \
    HEALTH_NETWORK_MODEL_PATH="DISABLE" \
    ip netns exec receiver_ns \
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/scripts/closed_loop/health_receiver.py" \
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

echo "[BASELINE-NATURAL] Starting static (RAW) senders in sender_ns..."
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

echo "[BASELINE-NATURAL] Launching natural ward stress scenario (~630 s)..."
ward_natural_scenario &
STRESS_PID=$!

sleep "${DURATION}"

echo ""
echo "[BASELINE-NATURAL] ──────────────────────────────────────────────"
echo "[BASELINE-NATURAL] Run complete."
echo "[BASELINE-NATURAL] Output directory: ${RUN_DIR}"
echo "[BASELINE-NATURAL] ──────────────────────────────────────────────"
echo ""
echo "To evaluate, run:"
echo "  python scripts/evaluation/analyze_results.py \\"
echo "    --baseline-dir  ${RUN_DIR} \\"
echo "    --closedloop-dir outputs/evaluation/closedloop_natural_<timestamp> \\"
echo "    --output-dir    outputs/evaluation/figures_natural \\"
echo "    --model         models/xgboost_network_model.pkl"
echo ""
