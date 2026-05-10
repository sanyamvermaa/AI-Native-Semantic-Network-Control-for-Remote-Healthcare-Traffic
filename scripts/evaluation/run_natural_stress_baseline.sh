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

# ── Deterministic stress seed ─────────────────────────────────────────────────
# MUST match the value in run_natural_stress_closedloop.sh.
# Seeding bash $RANDOM makes dwell() micro-noise identical across both runs,
# so the ONLY experimental variable is the semantic adaptation layer.
STRESS_SEED="${2:-20260101}"
RANDOM=$STRESS_SEED

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

# ─────────────────────────────────────────────────────────────────────────────
# replay_manifest  <manifest_csv>
#
#   Replays the pre-generated scenario_manifest.csv line by line.
#   Each row: step,phase,loss_pct,delay_ms,jitter_ms,sleep_s
#
#   This replaces dwell() + ward_natural_scenario() entirely.
#   Using a pre-generated manifest (Python random seed 20260101) guarantees
#   byte-identical network conditions across baseline and closed-loop runs,
#   regardless of bash version, PID, or process startup timing.
# ─────────────────────────────────────────────────────────────────────────────
replay_manifest() {
    local manifest="$1"
    local prev_phase=""

    # Strip Windows CRLF and skip CSV header, process each row
    tail -n +2 "$manifest" | tr -d '\r' | while IFS=',' read -r _step phase loss delay jitter slp; do
        # Print phase banner when phase changes
        if [[ "$phase" != "$prev_phase" ]]; then
            echo "[SCENARIO] ${phase}" | tee -a "${STRESS_LOG}"
            prev_phase="$phase"
        fi
        _apply_direct "$loss" "$delay" "$jitter"
        sleep "$slp"
    done

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
        --seed-base   20260101 \
        > "${SENDER_LOG}" 2>&1 &
    echo "$!" >> "${SENDER_PID_FILE}"
done

sleep 2

echo "[BASELINE-NATURAL] Launching natural ward stress scenario (~630 s)..."
MANIFEST_FILE="${PROJECT_DIR}/scripts/evaluation/scenario_manifest.csv"
if [[ ! -f "${MANIFEST_FILE}" ]]; then
    echo "[ERROR] scenario_manifest.csv not found. Generate it first:"
    echo "  python3 scripts/evaluation/generate_stress_manifest.py"
    exit 1
fi
echo "[BASELINE-NATURAL] Replaying manifest: ${MANIFEST_FILE}"
replay_manifest "${MANIFEST_FILE}" &
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
