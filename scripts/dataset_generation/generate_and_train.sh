#!/usr/bin/env bash
# =============================================================================
#  generate_and_train.sh
#
#  PURPOSE:
#    1. Simulate a realistic hospital-ward WiFi stress scenario using the
#       SAME netem infrastructure as run_natural_stress_baseline.sh
#       (network namespaces + tc qdisc + baseline_sender.py + health_receiver.py)
#    2. Collect the resulting network_telemetry.csv
#    3. Retrain the XGBoost classifier on ALL available training telemetry
#       (including this new run) via the new train_from_pipeline.py
#    4. Print a final labeled summary: rows collected this run, total training
#       rows, class distribution, and sanity-check predictions.
#
#  KEY DIFFERENCES from generate_live_dataset.sh:
#    - Single comprehensive run (no repeated loop; more phases, more blur)
#    - The netem scenario has FUZZY transitions — overlapping state boundaries
#      achieved by adding per-step Gaussian noise to every parameter so the
#      model learns from ambiguous real-world telemetry, not just textbook zones.
#    - The training step is integrated; you don't need to run a second script.
#    - A rich summary is printed at the end.
#
#  USAGE:
#    bash scripts/dataset_generation/generate_and_train.sh [duration_seconds]
#    Default duration: 660 seconds (~11 min, covers all 10 phases)
#
#  OUTPUT:
#    outputs/training_datasets/pipeline_<timestamp>/
#      network_telemetry.csv   — raw telemetry from this run
#      stress.log              — per-step netem changes applied
#      receiver.log            — health_receiver.py stdout
#      sender_*.log            — per-device sender logs
#    models/xgboost_network_model.pkl  — updated classifier
#    models/xgboost_network_model.json — scale metadata sidecar
# =============================================================================

set -euo pipefail

DURATION="${1:-660}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_DIR="${PROJECT_DIR}/scripts/evaluation"
DATAGEN_DIR="${PROJECT_DIR}/scripts/dataset_generation"
SETUP_SCRIPT="${PROJECT_DIR}/scripts/setup_namespaces.sh"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${PROJECT_DIR}/outputs/training_datasets/pipeline_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

RECEIVER_LOG="${RUN_DIR}/receiver.log"
SENDER_PID_FILE="${RUN_DIR}/sender_pids.txt"
STRESS_LOG="${RUN_DIR}/stress.log"

: > "${SENDER_PID_FILE}"
: > "${STRESS_LOG}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
else
    PYTHON_BIN="python3"
fi

echo "=========================================================="
echo "  generate_and_train.sh"
echo "  Project : ${PROJECT_DIR}"
echo "  Run dir : ${RUN_DIR}"
echo "  Duration: ${DURATION}s"
echo "  Python  : ${PYTHON_BIN}"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Netem helpers — identical mechanics to run_natural_stress_baseline.sh
# but we add Gaussian noise on every step so boundaries blur realistically.
# ---------------------------------------------------------------------------

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

# gaussian_noise <mean> <pct>  → returns mean ± (pct/100)*mean*N(0,1)
# The noise is clamped so we never go negative or absurdly large.
gaussian_noise() {
    local mean="$1" pct="${2:-15}"
    awk -v m="$mean" -v p="$pct" -v r1="$RANDOM" -v r2="$RANDOM" '
    BEGIN {
        # Box-Muller transform with two uniform [0,1] variates
        u1 = (r1 + 0.5) / 32768.0
        u2 = (r2 + 0.5) / 32768.0
        z  = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265 * u2)
        v  = m + m * (p / 100.0) * z
        if (v < 0.001) v = 0.001
        printf "%.3f", v
    }'
}

# interpolate_noisy: same linear ramp as the baseline script but each step
# is perturbed by ±noise_pct% Gaussian noise so the trajectory looks organic.
interpolate_noisy() {
    local to_loss="$1" to_delay="$2" to_jitter="$3"
    local steps="${4:-8}" step_sleep="${5:-4}" noise_pct="${6:-18}"
    local i frac base_loss base_delay base_jitter new_loss new_delay new_jitter

    for i in $(seq 1 "$steps"); do
        frac=$(awk "BEGIN{printf \"%.6f\", $i / $steps}")
        base_loss=$(awk  "BEGIN{v=$_cur_loss  + ($to_loss  - $_cur_loss)  * $frac; if(v<0.001)v=0.001; if(v>30)v=30; printf \"%.3f\",v}")
        base_delay=$(awk "BEGIN{v=$_cur_delay + ($to_delay - $_cur_delay) * $frac; if(v<1)v=1; printf \"%.1f\",v}")
        base_jitter=$(awk "BEGIN{v=$_cur_jitter+($to_jitter-$_cur_jitter)* $frac; if(v<0)v=0; if(v>90)v=90; printf \"%.1f\",v}")
        # Add realistic noise to each step
        new_loss=$(gaussian_noise "$base_loss"   "$noise_pct")
        new_delay=$(gaussian_noise "$base_delay"  "$(awk "BEGIN{printf \"%d\", $noise_pct - 5}")")
        new_jitter=$(gaussian_noise "$base_jitter" "$noise_pct")
        # Hard clamp
        new_loss=$(awk  -v v="$new_loss"   'BEGIN{if(v<0.001)v=0.001; if(v>30)v=30;  printf "%.3f",v}')
        new_delay=$(awk -v v="$new_delay"  'BEGIN{if(v<1)v=1;          printf "%.1f",v}')
        new_jitter=$(awk -v v="$new_jitter" 'BEGIN{if(v<0)v=0; if(v>90)v=90; printf "%.1f",v}')
        _apply_direct "$new_loss" "$new_delay" "$new_jitter"
        sleep "$step_sleep"
    done

    _cur_loss="$to_loss"
    _cur_delay="$to_delay"
    _cur_jitter="$to_jitter"
}

# dwell: hold at ~current target but jitter every 8 s with ±noise
dwell_noisy() {
    local duration="$1" noise_pct="${2:-20}"
    local step=8
    local elapsed=0
    local rl dl jl

    while (( elapsed + step <= duration )); do
        rl=$(gaussian_noise "$_cur_loss"   "$noise_pct")
        dl=$(gaussian_noise "$_cur_delay"  "$(awk "BEGIN{printf \"%d\", $noise_pct - 5}")")
        jl=$(gaussian_noise "$_cur_jitter" "$noise_pct")
        rl=$(awk  -v v="$rl" 'BEGIN{if(v<0.001)v=0.001; if(v>30)v=30;  printf "%.3f",v}')
        dl=$(awk  -v v="$dl" 'BEGIN{if(v<1)v=1;          printf "%.1f",v}')
        jl=$(awk  -v v="$jl" 'BEGIN{if(v<0)v=0; if(v>90)v=90; printf "%.1f",v}')
        _apply_direct "$rl" "$dl" "$jl"
        sleep "$step"
        elapsed=$(( elapsed + step ))
    done

    local remaining=$(( duration - elapsed ))
    if (( remaining > 0 )); then sleep "$remaining"; fi
}

# ---------------------------------------------------------------------------
# Fuzzy ward scenario
#
# Same 10-phase cadence as run_natural_stress_baseline.sh but with:
#   • ±18-22% Gaussian noise per netem step         → blurry boundaries
#   • Shorter dwell "plateaus" with micro-excursions → realistic variance
#   • An extra transient sub-spike mid-critical      → real-world bursts
#   • Extended recovery oscillation                 → model learns drift-back
# ---------------------------------------------------------------------------
fuzzy_ward_scenario() {
    echo "[SCENARIO] Phase 1/11 — Morning stable (quiet ward)" | tee -a "${STRESS_LOG}"
    interpolate_noisy "0.3" "12" "2" 3 5 20
    dwell_noisy 65 22

    echo "[SCENARIO] Phase 2/11 — Gradual degradation (shift change + device join)" | tee -a "${STRESS_LOG}"
    interpolate_noisy "4.5" "65" "13" 10 4 18
    dwell_noisy 40 20

    echo "[SCENARIO] Phase 3/11 — Brief partial recovery (admin reset)" | tee -a "${STRESS_LOG}"
    interpolate_noisy "2.2" "38" "8" 5 4 22
    dwell_noisy 20 24

    echo "[SCENARIO] Phase 4/11 — Escalation to critical (busy hour + EMI)" | tee -a "${STRESS_LOG}"
    interpolate_noisy "13.0" "195" "52" 10 5 15
    dwell_noisy 30 18

    echo "[SCENARIO] Phase 5/11 — Sustained critical (peak congestion)" | tee -a "${STRESS_LOG}"
    interpolate_noisy "17.5" "255" "68" 4 4 12
    dwell_noisy 30 15

    # Mid-critical transient spike — unexpected device flood
    echo "[SCENARIO] Phase 5b/11 — Transient burst (device flood)" | tee -a "${STRESS_LOG}"
    interpolate_noisy "25.0" "320" "80" 3 3 10
    dwell_noisy 12 12
    interpolate_noisy "17.0" "240" "65" 3 3 15

    dwell_noisy 22 15

    echo "[SCENARIO] Phase 6/11 — Slow recovery to Unstable" | tee -a "${STRESS_LOG}"
    interpolate_noisy "4.2" "58" "12" 10 5 20
    dwell_noisy 20 22

    echo "[SCENARIO] Phase 7/11 — Stable recovery window" | tee -a "${STRESS_LOG}"
    interpolate_noisy "0.6" "18" "3.5" 6 4 22
    dwell_noisy 46 25

    echo "[SCENARIO] Phase 8/11 — Sharp interference burst (microwave / equipment)" | tee -a "${STRESS_LOG}"
    interpolate_noisy "19.5" "215" "70" 3 3 12
    dwell_noisy 21 14

    echo "[SCENARIO] Phase 9/11 — Fast recovery post-burst" | tee -a "${STRESS_LOG}"
    interpolate_noisy "1.2" "22" "4.5" 5 4 20
    dwell_noisy 20 22

    echo "[SCENARIO] Phase 10/11 — Stable-Unstable oscillation (nurses re-connecting)" | tee -a "${STRESS_LOG}"
    # Oscillate between stable and unstable edges to generate boundary samples
    interpolate_noisy "0.8" "28" "6"  4 4 25
    dwell_noisy 12 28
    interpolate_noisy "3.5" "42" "10" 4 4 25
    dwell_noisy 12 28
    interpolate_noisy "0.4" "18" "3"  4 4 25
    dwell_noisy 12 28

    echo "[SCENARIO] Phase 11/11 — Stable finish" | tee -a "${STRESS_LOG}"
    interpolate_noisy "0.2" "10" "2" 4 4 18
    dwell_noisy 40 20

    echo "[SCENARIO] Fuzzy ward scenario complete (~660 s)" | tee -a "${STRESS_LOG}"
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
# 1. Setup namespaces
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"

echo ""
echo "[PIPELINE] Setting up network namespaces..."
sudo "${SETUP_SCRIPT}"

# ---------------------------------------------------------------------------
# 2. Start receiver (writes network_telemetry.csv into RUN_DIR)
# ---------------------------------------------------------------------------
echo "[PIPELINE] Starting health_receiver.py in receiver_ns..."
umask 022
sudo env HEALTH_DATA_BASE_DIR="${RUN_DIR}" \
    HEALTH_NETWORK_MODEL_PATH="${PROJECT_DIR}/models/best_network_model.pkl" \
    ip netns exec receiver_ns \
    "${PYTHON_BIN}" -u "${PROJECT_DIR}/scripts/closed_loop/health_receiver.py" \
    > "${RECEIVER_LOG}" 2>&1 &
RECEIVER_PID=$!

sleep 2

# ---------------------------------------------------------------------------
# 3. Start static (RAW) senders — same 8 devices as the baseline test
# ---------------------------------------------------------------------------
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

echo "[PIPELINE] Starting ${#DEVICES[@]} static senders in sender_ns..."
for dev in "${DEVICES[@]}"; do
    read -r dev_id dev_type <<< "$dev"
    SENDER_LOG="${RUN_DIR}/sender_${dev_id}_${dev_type}.log"
    sudo env HEALTH_DATA_BASE_DIR="${RUN_DIR}" \
        ip netns exec sender_ns \
        "${PYTHON_BIN}" -u "${EVAL_DIR}/baseline_sender.py" \
        --device-id   "${dev_id}" \
        --device-type "${dev_type}" \
        --receiver-ip 10.0.0.2 \
        --base-dir    "${RUN_DIR}" \
        > "${SENDER_LOG}" 2>&1 &
    echo "$!" >> "${SENDER_PID_FILE}"
done

sleep 2

# ---------------------------------------------------------------------------
# 4. Run the fuzzy stress scenario (background)
# ---------------------------------------------------------------------------
echo "[PIPELINE] Launching fuzzy natural-stress scenario (~${DURATION}s)..."
fuzzy_ward_scenario &
STRESS_PID=$!

# ---------------------------------------------------------------------------
# 5. Wait for the full duration
# ---------------------------------------------------------------------------
sleep "${DURATION}"

echo ""
echo "[PIPELINE] ─────────────────────────────────────────────────"
echo "[PIPELINE] Data collection complete."
echo "[PIPELINE] Sending SIGINT to receiver to flush final window..."
kill -INT "${RECEIVER_PID}" 2>/dev/null || true
sleep 3   # give receiver time to flush

# cleanup trap fires here for senders + netem

# ---------------------------------------------------------------------------
# 6. Retrain using new integrated training script
# ---------------------------------------------------------------------------
echo ""
echo "[PIPELINE] ─────────────────────────────────────────────────"
echo "[PIPELINE] Starting retrain phase..."
echo "[PIPELINE] ─────────────────────────────────────────────────"

"${PYTHON_BIN}" "${DATAGEN_DIR}/train_from_pipeline.py" \
    --new-run-dir "${RUN_DIR}"

echo ""
echo "[PIPELINE] ════════════════════════════════════════════════"
echo "[PIPELINE] ALL DONE."
echo "[PIPELINE] Run directory : ${RUN_DIR}"
echo "[PIPELINE] Updated model : models/xgboost_network_model.pkl"
echo "[PIPELINE] ════════════════════════════════════════════════"
