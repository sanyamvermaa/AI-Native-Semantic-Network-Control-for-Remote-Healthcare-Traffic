#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_closed_loop_multi_sender_test.sh [duration_seconds]
# Example:
#   bash scripts/run_closed_loop_multi_sender_test.sh 120

DURATION="${1:-120}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
	PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "/home/ayhm23/miniconda3/bin/python3" ]]; then
	PYTHON_BIN="/home/ayhm23/miniconda3/bin/python3"
else
	PYTHON_BIN="python3"
fi
RECEIVER_IP="${RECEIVER_IP:-127.0.0.1}"
RECEIVER_PORT="${RECEIVER_PORT:-9000}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/outputs/results/closed_loop_test_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${HEALTH_DATA_BASE_DIR:-${PROJECT_DIR}/csv}"

mkdir -p "${RUN_DIR}"
mkdir -p "${DATA_DIR}"

RECEIVER_LOG="${RUN_DIR}/receiver.log"
WARD_LOG="${RUN_DIR}/ward_controller.log"
SENDER_PID_FILE="${RUN_DIR}/sender_pids.txt"

touch "${SENDER_PID_FILE}"

echo "[RUN] Project dir     : ${PROJECT_DIR}"
echo "[RUN] Duration (sec)  : ${DURATION}"
echo "[RUN] Python          : ${PYTHON_BIN}"
echo "[RUN] Output dir      : ${RUN_DIR}"
echo "[RUN] Data dir        : ${DATA_DIR}"

cleanup() {
	echo "[CLEANUP] Stopping sender processes..."
	if [[ -f "${SENDER_PID_FILE}" ]]; then
		while read -r spid; do
			if [[ -n "${spid}" ]] && kill -0 "${spid}" 2>/dev/null; then
				kill "${spid}" 2>/dev/null || true
			fi
		done < "${SENDER_PID_FILE}"
	fi

	echo "[CLEANUP] Stopping receiver process..."
	if [[ -n "${RECEIVER_PID:-}" ]] && kill -0 "${RECEIVER_PID}" 2>/dev/null; then
		kill "${RECEIVER_PID}" 2>/dev/null || true
	fi

	echo "[CLEANUP] Stopping ward controller..."
	if [[ -n "${WARD_PID:-}" ]] && kill -0 "${WARD_PID}" 2>/dev/null; then
		kill "${WARD_PID}" 2>/dev/null || true
	fi
}

trap cleanup EXIT INT TERM

cd "${PROJECT_DIR}"

echo "[RUN] Starting receiver..."
HEALTH_DATA_BASE_DIR="${DATA_DIR}" "${PYTHON_BIN}" -u health_receiver.py > "${RECEIVER_LOG}" 2>&1 &
RECEIVER_PID=$!
echo "[RUN] Receiver PID    : ${RECEIVER_PID}"

sleep 2

echo "[RUN] Starting ward controller..."
HEALTH_DATA_BASE_DIR="${DATA_DIR}" "${PYTHON_BIN}" -u ward_controller.py --base-dir "${DATA_DIR}" > "${WARD_LOG}" 2>&1 &
WARD_PID=$!
echo "[RUN] Ward PID        : ${WARD_PID}"

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

echo "[RUN] Starting senders..."
for dev in "${DEVICES[@]}"; do
	dev_id="$(awk '{print $1}' <<< "${dev}")"
	dev_type="$(awk '{print $2}' <<< "${dev}")"
	sender_log="${RUN_DIR}/sender_${dev_id}_${dev_type}.log"

	"${PYTHON_BIN}" -u health_sender.py \
		--device-id "${dev_id}" \
		--device-type "${dev_type}" \
		--receiver-ip "${RECEIVER_IP}" \
		--receiver-port "${RECEIVER_PORT}" \
		--base-dir "${DATA_DIR}" \
		> "${sender_log}" 2>&1 &

	spid=$!
	echo "${spid}" >> "${SENDER_PID_FILE}"
	echo "[RUN] Sender PID=${spid} id=${dev_id} type=${dev_type}"
done

echo "[RUN] Collecting data for ${DURATION}s..."
sleep "${DURATION}"

echo "[CHECK] Stopping processes to finalize logs..."
cleanup

sleep 1

CSV_DIR="${DATA_DIR}"
COMMAND_LOG="${CSV_DIR}/command_log.csv"

predict_count="$(grep -c "\[PREDICT\]" "${RECEIVER_LOG}" || true)"
latency_count="$(grep -c "\[LATENCY\]" "${RECEIVER_LOG}" || true)"
warn_count="$(grep -c "\[WARN\]" "${RECEIVER_LOG}" || true)"
ward_cmd_count="$(grep -c "\[WARD CMD\]" "${WARD_LOG}" || true)"

echo
echo "==== Closed-Loop Behavior Summary ===="
echo "Run directory        : ${RUN_DIR}"
echo "Receiver log         : ${RECEIVER_LOG}"
echo "Ward log             : ${WARD_LOG}"
echo "Receiver PREDICT cnt : ${predict_count}"
echo "Receiver LATENCY cnt : ${latency_count}"
echo "Receiver WARN cnt    : ${warn_count}"
echo "Ward command cnt     : ${ward_cmd_count}"

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
echo "Quick checks:"
echo "1) Ensure PREDICT count is > 0 (model inference active)."
echo "2) Ensure command_log.csv has rows (command channel active)."
echo "3) Inspect receiver log for repeated mode decisions and latency trends."

