#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

SESSION_TAG="${SESSION_TAG:-opd_science_qwen31b_from_qwen38b_wait4gpu}"
TIMESTAMP="$(date +%F_%H_%M_%S)"
WAIT_LOG="${ROOT_DIR}/logs/${SESSION_TAG}_${TIMESTAMP}.log"

GPU_MEMORY_THRESHOLD_MB="${GPU_MEMORY_THRESHOLD_MB:-5000}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-10}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-30}"

pick_four_gpus() {
  local selected=()
  while IFS=',' read -r idx mem_used util; do
    idx="$(echo "${idx}" | xargs)"
    mem_used="$(echo "${mem_used}" | xargs)"
    util="$(echo "${util}" | xargs)"
    if [[ "${mem_used}" -le "${GPU_MEMORY_THRESHOLD_MB}" && "${util}" -le "${GPU_UTIL_THRESHOLD}" ]]; then
      selected+=("${idx}")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

  if [[ "${#selected[@]}" -ge 4 ]]; then
    printf "%s,%s,%s,%s" "${selected[0]}" "${selected[1]}" "${selected[2]}" "${selected[3]}"
  fi
}

nohup bash -lc '
  set -euo pipefail
  cd "'"${ROOT_DIR}"'"

  echo "Watcher started at $(date -Is)"
  echo "Waiting for 4 GPUs with memory <= '"${GPU_MEMORY_THRESHOLD_MB}"' MB and util <= '"${GPU_UTIL_THRESHOLD}"'%"

  while true; do
    GPUS="$('"$(declare -f pick_four_gpus)"'; pick_four_gpus)"
    if [[ -n "${GPUS}" ]]; then
      echo "Found available GPUs at $(date -Is): ${GPUS}"
      CUDA_VISIBLE_DEVICES="${GPUS}" WANDB_MODE=online WANDB_ENTITY=xuyan923r-renmin-university-of-china WANDB_PROJECT=OPSD \
        bash "'"${ROOT_DIR}"'/scripts/run_opd_bg.sh"
      exit 0
    fi

    echo "Still waiting at $(date -Is)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep '"${POLL_INTERVAL_SECONDS}"'
  done
' > "${WAIT_LOG}" 2>&1 &

WATCHER_PID=$!
echo "WATCHER_PID=${WATCHER_PID}"
echo "WATCHER_LOG=${WAIT_LOG}"
