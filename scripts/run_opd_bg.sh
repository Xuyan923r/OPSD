#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs

SESSION_TAG="${SESSION_TAG:-opd_science_qwen31b_from_qwen38b}"
TIMESTAMP="$(date +%F_%H_%M_%S)"
LOG_PATH="${ROOT_DIR}/logs/${SESSION_TAG}_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-xuyan923r-renmin-university-of-china}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"

if [[ "${PRESERVE_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export no_proxy="${no_proxy:-127.0.0.1,localhost}"
fi

nohup stdbuf -oL -eL bash "${ROOT_DIR}/scripts/run_opd.sh" > "${LOG_PATH}" 2>&1 &
PID=$!

echo "PID=${PID}"
echo "LOG=${LOG_PATH}"
