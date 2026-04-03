#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="/idfsdata/yexuyan/verl"

if [[ ! -d "${VERL_ROOT}" ]]; then
  echo "VERL repo not found at ${VERL_ROOT}" >&2
  exit 1
fi

if [[ ! -x "/idfsdata/yexuyan/conda_envs/verl/bin/python" ]]; then
  echo "Expected VERL python not found at /idfsdata/yexuyan/conda_envs/verl/bin/python" >&2
  exit 1
fi

cd "${VERL_ROOT}"
export PYTHON_BIN=/idfsdata/yexuyan/conda_envs/verl/bin/python

GPU_SET="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
GPU_TAG="${GPU_SET//,/}"
PROJECT_NAME="${PROJECT_NAME:-opsd_ifeval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opsd_ifeval_qwen3_1p7b_gpu${GPU_TAG}}"
LOG_FILE="${LOG_FILE:-${VERL_ROOT}/logs/${EXPERIMENT_NAME}.nohup.log}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate /idfsdata/yexuyan/conda_envs/verl >/dev/null 2>&1 || true
  fi
fi

nohup env \
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
  bash "${VERL_ROOT}/distillation/run_opsd.sh" \
    --teacher-model "${ROOT_DIR}/models/Qwen3-1.7B" \
    --student-model "${ROOT_DIR}/models/Qwen3-1.7B" \
    --train-files "${ROOT_DIR}/instruction_following_eval/data/splits/train.jsonl" \
    --val-files "${ROOT_DIR}/instruction_following_eval/data/splits/dev.jsonl" \
    --prompt-key prompt \
    --teacher-prompt-key teacher_prompt \
    --max-prompt-length 512 \
    --max-teacher-prompt-length 2048 \
    --max-response-length 1024 \
    --train-batch-size 32 \
    --micro-batch-size 1 \
    --gpus 4 \
    --logger console \
    --project-name "${PROJECT_NAME}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --resume-mode disable \
    --total-epochs 4 \
    --total-training-steps 48 \
    --save-freq 12 \
    --test-freq 6 \
    > "${LOG_FILE}" 2>&1 &

echo "Launched VERL OPSD IFEval training in background."
echo "Log file: ${LOG_FILE}"
