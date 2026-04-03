#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate || true)}"

USE_PEFT="${USE_PEFT:-1}"
FIXED_TEACHER="${FIXED_TEACHER:-1}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/idfsdata/yexuyan/OPSD/models/Qwen3-1.7B}"
TRAIN_DATASET="${TRAIN_DATASET:-/idfsdata/yexuyan/OPSD/instruction_following_eval/data/splits/train.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-/idfsdata/yexuyan/OPSD/instruction_following_eval/data/splits/dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/idfsdata/yexuyan/OPSD/outputs/opsd_ifeval/}"
RUN_CONFIG_DEFAULT="qwen31b_ifeval_gpt4pi_gen3072_fixteacher"
if [[ "${USE_PEFT}" != "1" ]]; then
  RUN_CONFIG_DEFAULT="${RUN_CONFIG_DEFAULT}_fullft"
fi
RUN_CONFIG="${RUN_CONFIG:-${RUN_CONFIG_DEFAULT}}"

LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-15}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-3072}"
EVAL_STEPS="${EVAL_STEPS:-6}"
SAVE_STEPS="${SAVE_STEPS:-25}"
LOGGING_STEPS="${LOGGING_STEPS:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
LMBDA="${LMBDA:-1}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.05}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
REPORT_TO="${REPORT_TO:-swanlab}"
WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No python interpreter found in the current shell." >&2
  echo "Please activate the training environment from environment.yml first." >&2
  exit 1
fi

if [[ -z "${ACCELERATE_BIN}" ]]; then
  echo "No accelerate executable found in the current shell." >&2
  echo "Please activate the training environment from environment.yml first." >&2
  exit 1
fi

REQUIRED_MODULES=(trl deepspeed vllm datasets wandb)
if [[ "${USE_PEFT}" == "1" ]]; then
  REQUIRED_MODULES+=(peft)
fi

MISSING_MODULES="$(
  REQUIRED_MODULES_JOINED="${REQUIRED_MODULES[*]}" "${PYTHON_BIN}" - <<'PY'
import importlib.util
import os

required = os.environ["REQUIRED_MODULES_JOINED"].split()
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)"

if [[ -n "${MISSING_MODULES}" ]]; then
  cat >&2 <<EOF
Current python environment is missing required training packages: ${MISSING_MODULES}

This TRL-based launcher expects the environment described in:
  ${ROOT_DIR}/environment.yml

Your current shell can use the already-working VERL backend instead:
  bash ${ROOT_DIR}/scripts/run_opsd_ifeval_verl.sh

Or create/activate a dedicated OPSD env, then rerun this script.
EOF
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"

read -r -a LORA_TARGET_MODULES_ARR <<< "${LORA_TARGET_MODULES}"

cmd=(
  "${ACCELERATE_BIN}" launch
  --config_file accelerate.yaml
  --num_processes 4
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --main_process_port 12951
  opsd_train.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --dataset_name_or_path "${TRAIN_DATASET}"
  --dataset_split train
  --eval_dataset_name_or_path "${EVAL_DATASET}"
  --eval_dataset_split dev
  --dataset_format prompt_teacher_prompt
  --prompt_column prompt
  --teacher_prompt_column teacher_prompt
  --teacher_reference_column gpt4_response
  --learning_rate "${LEARNING_RATE}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_checkpointing
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --run_config "${RUN_CONFIG}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --eval_strategy steps
  --eval_steps "${EVAL_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --report_to "${REPORT_TO}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --torch_dtype "${TORCH_DTYPE}"
  --max_length "${MAX_LENGTH}"
  --beta 0
  --use_vllm
  --vllm_mode colocate
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
  --lmbda "${LMBDA}"
  --jsd_token_clip "${JSD_TOKEN_CLIP}"
  --wandb_project "${WANDB_PROJECT}"
)

if [[ "${USE_PEFT}" == "1" ]]; then
  cmd+=(
    --use_peft
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_target_modules "${LORA_TARGET_MODULES_ARR[@]}"
  )
fi

if [[ "${FIXED_TEACHER}" == "1" ]]; then
  cmd+=(--fixed_teacher)
fi

"${cmd[@]}"
