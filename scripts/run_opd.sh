#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source /home/yexuyan/miniconda3/etc/profile.d/conda.sh
conda activate /idfsdata/yexuyan/conda_envs/verl

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/idfsdata/yexuyan/OPSD/models/Qwen3-1.7B}"
TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL_NAME_OR_PATH:-/idfsdata/yexuyan/OPSD/models/Qwen3-8B}"
TRAIN_DATASET="${TRAIN_DATASET:-/idfsdata/yexuyan/OPSD/data/Mixture-of-Thoughts/science_opd.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-/idfsdata/yexuyan/OPSD/MMLU-Pro-dev40-opd.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/idfsdata/yexuyan/OPSD/outputs/opd/science_fullft/}"

for required_path in \
    "${MODEL_NAME_OR_PATH}" \
    "${TEACHER_MODEL_NAME_OR_PATH}" \
    "${TRAIN_DATASET}" \
    "${EVAL_DATASET}"; do
    if [[ "${required_path}" = /* && ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --gradient_accumulation_steps 24 \
    --main_process_port 12952 \
    opd_train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --teacher_model_name_or_path "${TEACHER_MODEL_NAME_OR_PATH}" \
    --dataset_name_or_path "${TRAIN_DATASET}" \
    --dataset_split train \
    --eval_dataset_name_or_path "${EVAL_DATASET}" \
    --eval_dataset_split validation \
    --dataset_format instruction \
    --prompt_column prompt \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 24 \
    --output_dir "${OUTPUT_DIR}" \
    --run_config qwen31b_from_qwen38b_science_bs96_fullft_gen1024_temp11_clip005 \
    --num_train_epochs 1 \
    --max_completion_length 1024 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 50 \
    --logging_steps 2 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --jsd_token_clip 0.05 \
    --wandb_entity xuyan923r-renmin-university-of-china \
    --wandb_project OPSD
