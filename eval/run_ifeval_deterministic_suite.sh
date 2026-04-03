#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/idfsdata/yexuyan/OPSD"
PYTHON_BIN="${PYTHON_BIN:-/idfsdata/yexuyan/conda_envs/verl/bin/python}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/models/Qwen3-1.7B}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT_DIR/outputs/opsd_ifeval/qwen31b_ifeval_gpt4pi_gen3072_fixteacher}"
RESULTS_JSONL="${RESULTS_JSONL:-$ROOT_DIR/final_results.jsonl}"
INPUT_DATA="${INPUT_DATA:-$ROOT_DIR/instruction_following_eval/data/splits/test.jsonl}"
GPUS="${GPUS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
SEED="${SEED:-0}"

run_eval() {
    local model_name="$1"
    local checkpoint_dir="$2"

    echo "===================================================================="
    echo "Running IFEval for: ${model_name}"
    echo "Checkpoint: ${checkpoint_dir:-<base>}"
    echo "===================================================================="

    local args=(
        "$ROOT_DIR/eval/evaluate_ifeval.py"
        --project_root "$ROOT_DIR"
        --base_model "$BASE_MODEL"
        --input_data "$INPUT_DATA"
        --split test
        --results_jsonl "$RESULTS_JSONL"
        --model_name "$model_name"
        --gpus "$GPUS"
        --tensor_parallel_size "$TP_SIZE"
        --max_num_seqs "$MAX_NUM_SEQS"
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
        --max_model_len "$MAX_MODEL_LEN"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --top_k "$TOP_K"
        --min_p "$MIN_P"
        --presence_penalty "$PRESENCE_PENALTY"
        --seed "$SEED"
    )

    if [[ -n "$checkpoint_dir" ]]; then
        args+=(--checkpoint_dir "$checkpoint_dir")
    fi

    "$PYTHON_BIN" "${args[@]}"
}

run_eval "Qwen3-1.7B" ""

while IFS= read -r checkpoint_dir; do
    checkpoint_name="$(basename "$checkpoint_dir")"
    run_eval "qwen31b_ifeval_gpt4pi_gen3072_fixteacher_${checkpoint_name}" "$checkpoint_dir"
done < <(find "$EXPERIMENT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
