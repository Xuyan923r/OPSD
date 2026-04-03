import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

from instruction_following_eval import evaluation_lib


def compute_instruction_following_metrics(outputs):
    prompt_total = len(outputs)
    prompt_correct = sum(1 for output in outputs if output.follow_all_instructions)
    instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
    instruction_correct = sum(sum(output.follow_instruction_list) for output in outputs)

    return {
        "prompt_level": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_level": instruction_correct / instruction_total if instruction_total else 0.0,
        "num_prompts": prompt_total,
        "num_instructions": instruction_total,
    }


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False))
            file_obj.write("\n")


def build_output_dir(args):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        gpu_tag = args.gpus.replace(",", "") if args.gpus else "auto"
        split_tag = args.split
        ckpt_tag = Path(args.checkpoint_dir).name if args.checkpoint_dir else Path(args.base_model).name
        output_dir = Path(args.project_root) / "logs" / f"ifeval_{split_tag}_{ckpt_tag}_gpu{gpu_tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_lora_rank(checkpoint_dir: str | None) -> int:
    if checkpoint_dir is None:
        return 64

    adapter_config_path = Path(checkpoint_dir) / "adapter_config.json"
    if not adapter_config_path.exists():
        return 64

    try:
        with open(adapter_config_path, "r", encoding="utf-8") as file_obj:
            adapter_config = json.load(file_obj)
        rank = int(adapter_config.get("r", 64))
        return max(rank, 1)
    except Exception:
        return 64


def force_vllm_cuda_platform_if_needed():
    import torch
    import vllm.platforms as vllm_platforms

    current_platform = getattr(vllm_platforms, "_current_platform", None)
    current_device_type = getattr(current_platform, "device_type", None)
    if current_device_type:
        return

    if not torch.cuda.is_available():
        return

    from vllm.platforms.cuda import CudaPlatform

    vllm_platforms._current_platform = CudaPlatform()
    print("Forced vLLM platform to CUDA because automatic platform detection returned empty.")


def load_vllm_model(args):
    print(f"Loading base model from: {args.base_model}")
    force_vllm_cuda_platform_if_needed()

    llm_config = {
        "model": args.base_model,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": True,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if args.checkpoint_dir is not None:
        adapter_safetensors = Path(args.checkpoint_dir) / "adapter_model.safetensors"
        adapter_bin = Path(args.checkpoint_dir) / "adapter_model.bin"
        if adapter_safetensors.exists() or adapter_bin.exists():
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = max(64, resolve_lora_rank(args.checkpoint_dir))
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
            print(f"LoRA adapter detected at: {args.checkpoint_dir}")
        else:
            raise FileNotFoundError(
                f"No LoRA weights found under {args.checkpoint_dir}. "
                "Expected adapter_model.safetensors or adapter_model.bin."
            )

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    return llm, tokenizer


def create_lora_request(checkpoint_dir: str | None):
    if checkpoint_dir is None:
        return None

    from vllm.lora.request import LoRARequest

    return LoRARequest("ifeval_checkpoint_lora", 1, checkpoint_dir)


def load_input_examples(input_data_path: str):
    examples = evaluation_lib.read_prompt_list(input_data_path)
    print(f"Loaded {len(examples)} IFEval prompts from: {input_data_path}")
    return examples


def build_generation_prompts(tokenizer, examples, enable_thinking: bool):
    prompts = []
    for example in examples:
        messages = [{"role": "user", "content": example.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt)
    return prompts


def generate_responses(llm, prompts, args, lora_request=None):
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
    )

    print("=" * 80)
    print("IFEVAL GENERATION CONFIGURATION")
    print("=" * 80)
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max num seqs: {args.max_num_seqs}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Seed: {args.seed}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Min-p: {args.min_p}")
    print(f"Presence penalty: {args.presence_penalty}")
    print(f"Enable thinking: {args.enable_thinking}")
    print(f"Using LoRA: {lora_request is not None}")
    print("=" * 80)

    start_time = time.time()
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    generation_seconds = time.time() - start_time

    responses = []
    total_generated_tokens = 0
    for output in outputs:
        completion = output.outputs[0]
        total_generated_tokens += len(completion.token_ids)
        responses.append(completion.text)

    avg_generated_tokens = total_generated_tokens / len(responses) if responses else 0.0
    print(
        f"Generation finished in {generation_seconds:.2f}s "
        f"for {len(responses)} prompts; avg completion length: {avg_generated_tokens:.1f} tokens"
    )
    return responses, generation_seconds


def score_responses(examples, responses):
    prompt_to_response = {example.prompt: response for example, response in zip(examples, responses)}
    strict_outputs = [
        evaluation_lib.test_instruction_following_strict(example, prompt_to_response) for example in examples
    ]
    loose_outputs = [
        evaluation_lib.test_instruction_following_loose(example, prompt_to_response) for example in examples
    ]
    strict_metrics = compute_instruction_following_metrics(strict_outputs)
    loose_metrics = compute_instruction_following_metrics(loose_outputs)
    return strict_outputs, loose_outputs, strict_metrics, loose_metrics


def append_summary(results_jsonl: str | None, summary: dict):
    if not results_jsonl:
        return

    results_path = Path(results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(summary, ensure_ascii=False))
        file_obj.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a base model or LoRA checkpoint on IFEval with vLLM.")
    parser.add_argument(
        "--project_root",
        type=str,
        default="/idfsdata/yexuyan/OPSD",
        help="Project root used to build default output paths.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base model used by the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Optional LoRA checkpoint directory containing adapter_model.safetensors.",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="IFEval jsonl file containing key/prompt/instruction_id_list/kwargs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Logical split name to record in the summary.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save responses and scoring artifacts.",
    )
    parser.add_argument(
        "--results_jsonl",
        type=str,
        default=None,
        help="Optional jsonl file to append a one-line run summary to.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional display name recorded in the summary. Defaults to base model name plus checkpoint tag.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU list for bookkeeping only, for example 0,1,2,3.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs used by vLLM tensor parallelism.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=32,
        help="Maximum number of concurrent sequences handled by vLLM.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.7,
        help="vLLM GPU memory utilization ratio.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Total max context length for vLLM.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3072,
        help="Maximum completion length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to vLLM for reproducible generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter; use -1 to disable.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Minimum probability threshold.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty for generation.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Whether to enable Qwen thinking mode when building the student prompt.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None and not Path(args.checkpoint_dir).exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint_dir}")

    output_dir = build_output_dir(args)
    llm, tokenizer = load_vllm_model(args)
    lora_request = create_lora_request(args.checkpoint_dir)
    examples = load_input_examples(args.input_data)
    prompts = build_generation_prompts(tokenizer, examples, enable_thinking=args.enable_thinking)
    responses, generation_seconds = generate_responses(llm, prompts, args, lora_request=lora_request)

    response_rows = [
        {"prompt": example.prompt, "response": response} for example, response in zip(examples, responses)
    ]
    responses_path = output_dir / "responses.jsonl"
    strict_results_path = output_dir / "eval_results_strict.jsonl"
    loose_results_path = output_dir / "eval_results_loose.jsonl"
    summary_path = output_dir / "summary.json"

    write_jsonl(responses_path, response_rows)

    strict_outputs, loose_outputs, strict_metrics, loose_metrics = score_responses(examples, responses)
    evaluation_lib.write_outputs(str(strict_results_path), strict_outputs)
    evaluation_lib.write_outputs(str(loose_results_path), loose_outputs)

    checkpoint_tag = Path(args.checkpoint_dir).name if args.checkpoint_dir else "base"
    default_model_name = (
        f"{Path(args.base_model).name}+{checkpoint_tag}" if args.checkpoint_dir else Path(args.base_model).name
    )

    summary = {
        "benchmark": "instruction_following_eval",
        "split": args.split,
        "model_name": args.model_name or default_model_name,
        "model_path": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset_path": args.input_data,
        "gpus": args.gpus or "",
        "num_prompts": strict_metrics["num_prompts"],
        "num_instructions": strict_metrics["num_instructions"],
        "strict_prompt_level": strict_metrics["prompt_level"],
        "strict_instruction_level": strict_metrics["instruction_level"],
        "loose_prompt_level": loose_metrics["prompt_level"],
        "loose_instruction_level": loose_metrics["instruction_level"],
        "generation_seconds": round(generation_seconds, 2),
        "run_dir": str(output_dir),
        "responses_path": str(responses_path),
        "strict_results_path": str(strict_results_path),
        "loose_results_path": str(loose_results_path),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "enable_thinking": args.enable_thinking,
    }

    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    append_summary(args.results_jsonl, summary)

    print("=" * 80)
    print("IFEVAL SCORES")
    print("=" * 80)
    print(f"strict_prompt_level: {summary['strict_prompt_level']:.6f}")
    print(f"strict_instruction_level: {summary['strict_instruction_level']:.6f}")
    print(f"loose_prompt_level: {summary['loose_prompt_level']:.6f}")
    print(f"loose_instruction_level: {summary['loose_instruction_level']:.6f}")
    print(f"responses saved to: {responses_path}")
    print(f"summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
