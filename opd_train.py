import os
import wandb
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from opd_data_collator import SharedPromptDataCollator
from opd_trainer import OPDTrainer

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Script arguments for naive OPD."""

    dataset_name_or_path: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd",
        metadata={
            "help": "Dataset name on Hugging Face Hub or a local file path (json/jsonl/parquet/csv) for OPD training."
        },
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use when loading a Hub dataset or naming a local file split."},
    )
    eval_dataset_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional evaluation dataset on Hugging Face Hub or a local file path "
            "(json/jsonl/parquet/csv) for periodic validation."
        },
    )
    eval_dataset_split: str = field(
        default="validation",
        metadata={"help": "Dataset split to use when loading the evaluation dataset."},
    )
    dataset_format: str = field(
        default="math",
        metadata={
            "help": "Dataset formatting mode. Use 'math' for problem data, or "
            "'instruction' / 'prompt_teacher_prompt' / 'teacher_prompt' for prompt-based data."
        },
    )
    prompt_column: str = field(
        default="problem",
        metadata={"help": "Column name containing the prompt/problem shown to both student and teacher."},
    )
    apply_chat_template: bool = field(
        default=True,
        metadata={"help": "Apply the tokenizer chat template to the shared prompt."},
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to enable Qwen thinking mode for the shared prompt."},
    )
    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of full-vocab JSD loss."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty passed to generation."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. "
            "Set to 0 to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. Set to 0 for no clipping."
        },
    )


def load_opd_dataset(dataset_name_or_path: str, dataset_split: str):
    dataset_path = Path(dataset_name_or_path)

    if dataset_path.exists():
        if dataset_path.is_dir():
            for extension, dataset_type in (
                (".jsonl", "json"),
                (".json", "json"),
                (".parquet", "parquet"),
                (".csv", "csv"),
            ):
                candidate = dataset_path / f"{dataset_split}{extension}"
                if candidate.exists():
                    dataset = load_dataset(dataset_type, data_files={dataset_split: str(candidate)})
                    return dataset[dataset_split]
            raise ValueError(
                f"Could not find a supported local dataset file inside '{dataset_name_or_path}' "
                f"for split '{dataset_split}'. Expected {dataset_split}.jsonl/json/parquet/csv."
            )

        suffix = dataset_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            dataset = load_dataset("json", data_files={dataset_split: str(dataset_path)})
        elif suffix == ".parquet":
            dataset = load_dataset("parquet", data_files={dataset_split: str(dataset_path)})
        elif suffix == ".csv":
            dataset = load_dataset("csv", data_files={dataset_split: str(dataset_path)})
        else:
            raise ValueError(
                f"Unsupported local dataset file '{dataset_name_or_path}'. "
                "Supported extensions: .json, .jsonl, .parquet, .csv."
            )
        return dataset[dataset_split]

    dataset = load_dataset(dataset_name_or_path)
    return dataset[dataset_split]


def validate_opd_columns(dataset, script_args: CustomScriptArguments, dataset_role: str = "dataset"):
    available_columns = set(dataset.column_names)
    required_columns = {script_args.prompt_column}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_role.capitalize()} is missing required columns for dataset_format='{script_args.dataset_format}': "
            f"{missing_columns}. Available columns: {sorted(available_columns)}"
        )


def initialize_wandb(training_args, model_args, script_args, full_wandb_run_config, effective_batch_size, num_processes):
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return None

    init_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "180"))
    init_kwargs = {
        "entity": training_args.wandb_entity,
        "project": training_args.wandb_project,
        "name": full_wandb_run_config,
        "settings": wandb.Settings(init_timeout=init_timeout),
        "config": {
            "student_model_name": model_args.model_name_or_path,
            "teacher_model_name": training_args.teacher_model_name_or_path,
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "num_train_epochs": training_args.num_train_epochs,
            "max_completion_length": training_args.max_completion_length,
            "temperature": training_args.temperature,
            "beta": training_args.beta,
            "lmbda": training_args.lmbda,
            "max_length": training_args.max_length,
            "use_peft": model_args.use_peft,
            "finetune_mode": "lora" if model_args.use_peft else "full",
            "lora_r": model_args.lora_r if model_args.use_peft else None,
            "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "num_processes": num_processes,
            "dataset_name_or_path": script_args.dataset_name_or_path,
            "dataset_split": script_args.dataset_split,
            "dataset_format": script_args.dataset_format,
            "prompt_column": script_args.prompt_column,
            "use_tinker_loss": script_args.use_tinker_loss,
            "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        },
    }

    wandb_disabled = os.environ.get("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    wandb_mode = os.environ.get("WANDB_MODE", "").strip().lower()

    if wandb_disabled:
        init_kwargs["mode"] = "disabled"
        print("WANDB_DISABLED is set. WandB logging is disabled.")
    elif wandb_mode:
        init_kwargs["mode"] = wandb_mode
        print(f"WandB mode set from environment: {wandb_mode}")
    elif not os.environ.get("WANDB_API_KEY"):
        init_kwargs["mode"] = "offline"
        print("WANDB_API_KEY is not set. Falling back to offline WandB logging.")

    try:
        return wandb.init(**init_kwargs)
    except wandb.errors.UsageError as exc:
        error_message = str(exc)
        if "No API key configured" in error_message and init_kwargs.get("mode") not in {"offline", "disabled"}:
            print("WandB login is unavailable. Retrying in offline mode.")
            init_kwargs["mode"] = "offline"
            os.environ["WANDB_MODE"] = "offline"
            return wandb.init(**init_kwargs)
        raise
    except wandb.errors.CommError:
        if init_kwargs.get("mode") not in {"offline", "disabled"}:
            print("WandB online init failed. Retrying in offline mode so training can continue.")
            init_kwargs["mode"] = "offline"
            os.environ["WANDB_MODE"] = "offline"
            return wandb.init(**init_kwargs)
        raise


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if not training_args.teacher_model_name_or_path:
        raise ValueError(
            "Naive OPD requires `--teacher_model_name_or_path`. "
            "The student model is still provided via `--model_name_or_path`."
        )

    if (
        training_args.teacher_tokenizer_name_or_path is not None
        and training_args.teacher_tokenizer_name_or_path != model_args.model_name_or_path
    ):
        raise ValueError(
            "This naive OPD implementation assumes teacher and student share the same tokenizer/vocabulary. "
            "Please omit `--teacher_tokenizer_name_or_path` or set it to the student tokenizer."
        )

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        student_model_name = model_args.model_name_or_path.split("/")[-1]
        teacher_model_name = training_args.teacher_model_name_or_path.split("/")[-1]
        full_wandb_run_config = (
            f"opd_{student_model_name}_from_{teacher_model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

    print(f"\n{'='*80}")
    print("RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Student Model: {model_args.model_name_or_path}")
    print(f"Teacher Model: {training_args.teacher_model_name_or_path}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    initialize_wandb(
        training_args=training_args,
        model_args=model_args,
        script_args=script_args,
        full_wandb_run_config=full_wandb_run_config,
        effective_batch_size=effective_batch_size,
        num_processes=num_processes,
    )

    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading student model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(training_args.teacher_model_init_kwargs or {})
    teacher_model_kwargs.setdefault("trust_remote_code", model_args.trust_remote_code)
    teacher_model_kwargs.setdefault("attn_implementation", model_args.attn_implementation or "flash_attention_2")
    teacher_model_kwargs.setdefault("torch_dtype", model_dtype)
    teacher_model_kwargs.setdefault("use_cache", True)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_opd_dataset(script_args.dataset_name_or_path, script_args.dataset_split)
    validate_opd_columns(train_dataset, script_args, dataset_role="train dataset")
    eval_dataset = None
    if script_args.eval_dataset_name_or_path:
        eval_dataset = load_opd_dataset(script_args.eval_dataset_name_or_path, script_args.eval_dataset_split)
        validate_opd_columns(eval_dataset, script_args, dataset_role="eval dataset")

    print(f"\n{'='*80}")
    print("DATASET CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset source: {script_args.dataset_name_or_path}")
    print(f"Dataset split: {script_args.dataset_split}")
    print(f"Dataset format: {script_args.dataset_format}")
    print(f"Available columns: {train_dataset.column_names}")
    print(f"Prompt column: {script_args.prompt_column}")
    print(f"Num training examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval dataset source: {script_args.eval_dataset_name_or_path}")
        print(f"Eval dataset split: {script_args.eval_dataset_split}")
        print(f"Num eval examples: {len(eval_dataset)}")
    print(f"{'='*80}\n")

    print(
        "Naive OPD assumption: the teacher forward pass reuses the student tokenizer and token ids. "
        "Please use teacher/student models with compatible vocabularies."
    )

    training_args.presence_penalty = script_args.presence_penalty
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = dict(training_args.dataset_kwargs or {})
    training_args.dataset_kwargs["skip_prepare_dataset"] = True

    data_collator = SharedPromptDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        dataset_format=script_args.dataset_format,
        prompt_column=script_args.prompt_column,
        apply_chat_template=script_args.apply_chat_template,
        enable_thinking=script_args.enable_thinking,
    )

    trainer = OPDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
