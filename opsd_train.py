import os
import wandb
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
from opsd_trainer import OPSDTrainer
from data_collator import SelfDistillationDataCollator
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    dataset_name_or_path: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd",
        metadata={
            "help": "Dataset name on Hugging Face Hub or a local file path (json/jsonl/parquet/csv) for OPSD training."
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
            "help": "Dataset formatting mode. Use 'math' for problem/solution data, or "
            "'instruction' / 'prompt_teacher_prompt' / 'teacher_prompt' for prompt-based data."
        },
    )
    prompt_column: str = field(
        default="problem",
        metadata={"help": "Column name containing the student-visible prompt/problem."},
    )
    solution_column: str = field(
        default="solution",
        metadata={"help": "Column name containing the math reference solution when dataset_format=math."},
    )
    teacher_prompt_column: str = field(
        default="teacher_prompt",
        metadata={
            "help": "Column name containing the fully constructed privileged teacher prompt "
            "for instruction-style OPSD."
        },
    )
    teacher_reference_column: str | None = field(
        default=None,
        metadata={
            "help": "Optional column containing the privileged reference answer (for example gpt4_response). "
            "Used to build the teacher prompt when teacher_prompt_column is missing or empty."
        },
    )
    student_apply_chat_template: bool = field(
        default=True,
        metadata={"help": "Apply the tokenizer chat template to student prompts."},
    )
    teacher_apply_chat_template: bool = field(
        default=True,
        metadata={"help": "Apply the tokenizer chat template to teacher prompts."},
    )
    student_enable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to enable Qwen thinking mode for student prompts in chat templating."},
    )
    teacher_enable_thinking: bool = field(
        default=True,
        metadata={"help": "Whether to enable Qwen thinking mode for teacher prompts in chat templating."},
    )

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. "
            "With PEFT, the teacher disables LoRA adapters; with full fine-tuning, "
            "the trainer uses a frozen snapshot of the initial trainable weights."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )


def load_opsd_dataset(dataset_name_or_path: str, dataset_split: str):
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


def validate_opsd_columns(dataset, script_args: CustomScriptArguments, dataset_role: str = "dataset"):
    available_columns = set(dataset.column_names)
    required_columns = {script_args.prompt_column}

    if script_args.dataset_format == "math":
        required_columns.add(script_args.solution_column)
    else:
        has_teacher_prompt = (
            script_args.teacher_prompt_column is not None and script_args.teacher_prompt_column in available_columns
        )
        has_teacher_reference = (
            script_args.teacher_reference_column is not None
            and script_args.teacher_reference_column in available_columns
        )
        if not has_teacher_prompt and not has_teacher_reference:
            raise ValueError(
                f"Instruction-style OPSD requires either teacher_prompt_column or teacher_reference_column to exist "
                f"in the {dataset_role}. "
                f"Requested teacher_prompt_column='{script_args.teacher_prompt_column}', "
                f"teacher_reference_column='{script_args.teacher_reference_column}', "
                f"available columns={sorted(available_columns)}"
            )

    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_role.capitalize()} is missing required columns for dataset_format='{script_args.dataset_format}': "
            f"{missing_columns}. Available columns: {sorted(available_columns)}"
        )


def initialize_wandb(training_args, model_args, script_args, full_wandb_run_config, effective_batch_size, num_processes):
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return None

    init_kwargs = {
        "entity": training_args.wandb_entity,
        "project": training_args.wandb_project,
        "name": full_wandb_run_config,
        "config": {
            "model_name": model_args.model_name_or_path,
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
            "solution_column": script_args.solution_column,
            "teacher_prompt_column": script_args.teacher_prompt_column,
            "teacher_reference_column": script_args.teacher_reference_column,
            "use_tinker_loss": script_args.use_tinker_loss,
            "fixed_teacher": script_args.fixed_teacher,
            "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
            "use_ema_teacher": script_args.use_ema_teacher,
            "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
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


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    initialize_wandb(
        training_args=training_args,
        model_args=model_args,
        script_args=script_args,
        full_wandb_run_config=full_wandb_run_config,
        effective_batch_size=effective_batch_size,
        num_processes=num_processes,
    )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
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
    print(f"Loading model with dtype: {model_dtype}")
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
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    train_dataset = load_opsd_dataset(script_args.dataset_name_or_path, script_args.dataset_split)
    validate_opsd_columns(train_dataset, script_args, dataset_role="train dataset")
    eval_dataset = None
    if script_args.eval_dataset_name_or_path:
        eval_dataset = load_opsd_dataset(script_args.eval_dataset_name_or_path, script_args.eval_dataset_split)
        validate_opsd_columns(eval_dataset, script_args, dataset_role="eval dataset")

    print(f"\n{'='*80}")
    print("DATASET CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset source: {script_args.dataset_name_or_path}")
    print(f"Dataset split: {script_args.dataset_split}")
    print(f"Dataset format: {script_args.dataset_format}")
    print(f"Available columns: {train_dataset.column_names}")
    print(f"Prompt column: {script_args.prompt_column}")
    if script_args.dataset_format == 'math':
        print(f"Solution column: {script_args.solution_column}")
    else:
        print(f"Teacher prompt column: {script_args.teacher_prompt_column}")
        print(f"Teacher reference column: {script_args.teacher_reference_column}")
    print(f"Num training examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval dataset source: {script_args.eval_dataset_name_or_path}")
        print(f"Eval dataset split: {script_args.eval_dataset_split}")
        print(f"Num eval examples: {len(eval_dataset)}")
    print(f"{'='*80}\n")

    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = dict(training_args.dataset_kwargs or {})
    training_args.dataset_kwargs["skip_prepare_dataset"] = True

    data_collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        reason_first=script_args.reason_first,
        dataset_format=script_args.dataset_format,
        prompt_column=script_args.prompt_column,
        solution_column=script_args.solution_column,
        teacher_prompt_column=script_args.teacher_prompt_column,
        teacher_reference_column=script_args.teacher_reference_column,
        student_apply_chat_template=script_args.student_apply_chat_template,
        teacher_apply_chat_template=script_args.teacher_apply_chat_template,
        student_enable_thinking=script_args.student_enable_thinking,
        teacher_enable_thinking=script_args.teacher_enable_thinking,
    )

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
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
