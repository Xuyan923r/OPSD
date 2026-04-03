import torch


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    INSTRUCTION_DATASET_FORMATS = {"instruction", "prompt_teacher_prompt", "teacher_prompt"}

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        dataset_format="math",
        prompt_column="problem",
        solution_column="solution",
        teacher_prompt_column="teacher_prompt",
        teacher_reference_column=None,
        student_apply_chat_template=True,
        teacher_apply_chat_template=True,
        student_enable_thinking=False,
        teacher_enable_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.dataset_format = dataset_format
        self.prompt_column = prompt_column
        self.solution_column = solution_column
        self.teacher_prompt_column = teacher_prompt_column
        self.teacher_reference_column = teacher_reference_column
        self.student_apply_chat_template = student_apply_chat_template
        self.teacher_apply_chat_template = teacher_apply_chat_template
        self.student_enable_thinking = student_enable_thinking
        self.teacher_enable_thinking = teacher_enable_thinking

        # Prompt for reasoning about the solution before teaching
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )
        self.reference_answer_intro = (
            "\n\nBelow is a high-quality reference answer to the same request. "
            "Use it as guidance, but write your own answer to the original request rather than copying it verbatim. "
            "Make sure your final answer follows all constraints in the original prompt.\n\nReference answer:\n"
        )
        self.reference_answer_outro = (
            "\n\nNow provide your own final answer to the original request. "
            "Do not mention the reference answer."
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Dataset format: {self.dataset_format}")

        if self.dataset_format in self.INSTRUCTION_DATASET_FORMATS and self.reason_first:
            raise ValueError(
                "reason_first=True is only implemented for the math dataset format. "
                "For instruction-following data, please set --reason_first False."
            )

    def _get_required_text(self, feature, key):
        if key not in feature:
            raise KeyError(f"Missing required column '{key}' in feature. Available keys: {list(feature.keys())}")
        value = feature[key]
        if not isinstance(value, str):
            raise TypeError(f"Column '{key}' must be a string, got {type(value)}")
        return value

    def _apply_chat_template(self, prompt_text, enable_thinking):
        messages = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def _build_instruction_teacher_prompt(self, prompt_text, reference_text):
        return (
            f"{prompt_text}"
            f"{self.reference_answer_intro}"
            f"{reference_text}"
            f"{self.reference_answer_outro}"
        )

    def _build_student_prompt(self, feature):
        if self.dataset_format == "math":
            problem = self._get_required_text(feature, self.prompt_column)
            prompt_text = (
                f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            )
        elif self.dataset_format in self.INSTRUCTION_DATASET_FORMATS:
            prompt_text = self._get_required_text(feature, self.prompt_column)
        else:
            raise ValueError(
                f"Unsupported dataset_format='{self.dataset_format}'. "
                "Expected 'math' or one of {'instruction', 'prompt_teacher_prompt', 'teacher_prompt'}."
            )

        if self.student_apply_chat_template:
            return self._apply_chat_template(prompt_text, enable_thinking=self.student_enable_thinking)
        return prompt_text

    def _build_math_reasoning_prompt(self, feature):
        problem = self._get_required_text(feature, self.prompt_column)
        solution = self._get_required_text(feature, self.solution_column)
        reasoning_user_message = (
            f"Problem: {problem}\n\n"
            f"Here is a correct reasoning to this problem:"
            f"=== Reference Reasoning Start ===\n"
            f"{solution}\n"
            f"=== Reference Reasoning End ===\n\n"
            f"{self.reason_first_prompt}"
        )
        return self._apply_chat_template(reasoning_user_message, enable_thinking=self.teacher_enable_thinking)

    def _build_teacher_prompt_text(self, feature):
        if self.dataset_format == "math":
            problem = self._get_required_text(feature, self.prompt_column)
            solution = self._get_required_text(feature, self.solution_column)
            return (
                f"Problem: {problem}\n\n"
                f"Here is a reference solution to this problem:\n"
                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                f"{self.transition_prompt}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}."
            )

        if self.dataset_format in self.INSTRUCTION_DATASET_FORMATS:
            if self.teacher_prompt_column:
                teacher_prompt = feature.get(self.teacher_prompt_column)
                if isinstance(teacher_prompt, str) and teacher_prompt.strip():
                    return teacher_prompt

            if self.teacher_reference_column:
                prompt_text = self._get_required_text(feature, self.prompt_column)
                reference_text = self._get_required_text(feature, self.teacher_reference_column)
                return self._build_instruction_teacher_prompt(prompt_text, reference_text)

            raise KeyError(
                "Instruction-style OPSD requires either a non-empty teacher prompt column "
                f"('{self.teacher_prompt_column}') or a teacher reference column "
                f"('{self.teacher_reference_column}') to build the privileged teacher prompt."
            )

        raise ValueError(
            f"Unsupported dataset_format='{self.dataset_format}'. "
            "Expected 'math' or one of {'instruction', 'prompt_teacher_prompt', 'teacher_prompt'}."
        )

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            student_prompt = self._build_student_prompt(feature)
            student_prompts.append(student_prompt)

            if self.dataset_format == "math" and self.reason_first:
                reasoning_prompt = self._build_math_reasoning_prompt(feature)
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                teacher_prompt_text = self._build_teacher_prompt_text(feature)
                if self.teacher_apply_chat_template:
                    teacher_prompt = self._apply_chat_template(
                        teacher_prompt_text, enable_thinking=self.teacher_enable_thinking
                    )
                else:
                    teacher_prompt = teacher_prompt_text
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.dataset_format == "math" and self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
