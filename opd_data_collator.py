import torch


class SharedPromptDataCollator:
    """
    Data collator for naive OPD.

    Both student and teacher receive the exact same prompt text. The student then
    generates on-policy continuations, and the teacher scores the same generated
    suffix under the same prompt prefix.
    """

    INSTRUCTION_DATASET_FORMATS = {"instruction", "prompt_teacher_prompt", "teacher_prompt"}

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        dataset_format="math",
        prompt_column="problem",
        apply_chat_template=True,
        enable_thinking=False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_format = dataset_format
        self.prompt_column = prompt_column
        self.apply_chat_template = apply_chat_template
        self.enable_thinking = enable_thinking

        print(f"[OPDDataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[OPDDataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[OPDDataCollator] Dataset format: {self.dataset_format}")
        print(f"[OPDDataCollator] Apply chat template: {self.apply_chat_template}")
        print(f"[OPDDataCollator] Enable thinking: {self.enable_thinking}")

    def _get_required_text(self, feature, key):
        if key not in feature:
            raise KeyError(f"Missing required column '{key}' in feature. Available keys: {list(feature.keys())}")
        value = feature[key]
        if not isinstance(value, str):
            raise TypeError(f"Column '{key}' must be a string, got {type(value)}")
        return value

    def _apply_chat_template(self, prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def _build_prompt(self, feature):
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

        if self.apply_chat_template:
            return self._apply_chat_template(prompt_text)
        return prompt_text

    def __call__(self, features):
        prompts = [self._build_prompt(feature) for feature in features]

        encoded_no_pad = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(prompt_lengths)

        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )

        prompt_ids = encoded["input_ids"]
        prompt_attention_mask = encoded["attention_mask"]
        prompt_lengths_tensor = torch.tensor(prompt_lengths)

        return {
            "student_prompts": prompt_ids,
            "student_prompt_attention_mask": prompt_attention_mask,
            "student_prompt_length": max_prompt_len,
            "student_prompt_lengths_per_example": prompt_lengths_tensor,
            "teacher_prompts": prompt_ids.clone(),
            "teacher_prompt_attention_mask": prompt_attention_mask.clone(),
            "teacher_prompt_length": max_prompt_len,
            "teacher_prompt_lengths_per_example": prompt_lengths_tensor.clone(),
        }
