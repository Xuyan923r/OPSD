from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset

from trl.models import prepare_deepspeed
from trl.trainer.utils import disable_dropout_in_model, empty_cache

from opsd_trainer import OPSDTrainer


class OPDTrainer(OPSDTrainer):
    _tag_names = ["trl", "opd"]
    _name = "OPD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str | None = None,
        args=None,
        data_collator=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics=None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
    ):
        if teacher_model is None:
            teacher_model = getattr(args, "teacher_model_name_or_path", None)
        if teacher_model is None:
            raise ValueError(
                "Naive OPD requires an explicit teacher model. Please pass `teacher_model=` or "
                "`--teacher_model_name_or_path`."
            )

        if getattr(args, "teacher_model_init_kwargs", None) is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs in the config, but the teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = dict(args.teacher_model_init_kwargs)
            if "torch_dtype" in teacher_model_init_kwargs:
                teacher_torch_dtype = teacher_model_init_kwargs["torch_dtype"]
                if isinstance(teacher_torch_dtype, str):
                    teacher_model_init_kwargs["torch_dtype"] = (
                        teacher_torch_dtype if teacher_torch_dtype in ["auto", None] else getattr(torch, teacher_torch_dtype)
                    )
                else:
                    teacher_model_init_kwargs["torch_dtype"] = teacher_torch_dtype

        if isinstance(teacher_model, str):
            init_kwargs = dict(teacher_model_init_kwargs)
            init_kwargs.pop("device_map", None)
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model,
                **init_kwargs,
            )

        if getattr(args, "disable_dropout", False):
            disable_dropout_in_model(teacher_model)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            use_thinking_machines_loss=use_thinking_machines_loss,
            fixed_teacher=False,
            reason_first=False,
            top_k_loss=top_k_loss,
            jsd_token_clip=jsd_token_clip,
            use_ema_teacher=False,
            ema_decay=0.999,
        )

        teacher_model.resize_token_embeddings(self.model.config.vocab_size)
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.teacher_model_name_or_path = (
            teacher_model.config._name_or_path if hasattr(teacher_model, "config") else str(teacher_model)
        )

        print(f"\n{'='*80}")
        print("NAIVE OPD MODE ENABLED")
        print(f"Student model: {self.model_name_or_path}")
        print(f"Teacher model: {self.teacher_model_name_or_path}")
        print("Teacher and student use the same prompt; the teacher is a separate frozen model.")
        print(f"{'='*80}\n")

    def _teacher_forward_context(self, model):
        return nullcontext()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the naive OPD loss.

        Student and teacher share the same prompt prefix, but the teacher is a
        separate frozen model instead of a privileged view of the student.
        """
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        if self.use_thinking_machines_loss:
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs
        else:
            student_logits_for_loss = student_logits
            del student_logits

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            del outputs_teacher
            empty_cache()

        if not self.use_thinking_machines_loss:
            student_log_probs_sampled = torch.gather(
                F.log_softmax(student_logits_for_loss / self.temperature, dim=-1),
                dim=-1,
                index=sampled_token_ids.unsqueeze(-1),
            ).squeeze(-1)
            teacher_log_probs_sampled = torch.gather(
                F.log_softmax(teacher_logits_for_loss / self.temperature, dim=-1),
                dim=-1,
                index=sampled_token_ids.unsqueeze(-1),
            ).squeeze(-1)

        token_mask = shifted_labels != -100 if shifted_labels is not None else torch.ones_like(
            sampled_token_ids, dtype=torch.bool
        )
        token_mask_float = token_mask.to(student_log_probs_sampled.dtype)
        valid_token_counts = token_mask_float.sum(dim=-1).clamp_min(1.0)
        teacher_reward_per_example = (teacher_log_probs_sampled * token_mask_float).sum(dim=-1) / valid_token_counts
        student_score_per_example = (student_log_probs_sampled * token_mask_float).sum(dim=-1) / valid_token_counts
        advantage_per_example = teacher_reward_per_example - student_score_per_example
        self._last_rollout_scores = [
            {
                "teacher_reward": float(teacher_reward_per_example[i].detach().cpu().item()),
                "student_score": float(student_score_per_example[i].detach().cpu().item()),
                "reward_advantage": float(advantage_per_example[i].detach().cpu().item()),
                "num_completion_tokens": int(valid_token_counts[i].detach().cpu().item()),
            }
            for i in range(teacher_reward_per_example.shape[0])
        ]

        if self.use_thinking_machines_loss:
            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            loss = -(advantage * student_log_probs_sampled_masked).mean()
            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
        else:
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
            del student_logits_for_loss, teacher_logits_for_loss
            del student_log_probs_sampled, teacher_log_probs_sampled

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        return loss
