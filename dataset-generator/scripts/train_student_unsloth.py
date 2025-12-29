"""Train Gemma 3 270M student adapters with Unsloth on the generated corpus."""
from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path
from typing import Any, Dict, Tuple

import _bootstrap  # noqa: F401
import torch
from datasets import load_dataset
from unsloth import FastModel
from lib.config import DEFAULT_CONFIG_PATH, load_section


# Patch Unsloth-generated PEFT forward to avoid missing constants in some builds.
def ensure_variant_keys_patch() -> None:
    cache_file = Path(__file__).resolve().parent.parent / "unsloth_compiled_cache" / "Linear_peft_forward.py"
    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8")
        marker = "VARIANT_KWARG_KEYS"
        if marker not in text:
            text = text.replace(
                "from peft.tuners.lora.torchao import (Any, torch)\n",
                "from peft.tuners.lora.torchao import (Any, torch)\nVARIANT_KWARG_KEYS = ('alora_offsets',)\n",
                1,
            )
            cache_file.write_text(text, encoding="utf-8")

    try:  # noqa: WPS501
        import unsloth_compiled_cache.Linear_peft_forward as _lpf  # type: ignore

        importlib.reload(_lpf)
        _lpf.VARIANT_KWARG_KEYS = getattr(_lpf, "VARIANT_KWARG_KEYS", ("alora_offsets",))
        _lpf.unsloth_forward.__globals__["VARIANT_KWARG_KEYS"] = _lpf.VARIANT_KWARG_KEYS
    except Exception:  # noqa: BLE001
        pass


ensure_variant_keys_patch()
try:  # noqa: WPS501
    import unsloth_compiled_cache.Linear_peft_forward as _lpf

    if not hasattr(_lpf, "VARIANT_KWARG_KEYS"):
        _lpf.VARIANT_KWARG_KEYS = ("alora_offsets",)
    # Ensure the forward function sees the constant even if the file is regenerated.
    _lpf.unsloth_forward.__globals__["VARIANT_KWARG_KEYS"] = _lpf.VARIANT_KWARG_KEYS
except Exception:  # noqa: BLE001
    pass
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback, TrainerControl, TrainerState

SYSTEM_PROMPT = (
    "You are a JSON AST generator. You must output a single JSON object that satisfies the"
    " provided JSON Schema and represents the program matching the user request."
)

DEFAULT_BASE_MODEL = "unsloth/gemma-3-270m-it"
DEFAULT_OUTPUT_DIR = Path("outputs/student_runs/gemma3-270m")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to defaults JSON (config/defaults.json).",
    )
    config_args, remaining = config_parser.parse_known_args()
    defaults = load_section("training", config_args.config)
    data_defaults = load_section("dataset_generation", config_args.config)

    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 270M with Unsloth LoRA adapters.", parents=[config_parser])
    parser.add_argument(
        "--training-corpus",
        type=Path,
        default=Path(
            defaults.get(
                "training_corpus",
                data_defaults.get("training_corpus_out", "outputs/d_05_training_corpus.jsonl"),
            )
        ),
        help="JSONL produced by build_training_corpus.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(defaults.get("output_dir", DEFAULT_OUTPUT_DIR)),
        help="Where to store checkpoints, logs, and final adapters.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(defaults.get("base_model", DEFAULT_BASE_MODEL)),
        help="HF model id for Gemma 3 270M.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=int(defaults.get("max_seq_length", 2048)),
        help="Maximum sequence length for training.",
    )
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 4)), help="Per-device batch size.")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=int(defaults.get("grad_accum", 4)),
        help="Gradient accumulation steps to reach effective batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(defaults.get("learning_rate", 5e-5)),
        help="Learning rate for adapter params.",
    )
    parser.add_argument("--weight-decay", type=float, default=float(defaults.get("weight_decay", 0.01)), help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=int(defaults.get("warmup_steps", 50)), help="Linear warmup steps.")
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=float(defaults.get("num_epochs", 1.0)),
        help="Number of training epochs if --max-steps is not set.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=defaults.get("max_steps", None),
        help="Optional cap on total training steps (overrides epochs when set).",
    )
    parser.add_argument("--logging-steps", type=int, default=int(defaults.get("logging_steps", 10)), help="Log frequency.")
    parser.add_argument("--eval-steps", type=int, default=int(defaults.get("eval_steps", 50)), help="Eval frequency when a val split exists.")
    parser.add_argument("--save-steps", type=int, default=int(defaults.get("save_steps", 200)), help="Checkpoint frequency.")
    parser.add_argument("--save-total-limit", type=int, default=int(defaults.get("save_total_limit", 3)), help="How many checkpoints to keep.")
    parser.add_argument("--val-split", type=float, default=float(defaults.get("val_split", 0.05)), help="Validation split fraction.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=defaults.get("max_samples", None),
        help="Optional cap on total samples.",
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 3407)), help="Random seed.")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("load_in_4bit", True)),
        help="Enable 4bit quantized training (default: on).",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("bf16", True)),
        help="Enable bf16 training if supported (default: on to match Gemma weights).",
    )
    parser.add_argument(
        "--use-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("use_gradient_checkpointing", False)),
        help="Turn on gradient checkpointing.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from a previous checkpoint directory.",
    )
    parser.add_argument("--lora-r", type=int, default=int(defaults.get("lora_r", 64)), help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=int(defaults.get("lora_alpha", 128)), help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=float(defaults.get("lora_dropout", 0.05)), help="LoRA dropout.")
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=int(defaults.get("eval_max_samples", 256)),
        help="Limit validation set to this many samples to keep eval quick.",
    )
    return parser.parse_args(remaining)


def load_corpus(
    path: Path,
    tokenizer,
    val_split: float,
    seed: int,
    max_samples: int | None,
    eval_max_samples: int,
) -> Tuple[Any, Any]:
    dataset = load_dataset("json", data_files=str(path), split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_row(sample: Dict[str, Any]) -> Dict[str, str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text.removeprefix("<bos>")}

    if val_split > 0:
        split = dataset.train_test_split(test_size=val_split, seed=seed)
        train_set = split["train"]
        eval_set = split["test"]
        if eval_max_samples and len(eval_set) > eval_max_samples:
            eval_set = eval_set.select(range(eval_max_samples))
    else:
        train_set = dataset
        eval_set = None

    train_set = train_set.map(format_row, remove_columns=train_set.column_names)
    if eval_set is not None:
        eval_set = eval_set.map(format_row, remove_columns=eval_set.column_names)
    return train_set, eval_set


class JsonlLogger(TrainerCallback):
    """Persist trainer logs to JSONL for later plotting/reporting."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: SFTConfig,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        payload = {"step": int(state.global_step), "epoch": state.epoch}
        payload.update({k: float(v) for k, v in logs.items()})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def save_run_config(path: Path, args: argparse.Namespace, train_len: int, eval_len: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({"train_samples": train_len, "eval_samples": eval_len or 0})
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        use_gradient_checkpointing="unsloth" if args.use_gradient_checkpointing else False,
    )

    train_set, eval_set = load_corpus(
        args.training_corpus,
        tokenizer,
        args.val_split,
        args.seed,
        args.max_samples,
        args.eval_max_samples,
    )
    print(
        f"[data] train samples={len(train_set)}"
        + (f" eval samples={len(eval_set)}" if eval_set is not None else "")
    )

    training_args = SFTConfig(
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_set is not None else "no",
        eval_steps=args.eval_steps if eval_set is not None else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=not args.bf16 and torch.cuda.is_available(),
        output_dir=str(args.output_dir),
        gradient_checkpointing=args.use_gradient_checkpointing,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=training_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    effective_bs = args.batch_size * args.grad_accum
    print(
        f"[config] base_model={args.base_model} 4bit={args.load_in_4bit} bf16={args.bf16} "
        f"lr={args.learning_rate} max_steps={args.max_steps or 'auto'} "
        f"batch_per_device={args.batch_size} grad_accum={args.grad_accum} effective_bs={effective_bs}"
    )

    metrics_log = args.output_dir / "training_metrics.jsonl"
    trainer.add_callback(JsonlLogger(metrics_log))
    save_run_config(args.output_dir / "run_config.json", args, len(train_set), len(eval_set) if eval_set else None)

    # Re-apply variant kwarg patch in case Unsloth regenerated its cache after imports.
    ensure_variant_keys_patch()

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"[train] finished steps={trainer.state.global_step} loss={trainer.state.log_history[-1].get('loss') if trainer.state.log_history else 'n/a'}")

    final_dir = args.output_dir / "checkpoint-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    (args.output_dir / "train_stats.json").write_text(
        json.dumps(trainer_stats.metrics, indent=2), encoding="utf-8"
    )

    print(f"Training complete. Final adapters saved to {final_dir}")


if __name__ == "__main__":
    main()
# Patch Unsloth-generated PEFT forward to avoid missing constants in some builds.
def ensure_variant_keys_patch() -> None:
    cache_file = Path(__file__).resolve().parent.parent / "unsloth_compiled_cache" / "Linear_peft_forward.py"
    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8")
        marker = "VARIANT_KWARG_KEYS"
        if marker not in text:
            text = text.replace(
                "from peft.tuners.lora.torchao import (Any, torch)\n",
                "from peft.tuners.lora.torchao import (Any, torch)\nVARIANT_KWARG_KEYS = ('alora_offsets',)\n",
                1,
            )
            cache_file.write_text(text, encoding="utf-8")

    try:  # noqa: WPS501
        import unsloth_compiled_cache.Linear_peft_forward as _lpf  # type: ignore

        importlib.reload(_lpf)
        _lpf.VARIANT_KWARG_KEYS = getattr(_lpf, "VARIANT_KWARG_KEYS", ("alora_offsets",))
        _lpf.unsloth_forward.__globals__["VARIANT_KWARG_KEYS"] = _lpf.VARIANT_KWARG_KEYS
    except Exception:  # noqa: BLE001
        pass


ensure_variant_keys_patch()
