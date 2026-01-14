"""Run a DPO/ORPO-style preference fine-tune using teacher vs student outputs."""

from __future__ import annotations

import argparse
import builtins
from pathlib import Path
from typing import Any, Dict

import _bootstrap  # noqa: F401
import unsloth  # noqa: F401  # ensure patches apply before other ML deps
import torch
from datasets import load_dataset
from lib.config import DEFAULT_CONFIG_PATH, load_section
from peft import PeftModel
from trl import DPOConfig, DPOTrainer
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

SYSTEM_PROMPT = (
    "You are a JSON AST generator. You must output a single JSON object that satisfies the"
    " provided JSON Schema and represents the program matching the user request."
    " If the request cannot be satisfied with the available fields or operators, output a refusal"
    " object with a reason and a suggestion (per the schema)."
)
GEMMA3_INSTRUCTION_PART = "<start_of_turn>user\n"
GEMMA3_RESPONSE_PART = "<start_of_turn>model\n"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to defaults JSON (config/defaults.json).",
    )
    config_args, remaining = config_parser.parse_known_args()
    defaults = load_section("alignment", config_args.config)
    logging_default = int(defaults.get("logging_steps", 10))
    save_steps_default = int(defaults.get("save_steps", logging_default * 10))
    save_total_limit_default = int(defaults.get("save_total_limit", 3))

    required_keys = ("pairs_out", "adapter", "output_dir")
    missing = [key for key in required_keys if not defaults.get(key)]
    if missing:
        raise SystemExit(f"Missing alignment config keys in config/defaults.json: {', '.join(missing)}")

    parser = argparse.ArgumentParser(description="Preference fine-tune with DPO/ORPO.", parents=[config_parser])
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path(defaults["pairs_out"]),
        help="Preference pairs produced by build_alignment_pairs.py.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(defaults.get("base_model", "unsloth/gemma-3-270m-it")),
        help="Base HF model id.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(defaults["adapter"]),
        help="Starting LoRA checkpoint from SFT stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(defaults["output_dir"]),
        help="Where to write the aligned adapters.",
    )
    parser.add_argument("--max-seq-length", type=int, default=int(defaults.get("max_seq_length", 2048)), help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 2)), help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=int(defaults.get("grad_accum", 8)), help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=float(defaults.get("learning_rate", 5e-6)), help="Learning rate for LoRA params.")
    parser.add_argument("--warmup-steps", type=int, default=int(defaults.get("warmup_steps", 50)), help="Warmup steps.")
    parser.add_argument("--max-steps", type=int, default=int(defaults.get("max_steps", 500)), help="Total training steps.")
    parser.add_argument("--logging-steps", type=int, default=logging_default, help="Log frequency.")
    parser.add_argument(
        "--save-steps",
        type=int,
        default=save_steps_default,
        help="Checkpoint save frequency (steps).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=save_total_limit_default,
        help="Maximum checkpoints to keep (older ones are removed).",
    )
    parser.add_argument("--beta", type=float, default=float(defaults.get("beta", 0.1)), help="DPO beta.")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("load_in_4bit", True)),
        help="Load model in 4bit for efficiency.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("bf16", True)),
        help="Use bfloat16 if available.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from a previous DPO checkpoint directory.",
    )
    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args


def load_pairs(path: Path, tokenizer, max_seq_length: int):
    dataset = load_dataset("json", data_files=str(path), split="train")

    def format_row(row: Dict[str, Any]) -> Dict[str, str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text = prompt_text.removeprefix("<bos>")

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_token_len = len(prompt_tokens)

        def clip_completion(text: str) -> str:
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            if prompt_token_len + len(toks) > max_seq_length:
                keep = max(max_seq_length - prompt_token_len, 1)
                return tokenizer.decode(toks[:keep], skip_special_tokens=False)
            return text

        chosen = clip_completion(row["chosen"])
        rejected = clip_completion(row["rejected"])

        # cheap truncation safeguard to reduce empty-label cases
        for key in ("chosen", "rejected"):
            toks = tokenizer(chosen if key == "chosen" else rejected, add_special_tokens=False)["input_ids"]
            if len(toks) > max_seq_length:
                truncated = tokenizer.decode(toks[:max_seq_length], skip_special_tokens=False)
                if key == "chosen":
                    chosen = truncated
                else:
                    rejected = truncated

        return {
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
        }

    return dataset.map(format_row, remove_columns=dataset.column_names)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid names that shadow existing module attributes (e.g., ".train").
    train_adapter_name = "dpo_train"
    ref_adapter_name = "dpo_reference"

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    if not hasattr(builtins, "VARIANT_KWARG_KEYS"):
        builtins.VARIANT_KWARG_KEYS = {"adapter_name"}
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name=train_adapter_name, is_trainable=True)
    model.load_adapter(args.adapter, adapter_name=ref_adapter_name, is_trainable=False)
    model.set_adapter(train_adapter_name)

    dataset = load_pairs(args.pairs, tokenizer, args.max_seq_length)
    print(f"[dpo] pairs={len(dataset)} beta={args.beta} 4bit={args.load_in_4bit}")
    if args.resume_from_checkpoint:
        print(f"[dpo] resuming from checkpoint: {args.resume_from_checkpoint}")

    training_args = DPOConfig(
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        output_dir=str(args.output_dir),
        report_to="none",
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=not args.bf16 and torch.cuda.is_available(),
        model_adapter_name=train_adapter_name,
        ref_adapter_name=ref_adapter_name,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    final_dir = args.output_dir / "checkpoint-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[dpo] finished; adapters saved to {final_dir}")


if __name__ == "__main__":
    main()
