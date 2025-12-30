"""Quick sanity check: send prompt/target pairs from the training corpus to GPT-OSS:120b."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import List

import _bootstrap  # noqa: F401
from lib.io import canonical_json, load_jsonl
from lib.llm import run_ollama
from lib.parsing import extract_json_object
from model_config import default_model

DEFAULT_MODEL = default_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training prompts through the teacher LLM.")
    parser.add_argument(
        "--training-corpus",
        type=Path,
        default=Path("outputs/d_04_training_corpus.jsonl"),
        help="JSONL produced by build_training_corpus.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Ollama model to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2,
        help="How many samples to test from the corpus.",
    )
    return parser.parse_args()


def call_model(prompt: str, model: str) -> str:
    return run_ollama(prompt, model)


def extract_json(text: str) -> Any:
    return extract_json_object(text)


def print_diff(expected: str, actual: str) -> None:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    for line in diff:
        print(line)


def main() -> None:
    args = parse_args()
    if not args.training_corpus.exists():
        raise FileNotFoundError(f"{args.training_corpus} not found")
    records = load_jsonl(args.training_corpus)
    for idx, sample in enumerate(records[: args.max_samples], start=1):
        print(f"\nSample {idx}/{min(args.max_samples, len(records))}: {sample['schema_id']}")
        prompt_text = sample["prompt"]
        try:
            response = call_model(prompt_text, args.model)
        except RuntimeError as exc:
            print(f"[error] LLM request failed: {exc}")
            continue

        try:
            parsed = extract_json(response)
        except ValueError as exc:
            print(f"[error] could not parse response: {exc}")
            print(response)
            continue

        try:
            target = json.loads(sample["completion"])
        except json.JSONDecodeError as exc:
            print(f"[error] corpus completion invalid JSON: {exc}")
            continue

        print("LLM output:")
        print(canonical_json(parsed))
        print("Target completion:")
        print(canonical_json(target))

        if parsed == target:
            print("[ok] exact match")
        else:
            print("[warn] outputs differ")
            print_diff(canonical(target), canonical(parsed))


if __name__ == "__main__":
    main()
