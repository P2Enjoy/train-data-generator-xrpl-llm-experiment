"""Quick sanity check: send prompt/target pairs from the training corpus to GPT-OSS:120b."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_MODEL = "gpt-oss:120b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training prompts through the teacher LLM.")
    parser.add_argument(
        "--training-corpus",
        type=Path,
        default=Path("outputs/d_05_training_corpus.jsonl"),
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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def call_model(prompt: str, model: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama failed: {result.stderr.strip()}")
    return result.stdout.strip()


def extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in output")
    return json.loads(match.group(0))


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=2)


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
        print(canonical(parsed))
        print("Target completion:")
        print(canonical(target))

        if parsed == target:
            print("[ok] exact match")
        else:
            print("[warn] outputs differ")
            print_diff(canonical(target), canonical(parsed))


if __name__ == "__main__":
    main()
