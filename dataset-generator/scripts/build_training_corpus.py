"""Convert validated dataset rows into prompt/target text pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401
from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import load_jsonl, write_jsonl


PROMPT_TEMPLATE = """You are a JSON AST generator.
You must output a single JSON object that satisfies the following JSON Schema
and represents the program matching the user request.

[SCHEMA]
{schema_json}
[/SCHEMA]

[QUERY]
{query}
[/QUERY]

[CURRENT_DATE]
{current_date}
[/CURRENT_DATE]

[OUTPUT]
"""


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to defaults JSON (config/defaults.json).",
    )
    config_args, remaining = config_parser.parse_known_args()
    defaults = load_section("dataset_generation", config_args.config)

    parser = argparse.ArgumentParser(description="Build training corpus JSONL.", parents=[config_parser])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(defaults.get("dataset_out", "outputs/d_03_dataset.jsonl")),
        help="JSONL produced by generate_dataset.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(defaults.get("training_corpus_out", "outputs/d_04_training_corpus.jsonl")),
        help="Where to write prompt/response JSONL.",
    )
    parser.add_argument(
        "--include-invalid",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("include_invalid", True)),
        help="Include invalid rows as negatives in the corpus.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on how many samples to export.",
    )
    return parser.parse_args(remaining)


def build_prompt(schema_json: str, query: str, current_date: str | None) -> str:
    return PROMPT_TEMPLATE.format(schema_json=schema_json, query=query, current_date=current_date or "N/A")


def main() -> None:
    args = parse_args()
    dataset = load_jsonl(args.dataset)
    total = len(dataset)
    print(
        f"[info] Building training corpus from {total} dataset rows (include_invalid={args.include_invalid}, max_samples={args.max_samples})"
    )
    rows: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset, start=1):
        if not args.include_invalid and not sample.get("is_valid", True):
            continue
        prompt_text = build_prompt(sample["schema_json"], sample["query"], sample.get("current_date"))
        rows.append(
            {
                "schema_id": sample["schema_id"],
                "domain": sample.get("domain"),
                "prompt": prompt_text,
                "completion": sample["ast_json"],
                "is_valid": sample.get("is_valid", True),
            }
        )
        if args.max_samples and len(rows) >= args.max_samples:
            print(f"[info] Reached max_samples={args.max_samples} at dataset row {idx}")
            break

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} training rows to {args.out}")


if __name__ == "__main__":
    main()
