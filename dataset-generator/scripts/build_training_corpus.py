"""Convert validated dataset rows into prompt/target text pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
    parser = argparse.ArgumentParser(description="Build training corpus JSONL.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/d_04_dataset.jsonl"),
        help="JSONL produced by generate_dataset.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/d_05_training_corpus.jsonl"),
        help="Where to write prompt/response JSONL.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include invalid rows as negatives in the corpus.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on how many samples to export.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_prompt(schema_json: str, query: str, current_date: str | None) -> str:
    return PROMPT_TEMPLATE.format(schema_json=schema_json, query=query, current_date=current_date or "N/A")


def main() -> None:
    args = parse_args()
    dataset = load_jsonl(args.dataset)
    rows: List[Dict[str, Any]] = []

    for sample in dataset:
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
            break

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} training rows to {args.out}")


if __name__ == "__main__":
    main()
