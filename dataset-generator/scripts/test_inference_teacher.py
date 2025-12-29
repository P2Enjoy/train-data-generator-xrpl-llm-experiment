"""Run a single schema+prompt pair through the teacher model for inference testing."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401
from lib.io import canonical_json, load_jsonl
from lib.llm import run_ollama
from lib.parsing import extract_json_object
from model_config import default_model

DEFAULT_MODEL = default_model()


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
    parser = argparse.ArgumentParser(description="Test teacher inference for one schema/prompt.")
    parser.add_argument("--schema-id", type=str, required=True, help="Schema identifier to load.")
    parser.add_argument("--prompt", type=str, required=True, help="Natural language prompt.")
    parser.add_argument(
        "--schema-source",
        type=Path,
        default=Path("outputs/d_02_final_schemas.jsonl"),
        help="JSONL file listing built schemas.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to query.")
    parser.add_argument(
        "--current-date",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="Current date (ISO) for prompt context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.schema_source.exists():
        raise FileNotFoundError(f"{args.schema_source} not found")
    records = load_jsonl(args.schema_source)
    entry = next((rec for rec in records if rec.get("schema_id") == args.schema_id), None)
    if not entry:
        raise ValueError(f"{args.schema_id} not found in {args.schema_source}")
    schema_json = entry["schema_json"]
    prompt_text = PROMPT_TEMPLATE.format(
        schema_json=schema_json, query=args.prompt, current_date=args.current_date.isoformat()
    )

    print("Prompt sent to model:\n", prompt_text)
    response = run_ollama(prompt_text, args.model)
    print("\nLLM raw output:\n", response)
    try:
        parsed = extract_json_object(response)
    except ValueError as exc:
        print(f"[error] failed to parse LLM output: {exc}")
        return

    print("\nParsed AST:")
    print(canonical_json(parsed))


if __name__ == "__main__":
    main()
