"""Run a single schema+prompt pair through the teacher model for inference testing."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

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
        raise RuntimeError(f"ollama returned {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        chunk = text[idx:].lstrip()
        if not chunk:
            break
        try:
            obj, consumed = decoder.raw_decode(chunk)
            return obj
        except json.JSONDecodeError:
            idx += len(text[idx:]) - len(chunk) + 1
            continue
    raise ValueError("No JSON object found in LLM response")


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=2)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.schema_source)
    entry = next((rec for rec in records if rec.get("schema_id") == args.schema_id), None)
    if not entry:
        raise ValueError(f"{args.schema_id} not found in {args.schema_source}")
    schema_json = entry["schema_json"]
    prompt_text = PROMPT_TEMPLATE.format(
        schema_json=schema_json, query=args.prompt, current_date=args.current_date.isoformat()
    )

    print("Prompt sent to model:\n", prompt_text)
    response = call_model(prompt_text, args.model)
    print("\nLLM raw output:\n", response)
    try:
        parsed = extract_json(response)
    except ValueError as exc:
        print(f"[error] failed to parse LLM output: {exc}")
        return

    print("\nParsed AST:")
    print(canonical(parsed))


if __name__ == "__main__":
    main()
