"""Use GPT-OSS:120b (ollama) to propose natural language queries per schema."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List

from model_config import default_model


OLLAMA_MODEL = default_model()
DEFAULT_OPERATORS = ["equals", "not_equals", "like", "in"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate example NL queries for schemas.")
    parser.add_argument(
        "--schemas",
        type=Path,
        default=Path("outputs/final_schemas.jsonl"),
        help="JSONL produced by build_schemas.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/schema_queries.jsonl"),
        help="Where to write JSONL queries.",
    )
    parser.add_argument(
        "--per-schema",
        type=int,
        default=6,
        help="How many queries to request per schema.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help="Ollama model name to call.",
    )
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        help="Skip ollama and generate deterministic stubs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=19,
        help="Seed for fallback generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries when Ollama output lacks parseable JSON.",
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


def extract_operator_consts(entry: Dict[str, Any]) -> List[str]:
    ops: List[str] = []
    raw_ops = entry.get("operators") or []
    if raw_ops:
        for op in raw_ops:
            if isinstance(op, dict) and "const" in op:
                ops.append(str(op["const"]))
            elif isinstance(op, str):
                ops.append(op)
    if not ops:
        try:
            one_of = (
                entry["schema"]["properties"]["steps"]["items"]["properties"]["conditions"]["items"]["properties"]["operator"]["oneOf"]
            )
            ops = [str(item["const"]) for item in one_of if isinstance(item, dict) and "const" in item]
        except Exception:  # noqa: BLE001
            ops = []
    return ops or DEFAULT_OPERATORS


def extract_json_array(text: str) -> List[str]:
    def clean_output(chunk: str) -> str:
        chunk = chunk.strip()
        chunk = re.sub(r"^```(?:json)?|```$", "", chunk, flags=re.MULTILINE).strip()
        return chunk

    def try_json_parse(chunk: str) -> Any:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return None

    cleaned = clean_output(text)
    for candidate in (cleaned, cleaned.splitlines()[0] if cleaned else ""):
        if not candidate:
            continue
        parsed = try_json_parse(candidate)
        if parsed is not None:
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("queries", "items", "responses"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]

    # fallback: locate first JSON array inside the text
    match = re.search(r"\[(?:[^\[\]]|\n)*\]", cleaned, flags=re.DOTALL)
    if match:
        parsed = try_json_parse(match.group(0))
        if isinstance(parsed, list):
            return parsed

    raise ValueError("No JSON array found in LLM output")


def build_prompt(entry: Dict[str, Any], count: int, operators: List[str]) -> str:
    fields = entry.get("fields", [])
    field_lines = "\n".join(f"- {field['const']}: {field.get('description', '')}" for field in fields)
    operator_lines = "\n".join(f"- {op}" for op in operators)
    return textwrap.dedent(
        f"""
        You are generating natural language user queries (EN or FR) for a funnel DSL.
        Schema id: {entry['schema_id']}
        Domain: {entry.get('domain')}
        Fields:
        {field_lines or '- None provided'}
        Operators:
        {operator_lines or '- None provided'}

        Produce exactly {count} short, diverse natural language queries that reference the above fields and rely on the operator names listed above.
        Return ONLY a JSON array of strings (no object, no extra keys).
        """
    ).strip()


def call_ollama(prompt: str, model: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama exited with {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def stub_queries(entry: Dict[str, Any], count: int, rng: random.Random) -> List[str]:
    fields = entry.get("fields", [])
    operators = extract_operator_consts(entry)
    if not fields:
        return [f"{entry.get('domain', 'domain').title()} sample query {i+1}" for i in range(count)]
    queries = []
    for i in range(count):
        field = rng.choice(fields)
        operator = rng.choice(operators)
        queries.append(f"{entry.get('domain', 'Domain')} where {field['const']} {operator} sample_value_{i+1}")
    return queries


def generate_queries(entry: Dict[str, Any], args: argparse.Namespace, rng: random.Random) -> List[str]:
    operators = extract_operator_consts(entry)
    if args.offline_fallback:
        return stub_queries(entry, args.per_schema, rng)
    prompt = build_prompt(entry, args.per_schema, operators)
    attempts = 0
    parsed: List[str] | None = None
    while parsed is None and attempts <= args.max_retries:
        try:
            raw = call_ollama(prompt, args.model)
            parsed = extract_json_array(raw)
        except ValueError as exc:
            attempts += 1
            if attempts > args.max_retries:
                print(f"[warn] ollama query generation failed for {entry['schema_id']}: {exc}")
                return stub_queries(entry, args.per_schema, rng)
            print(f"[retry] {entry['schema_id']} parsing failed, retrying ({attempts}/{args.max_retries})")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] ollama query generation failed for {entry['schema_id']}: {exc}")
            return stub_queries(entry, args.per_schema, rng)

    queries = [str(item) for item in parsed if isinstance(item, (str, int, float))]
    while len(queries) < args.per_schema:
        queries.append(f"{entry.get('domain', 'Domain')} fallback query {len(queries)+1}")
    return queries[: args.per_schema]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    schemas = load_jsonl(args.schemas)

    records = []
    for entry in schemas:
        queries = generate_queries(entry, args, rng)
        records.append(
            {
                "schema_id": entry["schema_id"],
                "domain": entry.get("domain"),
                "queries": queries,
            }
        )

    write_jsonl(args.out, records)
    print(f"Wrote queries for {len(records)} schemas to {args.out}")


if __name__ == "__main__":
    main()
