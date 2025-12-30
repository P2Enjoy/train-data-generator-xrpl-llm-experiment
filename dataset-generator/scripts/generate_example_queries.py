"""Use GPT-OSS:120b (ollama) to propose natural language queries per schema."""

from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401
from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import load_jsonl, write_jsonl
from lib.llm import run_ollama
from lib.parsing import extract_json_array
from model_config import default_model


OLLAMA_MODEL = default_model()
DEFAULT_OPERATORS = ["equals", "not_equals", "like", "in"]


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

    parser = argparse.ArgumentParser(description="Generate example NL queries for schemas.", parents=[config_parser])
    parser.add_argument(
        "--schemas",
        type=Path,
        default=Path(defaults.get("final_schemas_out", "outputs/final_schemas.jsonl")),
        help="JSONL produced by build_schemas.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(defaults.get("schema_queries_out", "outputs/schema_queries.jsonl")),
        help="Where to write JSONL queries.",
    )
    parser.add_argument(
        "--per-schema",
        type=int,
        default=int(defaults.get("per_schema_queries", 6)),
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
        default=bool(defaults.get("offline_fallback", False)),
        help="Skip ollama and generate deterministic stubs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("seed", 19)),
        help="Seed for fallback generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(defaults.get("max_retries", -1)),
        help="Retries when Ollama output lacks parseable JSON. Use -1 for unlimited (default).",
    )
    return parser.parse_args(remaining)


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
    return run_ollama(prompt, model)


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
    max_retries = args.max_retries
    while True:
        try:
            raw = call_ollama(prompt, args.model)
            parsed = extract_json_array(raw)
            queries = [str(item) for item in parsed if isinstance(item, (str, int, float))]
            if len(queries) >= args.per_schema:
                return queries[: args.per_schema]
            print(f"[retry] {entry['schema_id']} returned {len(queries)}/{args.per_schema} queries, retrying teacher model")
        except ValueError as exc:
            print(f"[retry] {entry['schema_id']} parsing failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[retry] {entry['schema_id']} ollama error: {exc}")

        attempts += 1
        if max_retries >= 0 and attempts >= max_retries:
            raise RuntimeError(f"Ollama query generation failed for {entry['schema_id']} after {attempts} attempts")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    schemas = load_jsonl(args.schemas)
    total = len(schemas)
    print(f"[info] Generating example queries for {total} schemas from {args.schemas}")

    records = []
    for idx, entry in enumerate(schemas, start=1):
        print(f"[info] [{idx}/{total}] {entry['schema_id']} ({entry.get('domain')})")
        queries = generate_queries(entry, args, rng)
        print(f"[info] ✓ {entry['schema_id']}: generated {len(queries)} queries")
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
