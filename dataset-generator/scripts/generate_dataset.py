"""Generate dataset rows by asking the teacher to produce ASTs for provided queries."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401
from jsonschema import Draft7Validator

from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import canonical_json, load_jsonl
from lib.llm import run_ollama
from lib.parsing import extract_json_object

PROMPT_TEMPLATE = """You are a JSON AST generator.
You must output a single JSON object that satisfies the following JSON Schema
and represents the program matching the user request.
If the request cannot be satisfied with the available fields or operators, output a refusal object with a reason and a suggestion (per the schema).
Do not invent fields or operators.

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

    required = ("final_schemas_out", "dataset_out")
    missing = [key for key in required if not defaults.get(key)]
    if missing:
        raise SystemExit(f"Missing dataset_generation config keys in config/defaults.json: {', '.join(missing)}")

    parser = argparse.ArgumentParser(description="Generate dataset JSONL using the teacher model.", parents=[config_parser])
    parser.add_argument(
        "--schemas",
        type=Path,
        default=Path(defaults["final_schemas_out"]),
        help="JSONL produced by build_schemas.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(defaults["dataset_out"]),
        help="Where to write dataset JSONL.",
    )
    parser.add_argument(
        "--samples-per-schema",
        type=int,
        default=int(defaults.get("samples_per_schema", 20)),
        help="How many teacher-generated samples to keep per schema.",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=str(defaults.get("teacher_model") or ""),
        help="Teacher model id to call via Ollama (required).",
    )
    parser.add_argument(
        "--teacher-retries",
        type=int,
        default=int(defaults.get("teacher_retries", 3)),
        help="Retries when the teacher output cannot be parsed as JSON (-1 for infinite).",
    )
    parser.add_argument(
        "--teacher-retry-wait",
        type=float,
        default=float(defaults.get("teacher_retry_wait", 1.0)),
        help="Seconds to wait between teacher retries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("seed", 11)),
        help="Seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(defaults["checkpoint_path"]) if defaults.get("checkpoint_path") else None,
        help="Where to store progress for resuming generation. Disabled if not set.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(defaults.get("checkpoint_every", 1)),
        help="Save checkpoint every N schemas (requires --checkpoint-path).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume dataset generation from an existing checkpoint.",
    )
    parser.add_argument(
        "--current-date",
        type=str,
        default=date.today().isoformat(),
        help="Current date to include in prompts (default: today).",
    )
    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args


def build_prompt(schema_json: str, query: str, current_date: str) -> str:
    return PROMPT_TEMPLATE.format(schema_json=schema_json, query=query, current_date=current_date or "N/A")


def call_teacher_json(prompt: str, model: str, retries: int, wait: float) -> Dict[str, Any]:
    last_error: Exception | None = None
    attempt = 0
    max_label = "∞" if retries < 0 else str(max(retries, 1))
    while True:
        attempt += 1
        try:
            answer = run_ollama(prompt, model)
            return extract_json_object(answer)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[warn][teacher] attempt {attempt}/{max_label} failed: {exc}")
            if retries >= 0 and attempt >= max(retries, 1):
                break
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def serialize_rng_state(state: tuple[Any, ...]) -> List[Any]:
    version, inner, gaussian = state
    return [version, list(inner), gaussian]


def deserialize_rng_state(state: List[Any]) -> tuple[Any, ...]:
    version, inner, gaussian = state
    return (version, tuple(inner), gaussian)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to read checkpoint at {path}: {exc}")


def save_checkpoint(
    path: Path,
    next_schema_index: int,
    rng: random.Random,
    current_date: str,
    dataset_out: Path,
    schemas_path: Path,
    rows_written: int,
    total_schemas: int,
) -> None:
    payload = {
        "next_schema_index": next_schema_index,
        "rng_state": serialize_rng_state(rng.getstate()),
        "current_date": current_date,
        "dataset_out": str(dataset_out),
        "schemas_path": str(schemas_path),
        "rows_written": rows_written,
        "total_schemas": total_schemas,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_queries(entry: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for raw in entry.get("example_queries") or []:
        if isinstance(raw, (str, int, float)):
            value = str(raw).strip()
            if value:
                queries.append(value)
    return queries


def main() -> None:
    args = parse_args()
    if not args.teacher_model:
        raise SystemExit("Dataset generation now requires a teacher model (--teacher-model or dataset_generation.teacher_model).")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1 when checkpointing is enabled.")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = args.out.parent / "dataset_generation_checkpoint.json"
    elif args.out != Path(load_section("dataset_generation", args.config).get("dataset_out", args.out)):
        # If output was overridden, keep checkpoint alongside the custom output.
        checkpoint_path = args.out.parent / "dataset_generation_checkpoint.json"
    args.checkpoint_path = checkpoint_path

    rng = random.Random(args.seed)
    schemas = load_jsonl(args.schemas)
    total = len(schemas)
    print(f"[info] Building teacher-generated dataset from {total} schemas in {args.schemas}")

    start_index = 1
    rows_written = 0
    current_date = args.current_date

    if args.resume:
        if not args.checkpoint_path or not args.checkpoint_path.exists():
            raise SystemExit(f"Checkpoint file not found: {args.checkpoint_path}")
        checkpoint = load_checkpoint(args.checkpoint_path)
        start_index = int(checkpoint.get("next_schema_index", 1))
        rows_written = int(checkpoint.get("rows_written", 0))
        checkpoint_date = checkpoint.get("current_date")
        if checkpoint_date:
            current_date = checkpoint_date
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            rng.setstate(deserialize_rng_state(rng_state))
        checkpoint_schemas = checkpoint.get("schemas_path")
        if checkpoint_schemas and Path(checkpoint_schemas) != args.schemas:
            print(f"[warn] Checkpoint was created with schemas at {checkpoint_schemas}; current --schemas is {args.schemas}")
        checkpoint_out = checkpoint.get("dataset_out")
        if checkpoint_out and Path(checkpoint_out) != args.out:
            print(f"[warn] Checkpoint expects output at {checkpoint_out}; current --out is {args.out}")
        if not args.out.exists():
            raise SystemExit(f"Expected dataset output at {args.out} when resuming, but it does not exist.")
        print(f"[resume] Resuming from schema {start_index}/{total} (rows written so far: {rows_written})")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.unlink(missing_ok=True)
        if args.checkpoint_path:
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if args.checkpoint_path.exists():
                print(f"[info] removing existing checkpoint at {args.checkpoint_path} (pass --resume to continue it).")
            args.checkpoint_path.unlink(missing_ok=True)

    if start_index > total:
        print(f"[info] Checkpoint covers all {total} schemas; nothing to do.")
        if args.checkpoint_path:
            args.checkpoint_path.unlink(missing_ok=True)
        print(f"Wrote {rows_written} rows to {args.out}")
        return

    with args.out.open("a", encoding="utf-8") as handle:
        for idx, entry in enumerate(schemas, start=1):
            checkpoint_due = bool(args.checkpoint_path and (idx % args.checkpoint_every == 0 or idx == total))
            if idx < start_index:
                continue
            print(f"[info] [{idx}/{total}] {entry.get('schema_id')}")

            queries = clean_queries(entry)
            if not queries:
                print(f"[warn] skipping {entry.get('schema_id')} because no example_queries were provided")
                if checkpoint_due:
                    save_checkpoint(
                        args.checkpoint_path,
                        next_schema_index=idx + 1,
                        rng=rng,
                        current_date=current_date,
                        dataset_out=args.out,
                        schemas_path=args.schemas,
                        rows_written=rows_written,
                        total_schemas=total,
                    )
                continue

            rng.shuffle(queries)
            validator = Draft7Validator(entry["schema"])
            records: List[Dict[str, Any]] = []
            attempts = 0
            max_attempts = max(args.samples_per_schema * 3, len(queries))

            while len(records) < args.samples_per_schema and attempts < max_attempts:
                query = queries[attempts % len(queries)]
                prompt = build_prompt(entry["schema_json"], query, current_date)
                attempts += 1

                try:
                    teacher_ast = call_teacher_json(prompt, args.teacher_model, args.teacher_retries, args.teacher_retry_wait)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] teacher call failed for query='{query[:80]}': {exc}")
                    continue

                error = next(validator.iter_errors(teacher_ast), None)
                if error:
                    print(f"[warn] teacher output invalid for query='{query[:80]}': {error.message}")
                    continue

                records.append(
                    {
                        "schema_id": entry["schema_id"],
                        "domain": entry.get("domain"),
                        "operators": entry.get("operators"),
                        "schema_json": entry["schema_json"],
                        "query": query,
                        "current_date": current_date,
                        "ast_json": canonical_json(teacher_ast),
                        "is_valid": True,
                        "validation_error": None,
                        "error_type": None,
                        "teacher_model": args.teacher_model,
                    }
                )

            if not records:
                print(f"[warn] no valid samples produced for {entry.get('schema_id')}; skipping.")
            else:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                rows_written += len(records)
                print(f"[info] ✓ {entry.get('schema_id')}: kept {len(records)} / target {args.samples_per_schema}")

            if checkpoint_due:
                save_checkpoint(
                    args.checkpoint_path,
                    next_schema_index=idx + 1,
                    rng=rng,
                    current_date=current_date,
                    dataset_out=args.out,
                    schemas_path=args.schemas,
                    rows_written=rows_written,
                    total_schemas=total,
                )

    if args.checkpoint_path:
        args.checkpoint_path.unlink(missing_ok=True)
    print(f"Wrote {rows_written} rows to {args.out}")


if __name__ == "__main__":
    main()
