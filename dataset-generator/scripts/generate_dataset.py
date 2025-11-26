"""Generate positive and negative dataset rows from domain JSON Schemas."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

OPERATORS = ["equals", "not_equals", "like", "in"]
OP_PHRASES = {
    "equals": "is",
    "not_equals": "is not",
    "like": "mentions",
    "in": "is in",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset JSONL.")
    parser.add_argument(
        "--schemas",
        type=Path,
        default=Path("outputs/d_02_final_schemas.jsonl"),
        help="JSONL produced by build_schemas.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/d_04_dataset.jsonl"),
        help="Where to write dataset JSONL.",
    )
    parser.add_argument(
        "--positives-per-schema",
        type=int,
        default=8,
        help="How many validated samples per schema.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.4,
        help="Negatives to generate per positive (approx).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Seed for deterministic generation.",
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


def canonical(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=2)


def format_error(error: ValidationError | None) -> str | None:
    if error is None:
        return None
    path = "/".join(str(elem) for elem in error.path)
    return f"{error.message} @ {path or '<root>'}"


def sample_timeframe(rng: random.Random, current_date: date) -> Dict[str, Any]:
    start = current_date - timedelta(days=rng.randint(7, 40))
    end = start + timedelta(days=rng.randint(5, 20))
    return {"start": start.isoformat(), "end": end.isoformat()}


def choose_value(field: Dict[str, Any], rng: random.Random) -> Any:
    enum_values = field.get("enum") or []
    if enum_values:
        return rng.choice(enum_values)["const"]
    free_text = [
        "Paris",
        "remote",
        "internal",
        "developer",
        "beta",
        "gamma",
        "priority",
    ]
    return rng.choice(free_text)


def build_conditions(fields: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    if not fields:
        raise ValueError("No fields available to build conditions.")
    count = min(len(fields), max(1, rng.randint(1, 3)))
    selected = rng.sample(fields, k=count)
    conditions = []
    for field in selected:
        conditions.append(
            {
                "field": field["const"],
                "operator": rng.choice(OPERATORS),
                "value": choose_value(field, rng),
            }
        )
    return conditions


def human_time_phrase(timeframe: Dict[str, Any], current_date: date) -> str:
    start = date.fromisoformat(timeframe["start"])
    end = date.fromisoformat(timeframe["end"])
    if end <= current_date and (current_date - end).days <= 21:
        days = (current_date - start).days
        if days <= 21:
            return f"in the last {days or 1} days"
        weeks = max(1, round(days / 7))
            # noqa: E117
        return f"in the last {weeks} weeks"
    if end >= current_date and (end - current_date).days <= 30:
        days = (end - current_date).days + 1
        weeks = max(1, round(days / 7))
        return f"over the next {weeks} weeks"
    if start.month == end.month and start.year == end.year:
        return f"during {start.strftime('%B %Y')}"
    return f"between {start.strftime('%B %-d')} and {end.strftime('%B %-d %Y')}"


def build_query(
    entry: Dict[str, Any],
    conditions: List[Dict[str, Any]],
    timeframe: Dict[str, Any],
    current_date: date,
    rng: random.Random,
) -> str:
    if entry.get("example_queries"):
        base = str(rng.choice(entry["example_queries"]))
        return f"{base.rstrip('. ')} {human_time_phrase(timeframe, current_date)}".strip()

    parts = []
    for cond in conditions:
        field_label = cond["field"].split(".")[-1].replace("_", " ")
        value = cond["value"]
        op_word = OP_PHRASES.get(cond["operator"], cond["operator"])
        parts.append(f"{field_label} {op_word} {value}")
    timeframe_text = human_time_phrase(timeframe, current_date)
    lead = rng.choice(["Show", "Find", "List", "Retrieve", "Give me"])
    joined = " and ".join(parts)
    return f"{lead} {entry.get('domain','')} where {joined} {timeframe_text}".strip()


def make_positive_sample(
    entry: Dict[str, Any],
    validator: Draft7Validator,
    rng: random.Random,
    current_date: date,
    attempts: int = 6,
) -> Tuple[str, Dict[str, Any]]:
    for _ in range(attempts):
        conditions = build_conditions(entry["fields"], rng)
        timeframe = sample_timeframe(rng, current_date)
        query = build_query(entry, conditions, timeframe, current_date, rng)
        ast = {
            "prompt": query,
            "steps": [
                {
                    "name": "step_1",
                    "description": f"Auto generated filter for {entry['domain']}",
                    "conditions": conditions,
                }
            ],
            "timeframe": timeframe,
        }
        if not next(validator.iter_errors(ast), None):
            return query, ast
    raise RuntimeError(f"Failed to synthesize valid AST for {entry['schema_id']}")


def mutate_ast(
    ast: Dict[str, Any],
    entry: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Dict[str, Any], str]:
    mutated = copy.deepcopy(ast)
    mutation = rng.choice(
        [
            "drop_timeframe",
            "empty_conditions",
            "extra_property",
            "bad_enum",
            "unknown_field",
            "wrong_operator",
            "missing_description",
        ]
    )

    if mutation == "drop_timeframe":
        mutated.pop("timeframe", None)
    elif mutation == "empty_conditions":
        mutated["steps"][0]["conditions"] = []
    elif mutation == "extra_property":
        mutated["steps"][0]["conditions"][0]["unexpected"] = "extra"
    elif mutation == "bad_enum":
        candidates = [field for field in entry.get("fields", []) if field.get("enum")]
        if candidates:
            target_field = rng.choice(candidates)
            mutated["steps"][0]["conditions"][0]["field"] = target_field["const"]
            mutated["steps"][0]["conditions"][0]["value"] = "INVALID_ENUM_VALUE"
        else:
            mutation = "unknown_field"
    if mutation == "unknown_field":
        mutated["steps"][0]["conditions"][0]["field"] = f"{entry['schema_id']}.unexpected"
    elif mutation == "wrong_operator":
        mutated["steps"][0]["conditions"][0]["operator"] = "gt"
    elif mutation == "missing_description":
        if mutated["steps"]:
            mutated["steps"][0].pop("description", None)

    return mutated, mutation


def ensure_invalid(
    validator: Draft7Validator, mutated: Dict[str, Any], mutation: str
) -> Tuple[Dict[str, Any], str, ValidationError | None]:
    error = next(validator.iter_errors(mutated), None)
    if error:
        return mutated, mutation, error

    fallback = copy.deepcopy(mutated)
    fallback.pop("prompt", None)
    error = next(validator.iter_errors(fallback), None)
    if error:
        return fallback, f"{mutation}_forced", error

    return fallback, f"{mutation}_unexpectedly_valid", None


def build_positive_records(
    entry: Dict[str, Any],
    validator: Draft7Validator,
    rng: random.Random,
    current_date: date,
    count: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _ in range(count):
        query, ast = make_positive_sample(entry, validator, rng, current_date)
        records.append(
            {
                "schema_id": entry["schema_id"],
                "domain": entry.get("domain"),
                "schema_json": entry["schema_json"],
                "query": query,
                "current_date": current_date.isoformat(),
                "ast_json": canonical(ast),
                "is_valid": True,
                "validation_error": None,
                "error_type": None,
                "_ast": ast,
            }
        )
    return records


def build_negative_records(
    entry: Dict[str, Any],
    validator: Draft7Validator,
    rng: random.Random,
    positives: List[Dict[str, Any]],
    target_count: int,
) -> List[Dict[str, Any]]:
    negatives: List[Dict[str, Any]] = []
    if not positives:
        return negatives

    while len(negatives) < target_count:
        base = rng.choice(positives)
        mutated, mutation = mutate_ast(base["_ast"], entry, rng)
        mutated, mutation, error = ensure_invalid(validator, mutated, mutation)
        negatives.append(
            {
                "schema_id": entry["schema_id"],
                "domain": entry.get("domain"),
                "schema_json": entry["schema_json"],
                "query": base["query"],
                "current_date": base.get("current_date"),
                "ast_json": canonical(mutated),
                "is_valid": error is None,
                "validation_error": format_error(error) if error else "Unexpectedly valid",
                "error_type": mutation,
            }
        )
    return negatives


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    schemas = load_jsonl(args.schemas)

    dataset_rows: List[Dict[str, Any]] = []
    current_date = date.today()
    for entry in schemas:
        if not entry.get("fields"):
            print(f"[warn] skipping {entry.get('schema_id')} because no fields were provided")
            continue
        validator = Draft7Validator(entry["schema"])
        positives = build_positive_records(entry, validator, rng, current_date, args.positives_per_schema)
        negative_target = int(math.ceil(len(positives) * args.negative_ratio))
        negatives = build_negative_records(entry, validator, rng, positives, negative_target)

        for record in positives:
            record.pop("_ast", None)
        dataset_rows.extend(positives)
        dataset_rows.extend(negatives)

    write_jsonl(args.out, dataset_rows)
    print(f"Wrote {len(dataset_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
