"""Generate positive and negative dataset rows from domain JSON Schemas."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import _bootstrap  # noqa: F401
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import canonical_json, load_jsonl

DEFAULT_OPERATORS = ["equals", "not_equals", "like", "in"]
OP_PHRASES = {
    "equals": "is",
    "not_equals": "is not",
    "like": "mentions",
    "in": "is in",
    "not_in": "is not in",
    "contains": "contains",
    "starts_with": "starts with",
    "ends_with": "ends with",
    "gt": "after",
    "lt": "before",
}
LEAD_PHRASES = [
    "Show",
    "Find",
    "List",
    "Retrieve",
    "Give me",
    "Montre-moi",
    "Trouve",
    "Liste",
]
UNSUPPORTED_OPERATOR_PHRASES = {
    "lt": "less than",
    "gt": "greater than",
    "contains": "contains",
    "starts_with": "starts with",
    "ends_with": "ends with",
    "not_equals": "not equal to",
    "not_in": "not in",
    "like": "similar to",
}
SUGGESTION_OPERATOR_PREFERENCE = [
    "equals",
    "in",
    "like",
    "contains",
    "starts_with",
    "ends_with",
    "gt",
    "lt",
    "not_equals",
    "not_in",
]


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

    if not defaults.get("final_schemas_out") or not defaults.get("dataset_out"):
        raise SystemExit("dataset_generation.final_schemas_out and dataset_out must be set in config/defaults.json.")

    parser = argparse.ArgumentParser(description="Generate dataset JSONL.", parents=[config_parser])
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
        "--positives-per-schema",
        type=int,
        default=int(defaults.get("positives_per_schema", 8)),
        help="How many validated samples per schema.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=float(defaults.get("negative_ratio", 0.4)),
        help="Negatives to generate per positive (approx).",
    )
    parser.add_argument(
        "--refusals-per-schema",
        type=int,
        default=int(defaults.get("refusals_per_schema", 0)),
        help="How many refusal samples to synthesize per schema.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("seed", 11)),
        help="Seed for deterministic generation.",
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
    args = parser.parse_args(remaining)
    args.checkpoint_path_default = Path(defaults["checkpoint_path"]) if defaults.get("checkpoint_path") else None
    args.dataset_out_default = Path(defaults["dataset_out"])
    return args


def extract_operator_metadata(entry: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    ops: List[str] = []
    canonical_map: Dict[str, str] = {}
    raw_ops = entry.get("operators") or []
    if raw_ops:
        for op in raw_ops:
            if isinstance(op, dict) and "const" in op:
                const = str(op["const"]).strip()
                if not const:
                    continue
                ops.append(const)
                canonical_map[const] = str(op.get("canonical") or const)
            elif isinstance(op, str):
                const = op.strip()
                if not const:
                    continue
                ops.append(const)
                canonical_map[const] = const
    if not ops:
        try:
            one_of = (
                entry["schema"]["properties"]["steps"]["items"]["properties"]["conditions"]["items"]["properties"]["operator"]["oneOf"]
            )
            for item in one_of:
                if isinstance(item, dict) and "const" in item:
                    const = str(item["const"]).strip()
                    if const:
                        ops.append(const)
                        canonical_map[const] = const
        except Exception:  # noqa: BLE001
            ops = []
            canonical_map = {}
    if not ops:
        ops = DEFAULT_OPERATORS.copy()
        canonical_map = {op: op for op in ops}
    else:
        for op in ops:
            canonical_map.setdefault(op, op)
    return ops, canonical_map


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


def field_text_blob(fields: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for field in fields:
        parts.append(str(field.get("const", "")))
        parts.append(str(field.get("description", "")))
        for enum in field.get("enum") or []:
            parts.append(str(enum.get("const", "")))
            parts.append(str(enum.get("description", "")))
    return " ".join(parts).lower()


def pick_missing_concept(fields: List[Dict[str, Any]], rng: random.Random) -> str:
    blob = field_text_blob(fields)
    # Generate a placeholder attribute that is unlikely to exist by mutating known fields.
    if fields:
        base_field = rng.choice(fields)["const"]
        mutated = base_field.rsplit(".", 1)[0] + ".extra_attribute"
        if mutated.lower() not in blob:
            return mutated
    generic_tokens = ["attribute", "metric", "score", "flag", "status_extra", "tag"]
    candidate = f"{rng.choice(generic_tokens)}_{rng.randint(1, 999)}"
    return candidate


def choose_suggestion_operator(
    operators: List[str],
    canonical_map: Dict[str, str],
    rng: random.Random,
) -> str:
    allowed = [str(op) for op in operators if op]
    for canonical in SUGGESTION_OPERATOR_PREFERENCE:
        for op in allowed:
            if canonical_map.get(op, op) == canonical:
                return op
    return rng.choice(allowed) if allowed else "equals"


def build_suggestion(
    fields: List[Dict[str, Any]],
    operators: List[str],
    canonical_map: Dict[str, str],
    rng: random.Random,
) -> str:
    if not fields:
        return "I can suggest using the available filters in this schema."
    field = rng.choice(fields)
    operator = choose_suggestion_operator(operators, canonical_map, rng)
    value = choose_value(field, rng)
    canonical = canonical_map.get(operator, operator)
    op_word = OP_PHRASES.get(canonical, operator.replace("_", " "))
    return f"I can suggest filtering by {field['const']} {op_word} {value}."


def choose_refusal_value(operator: str, field: Dict[str, Any], canonical_map: Dict[str, str], rng: random.Random) -> str:
    canonical = canonical_map.get(operator, operator)
    if canonical in {"lt", "gt"}:
        return str(rng.randint(1, 100))
    return str(choose_value(field, rng))


def make_refusal_sample(
    entry: Dict[str, Any],
    operators: List[str],
    canonical_map: Dict[str, str],
    rng: random.Random,
    current_date: date,
) -> Tuple[str, Dict[str, Any], str]:
    fields = entry.get("fields", [])
    allowed_ops = {str(op) for op in operators}
    allowed_canonicals = {canonical_map.get(op, op) for op in allowed_ops}
    unsupported_ops = [op for op in UNSUPPORTED_OPERATOR_PHRASES if op not in allowed_canonicals]

    if unsupported_ops and fields and rng.random() < 0.5:
        operator = rng.choice(unsupported_ops)
        field = rng.choice(fields)
        value = choose_refusal_value(operator, field, canonical_map, rng)
        field_label = field["const"].split(".")[-1].replace("_", " ")
        phrase = UNSUPPORTED_OPERATOR_PHRASES[operator]
        lead = rng.choice(LEAD_PHRASES)
        query = f"{lead} {entry.get('domain', 'items')} where {field_label} {phrase} {value}".strip()
        reason = f"Cannot apply '{phrase}' because this schema does not support the '{operator}' operator."
        refusal_type = f"refusal_operator_{operator}"
    else:
        concept = pick_missing_concept(fields, rng)
        lead = rng.choice(LEAD_PHRASES)
        query = f"{lead} {entry.get('domain', 'items')} with {concept}".strip()
        if rng.random() < 0.5:
            timeframe = sample_timeframe(rng, current_date)
            query = f"{query} {human_time_phrase(timeframe, current_date)}"
        reason = f"Cannot filter by {concept} because there is no field for it in this schema."
        refusal_type = "refusal_missing_field"

    suggestion = build_suggestion(fields, operators, canonical_map, rng)
    ast = {
        "prompt": query,
        "refusal": {
            "reason": reason,
            "suggestion": suggestion,
        },
    }
    return query, ast, refusal_type

def pick_invalid_operator(allowed: List[str], rng: random.Random) -> str:
    candidates = [
        "unsupported_operator",
        "invalid_op",
        "regex",
        "between",
        "approx",
        "any_of",
        "none",
        "greater_or_equal",
    ]
    pool = candidates + [f"op_{rng.randint(1, 999)}" for _ in range(4)]
    choice = rng.choice(pool)
    while choice in allowed:
        choice = rng.choice(pool)
    return choice


def build_conditions(fields: List[Dict[str, Any]], operators: List[str], rng: random.Random) -> List[Dict[str, Any]]:
    if not fields:
        raise ValueError("No fields available to build conditions.")
    if not operators:
        raise ValueError("No operators available to build conditions.")
    count = min(len(fields), max(1, rng.randint(1, 3)))
    selected = rng.sample(fields, k=count)
    conditions = []
    for field in selected:
        conditions.append(
            {
                "field": field["const"],
                "operator": rng.choice(operators),
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
    canonical_map: Dict[str, str],
    rng: random.Random,
) -> str:
    if entry.get("example_queries"):
        candidates = [str(item) for item in entry["example_queries"] if isinstance(item, (str, int, float))]
        if candidates:
            base = rng.choice(candidates)
            return f"{base.rstrip('. ')} {human_time_phrase(timeframe, current_date)}".strip()

    parts = []
    for cond in conditions:
        field_label = cond["field"].split(".")[-1].replace("_", " ")
        value = cond["value"]
        canonical = canonical_map.get(cond["operator"], cond["operator"])
        op_word = OP_PHRASES.get(canonical, canonical.replace("_", " "))
        parts.append(f"{field_label} {op_word} {value}")
    timeframe_text = human_time_phrase(timeframe, current_date)
    lead = rng.choice(LEAD_PHRASES)
    joined = " and ".join(parts)
    return f"{lead} {entry.get('domain','')} where {joined} {timeframe_text}".strip()


def make_positive_sample(
    entry: Dict[str, Any],
    validator: Draft7Validator,
    operators: List[str],
    canonical_map: Dict[str, str],
    rng: random.Random,
    current_date: date,
    attempts: int = 6,
) -> Tuple[str, Dict[str, Any]]:
    for _ in range(attempts):
        conditions = build_conditions(entry["fields"], operators, rng)
        timeframe = sample_timeframe(rng, current_date)
        query = build_query(entry, conditions, timeframe, current_date, canonical_map, rng)
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
    operators: List[str],
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
        mutated["steps"][0]["conditions"][0]["operator"] = pick_invalid_operator(operators, rng)
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
    operators: List[str],
    canonical_map: Dict[str, str],
    rng: random.Random,
    current_date: date,
    count: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _ in range(count):
        query, ast = make_positive_sample(entry, validator, operators, canonical_map, rng, current_date)
        records.append(
            {
                "schema_id": entry["schema_id"],
                "domain": entry.get("domain"),
                "operators": operators,
                "schema_json": entry["schema_json"],
                "query": query,
                "current_date": current_date.isoformat(),
                "ast_json": canonical_json(ast),
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
    operators: List[str],
    rng: random.Random,
    positives: List[Dict[str, Any]],
    target_count: int,
) -> List[Dict[str, Any]]:
    negatives: List[Dict[str, Any]] = []
    if not positives:
        return negatives

    while len(negatives) < target_count:
        base = rng.choice(positives)
        mutated, mutation = mutate_ast(base["_ast"], entry, operators, rng)
        mutated, mutation, error = ensure_invalid(validator, mutated, mutation)
        negatives.append(
            {
                "schema_id": entry["schema_id"],
                "domain": entry.get("domain"),
                "operators": operators,
                "schema_json": entry["schema_json"],
                "query": base["query"],
                "current_date": base.get("current_date"),
                "ast_json": canonical_json(mutated),
                "is_valid": error is None,
                "validation_error": format_error(error) if error else "Unexpectedly valid",
                "error_type": mutation,
            }
        )
    return negatives


def build_refusal_records(
    entry: Dict[str, Any],
    operators: List[str],
    canonical_map: Dict[str, str],
    rng: random.Random,
    current_date: date,
    count: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _ in range(count):
        query, ast, refusal_type = make_refusal_sample(entry, operators, canonical_map, rng, current_date)
        records.append(
            {
                "schema_id": entry["schema_id"],
                "domain": entry.get("domain"),
                "operators": operators,
                "schema_json": entry["schema_json"],
                "query": query,
                "current_date": current_date.isoformat(),
                "ast_json": canonical_json(ast),
                "is_valid": True,
                "validation_error": None,
                "error_type": refusal_type,
            }
        )
    return records


def serialize_rng_state(state: Tuple[Any, ...]) -> List[Any]:
    version, inner, gaussian = state
    return [version, list(inner), gaussian]


def deserialize_rng_state(state: List[Any]) -> Tuple[Any, ...]:
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
    current_date: date,
    dataset_out: Path,
    schemas_path: Path,
    rows_written: int,
    total_schemas: int,
) -> None:
    payload = {
        "next_schema_index": next_schema_index,
        "rng_state": serialize_rng_state(rng.getstate()),
        "current_date": current_date.isoformat(),
        "dataset_out": str(dataset_out),
        "schemas_path": str(schemas_path),
        "rows_written": rows_written,
        "total_schemas": total_schemas,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = args.out.parent / "dataset_generation_checkpoint.json"
    elif args.checkpoint_path_default and checkpoint_path == args.checkpoint_path_default and args.out != args.dataset_out_default:
        checkpoint_path = args.out.parent / "dataset_generation_checkpoint.json"
    args.checkpoint_path = checkpoint_path
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1 when checkpointing is enabled.")

    rng = random.Random(args.seed)
    schemas = load_jsonl(args.schemas)
    total = len(schemas)
    print(f"[info] Building dataset from {total} schemas in {args.schemas}")

    current_date = date.today()
    start_index = 1
    rows_written = 0

    if args.resume:
        if not args.checkpoint_path or not args.checkpoint_path.exists():
            raise SystemExit(f"Checkpoint file not found: {args.checkpoint_path}")
        checkpoint = load_checkpoint(args.checkpoint_path)
        start_index = int(checkpoint.get("next_schema_index", 1))
        rows_written = int(checkpoint.get("rows_written", 0))
        checkpoint_date = checkpoint.get("current_date")
        if checkpoint_date:
            current_date = date.fromisoformat(checkpoint_date)
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
            if not entry.get("fields"):
                print(f"[warn] skipping {entry.get('schema_id')} because no fields were provided")
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
            operators, canonical_map = extract_operator_metadata(entry)
            validator = Draft7Validator(entry["schema"])
            positives = build_positive_records(
                entry,
                validator,
                operators,
                canonical_map,
                rng,
                current_date,
                args.positives_per_schema,
            )
            negative_target = int(math.ceil(len(positives) * args.negative_ratio))
            negatives = build_negative_records(entry, validator, operators, rng, positives, negative_target)
            refusals = build_refusal_records(entry, operators, canonical_map, rng, current_date, args.refusals_per_schema)
            print(
                f"[info] ✓ {entry.get('schema_id')}: {len(positives)} positives, {len(negatives)} negatives, {len(refusals)} refusals (target ratio {args.negative_ratio})"
            )

            for record in positives:
                record.pop("_ast", None)
            all_records = positives + negatives + refusals
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            rows_written += len(all_records)
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
