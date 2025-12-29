"""Inject domain fields/enums/operators into the base JSON Schema template."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401
from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import canonical_json, load_json, load_jsonl, write_jsonl


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

    parser = argparse.ArgumentParser(description="Build per-domain JSON Schemas.", parents=[config_parser])
    parser.add_argument(
        "--base-template",
        type=Path,
        default=Path("data/base_schema_template.json"),
        help="Base JSON Schema template to clone.",
    )
    parser.add_argument(
        "--domain-specs",
        type=Path,
        default=Path(defaults.get("schema_specs_out", "outputs/domain_specs.jsonl")),
        help="JSONL file produced by generate_domain_specs.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(defaults.get("final_schemas_out", "outputs/final_schemas.jsonl")),
        help="Where to write JSONL with full schemas.",
    )
    parser.add_argument(
        "--operator-catalog",
        type=Path,
        default=Path("data/operator_catalog.json"),
        help="JSON array of operator const/description pairs to sample from.",
    )
    parser.add_argument(
        "--min-operators",
        type=int,
        default=int(defaults.get("min_operators", 4)),
        help="Minimum operator count to include in each schema.",
    )
    parser.add_argument(
        "--max-operators",
        type=int,
        default=int(defaults.get("max_operators", 7)),
        help="Maximum operator count to include in each schema.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("seed", 17)),
        help="Seed for deterministic operator sampling.",
    )
    return parser.parse_args(remaining)


def normalize_operators(operators: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(operators, list):
        return normalized
    for op in operators:
        if isinstance(op, dict) and "const" in op:
            normalized.append({"const": str(op["const"]), "description": str(op.get("description", ""))})
        elif isinstance(op, str):
            normalized.append({"const": op, "description": ""})
    return normalized


def load_operator_catalog(path: Path) -> List[Dict[str, str]]:
    operators = normalize_operators(load_json(path))
    if not operators:
        raise ValueError(f"No valid operators found in {path}")
    return operators


def choose_operators(
    spec: Dict[str, Any],
    catalog: List[Dict[str, str]],
    rng: random.Random,
    min_count: int,
    max_count: int,
) -> List[Dict[str, str]]:
    # Honor explicitly provided operators on the spec if present.
    explicit = normalize_operators(spec.get("operators"))
    if explicit:
        return explicit

    max_available = len(catalog)
    lower = max(1, min(min_count, max_available))
    upper = max(lower, min(max_count, max_available))
    count = rng.randint(lower, upper)
    return rng.sample(catalog, k=count)


def build_field_one_of(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for field in fields:
        items.append({"const": field["const"], "description": field.get("description", "")})
    return items


def build_enum_rules(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for field in fields:
        values = field.get("enum") or []
        if not values:
            continue
        value_one_of = [{"const": v["const"], "description": v.get("description", "")} for v in values]
        rules.append(
            {
                "if": {"properties": {"field": {"const": field["const"]}}, "required": ["field"]},
                "then": {"properties": {"value": {"oneOf": value_one_of}}},
            }
        )
    return rules


def build_schema(spec: Dict[str, Any], template: Dict[str, Any], operators: List[Dict[str, str]]) -> Dict[str, Any]:
    schema = copy.deepcopy(template)
    schema["$id"] = spec["schema_id"]
    schema["title"] = f"{spec['domain']} funnel definition"
    if spec.get("description"):
        schema["description"] = spec["description"]

    try:
        condition_schema = schema["properties"]["steps"]["items"]["properties"]["conditions"]["items"]
    except KeyError as exc:
        raise RuntimeError("Base template does not expose expected condition path") from exc

    fields = spec.get("fields", [])
    condition_schema["properties"]["field"]["oneOf"] = build_field_one_of(fields)
    condition_schema["properties"]["operator"]["oneOf"] = operators
    condition_schema["allOf"] = build_enum_rules(fields)
    return schema


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    template = load_json(args.base_template)
    specs = load_jsonl(args.domain_specs)
    operator_catalog = load_operator_catalog(args.operator_catalog)

    built = []
    for spec in specs:
        operators = choose_operators(spec, operator_catalog, rng, args.min_operators, args.max_operators)
        schema = build_schema(spec, template, operators)
        built.append(
            {
                "schema_id": spec["schema_id"],
                "domain": spec.get("domain"),
                "description": spec.get("description"),
                "fields": spec.get("fields", []),
                "operators": operators,
                "example_queries": spec.get("example_queries", []),
                "schema": schema,
                "schema_json": canonical_json(schema),
            }
        )

    write_jsonl(args.out, built)
    print(f"Wrote {len(built)} schemas to {args.out}")


if __name__ == "__main__":
    main()
