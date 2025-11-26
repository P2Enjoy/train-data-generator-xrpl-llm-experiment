"""Inject domain fields/enums into the base JSON Schema template."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-domain JSON Schemas.")
    parser.add_argument(
        "--base-template",
        type=Path,
        default=Path("data/base_schema_template.json"),
        help="Base JSON Schema template to clone.",
    )
    parser.add_argument(
        "--domain-specs",
        type=Path,
        default=Path("outputs/domain_specs.jsonl"),
        help="JSONL file produced by generate_domain_specs.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/final_schemas.jsonl"),
        help="Where to write JSONL with full schemas.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def build_schema(spec: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
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
    condition_schema["allOf"] = build_enum_rules(fields)
    return schema


def main() -> None:
    args = parse_args()
    template = load_json(args.base_template)
    specs = load_jsonl(args.domain_specs)

    built = []
    for spec in specs:
        schema = build_schema(spec, template)
        built.append(
            {
                "schema_id": spec["schema_id"],
                "domain": spec.get("domain"),
                "description": spec.get("description"),
                "fields": spec.get("fields", []),
                "example_queries": spec.get("example_queries", []),
                "schema": schema,
                "schema_json": canonical(schema),
            }
        )

    write_jsonl(args.out, built)
    print(f"Wrote {len(built)} schemas to {args.out}")


if __name__ == "__main__":
    main()
