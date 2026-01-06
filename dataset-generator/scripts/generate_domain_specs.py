"""Generate domain-specific field and enum specs with GPT-OSS:120b via ollama."""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401
from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import write_jsonl
from lib.llm import run_ollama
from lib.parsing import extract_json_object
from model_config import default_model


OLLAMA_MODEL = default_model()


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
    if not defaults.get("schema_specs_out"):
        raise SystemExit("dataset_generation.schema_specs_out must be set in config/defaults.json.")
    parser = argparse.ArgumentParser(description="Generate domain specs JSONL.", parents=[config_parser])
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(defaults.get("prompts_path", "data/domain_prompts.jsonl")),
        help="Path to JSONL domain prompts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(defaults["schema_specs_out"]),
        help="Where to write JSONL specs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help="Ollama model name to call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("seed", 11)),
        help="Seed for deterministic fallbacks.",
    )
    parser.add_argument(
        "--examples-per-schema",
        type=int,
        default=int(defaults.get("examples_per_schema", 4)),
        help="How many example natural language queries to request from the model.",
    )
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        default=bool(defaults.get("offline_fallback", False)),
        help="Skip ollama and use deterministic stubs instead (for testing).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(defaults.get("max_retries", -1)),
        help="Retries when Ollama output is invalid. Use -1 for unlimited retries.",
    )
    return parser.parse_args(remaining)


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "schema"


def normalize_enum(values: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not values:
        return normalized
    for item in values:
        if isinstance(item, dict) and "const" in item:
            normalized.append({"const": str(item["const"]), "description": str(item.get("description", ""))})
        else:
            normalized.append({"const": str(item), "description": ""})
    return normalized


def normalize_fields(fields: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(fields, list):
        return normalized
    for field in fields:
        if not isinstance(field, dict) or "const" not in field:
            continue
        normalized.append(
            {
                "const": str(field["const"]),
                "description": str(field.get("description", "")),
                "enum": normalize_enum(field.get("enum") or field.get("enums")),
            }
        )
    return normalized


def stub_spec(prompt: Dict[str, Any], rng: random.Random, examples_per_schema: int) -> Dict[str, Any]:
    domain = prompt["domain"]
    slug = slugify(domain)
    fields = [
        {"const": f"{slug}.category", "description": f"{domain} category.", "enum": [{"const": "Default", "description": ""}]},
        {"const": f"{slug}.status", "description": f"{domain} lifecycle status.", "enum": [{"const": "active", "description": ""}, {"const": "draft", "description": ""}]},
        {"const": f"{slug}.owner", "description": "Owner or assignee.", "enum": []},
        {"const": f"{slug}.priority", "description": "Priority label.", "enum": [{"const": "High", "description": ""}, {"const": "Low", "description": ""}]},
    ]
    example_queries = [
        f"{domain.title()} with {fields[0]['const']} equals Default",
        f"{domain.title()} where {fields[1]['const']} is active",
    ]
    while len(example_queries) < examples_per_schema:
        example_queries.append(f"{domain.title()} filter {rng.choice(fields)['const']} equals sample")
    return {
        "schema_id": f"{slug}_funnel_v1",
        "domain": domain,
        "description": prompt.get("description", ""),
        "fields": fields,
        "example_queries": example_queries[:examples_per_schema],
        "source_prompt": prompt,
    }


def build_prompt(domain: str, description: str, examples_per_schema: int) -> str:
    slug = slugify(domain)
    return textwrap.dedent(
        f"""
        You are designing a JSON Schema-compatible DSL catalog.
        Domain: {domain}
        Domain description: {description or "N/A"}

        Produce a SINGLE JSON object with this shape:
        {{
          "schema_id": "{slug}_funnel_v1",
          "domain": "{domain}",
          "description": "...",
          "fields": [
            {{
              "const": "ec.team",
              "description": "High-level team or track.",
              "enum": [
                {{ "const": "Launch", "description": "..." }},
                {{ "const": "Build", "description": "..." }}
              ]
            }}
          ],
          "example_queries": ["English or French natural language queries referencing the fields"]
        }}

        Guidance:
        - Provide 5-12 fields using dot-separated identifiers.
        - Where possible, include enum values with short descriptions; otherwise leave enum empty.
        - Keep strings concise; no prose outside the JSON object.
        - example_queries must include at least {examples_per_schema} short, varied queries mixing EN/FR.
        - Never wrap the JSON object in explanations or extra text.
        """
    ).strip()


def call_ollama(prompt: str, model: str) -> str:
    return run_ollama(prompt, model)


def synthesize_spec(prompt: Dict[str, Any], args: argparse.Namespace, rng: random.Random) -> Dict[str, Any] | None:
    if args.offline_fallback:
        return stub_spec(prompt, rng, args.examples_per_schema)

    llm_prompt = build_prompt(prompt["domain"], prompt.get("description", ""), args.examples_per_schema)
    attempts = 0
    max_retries = args.max_retries
    while True:
        try:
            raw = call_ollama(llm_prompt, args.model)
            parsed = extract_json_object(raw)

            fields = normalize_fields(parsed.get("fields"))
            if not fields:
                raise ValueError("no fields returned")

            example_queries = parsed.get("example_queries") or []
            example_queries = [str(q) for q in example_queries if isinstance(q, (str, int, float))]
            if len(example_queries) < args.examples_per_schema:
                raise ValueError(f"only {len(example_queries)} example queries")

            return {
                "schema_id": parsed.get("schema_id") or f"{slugify(prompt['domain'])}_funnel_v1",
                "domain": parsed.get("domain") or prompt["domain"],
                "description": parsed.get("description") or prompt.get("description", ""),
                "fields": fields,
                "example_queries": example_queries[: args.examples_per_schema],
                "source_prompt": prompt,
            }
        except Exception as exc:  # noqa: BLE001
            attempts += 1
            limit_hit = max_retries >= 0 and attempts > max_retries
            if limit_hit:
                print(f"[warn] ollama generation failed for {prompt['domain']} after {attempts} attempts: {exc}. Skipping domain.")
                return None
            print(
                f"[retry] {prompt['domain']} generation failed (attempt {attempts}/{max_retries if max_retries >= 0 else 'inf'}): {exc}"
            )
            continue


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    prompts = load_prompts(args.prompts)
    total = len(prompts)
    print(f"[info] Loaded {total} domain prompts from {args.prompts}")
    specs: List[Dict[str, Any]] = []
    skipped = 0
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[info] [{idx}/{total}] Generating spec for domain '{prompt['domain']}'")
        spec = synthesize_spec(prompt, args, rng)
        if spec is None:
            skipped += 1
            continue
        specs.append(spec)
        print(
            f"[info] ✓ {prompt['domain']}: {len(spec['fields'])} fields, {len(spec['example_queries'])} example queries"
        )
    write_jsonl(args.out, specs)
    suffix = f" (skipped {skipped} domains)" if skipped else ""
    print(f"Wrote {len(specs)} specs to {args.out}{suffix}")


if __name__ == "__main__":
    main()
