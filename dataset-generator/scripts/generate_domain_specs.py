"""Generate domain-specific field and enum specs with GPT-OSS:120b via ollama."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List


OLLAMA_MODEL = "gpt-oss:120b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate domain specs JSONL.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/domain_prompts.jsonl"),
        help="Path to JSONL domain prompts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/domain_specs.jsonl"),
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
        default=11,
        help="Seed for deterministic fallbacks.",
    )
    parser.add_argument(
        "--examples-per-schema",
        type=int,
        default=4,
        help="How many example natural language queries to request from the model.",
    )
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        help="Skip ollama and use deterministic stubs instead (for testing).",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "schema"


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    candidate = match.group(0)
    return json.loads(candidate)


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


def synthesize_spec(prompt: Dict[str, Any], args: argparse.Namespace, rng: random.Random) -> Dict[str, Any]:
    if args.offline_fallback:
        return stub_spec(prompt, rng, args.examples_per_schema)

    llm_prompt = build_prompt(prompt["domain"], prompt.get("description", ""), args.examples_per_schema)
    try:
        raw = call_ollama(llm_prompt, args.model)
        parsed = extract_json_block(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] ollama generation failed for {prompt['domain']}: {exc}")
        return stub_spec(prompt, rng, args.examples_per_schema)

    fields = normalize_fields(parsed.get("fields"))
    if not fields:
        fields = stub_spec(prompt, rng, args.examples_per_schema)["fields"]

    example_queries = parsed.get("example_queries") or []
    example_queries = [str(q) for q in example_queries if isinstance(q, (str, int, float))]
    while len(example_queries) < args.examples_per_schema:
        example_queries.append(f"{prompt['domain'].title()} example query {len(example_queries) + 1}")

    return {
        "schema_id": parsed.get("schema_id") or f"{slugify(prompt['domain'])}_funnel_v1",
        "domain": parsed.get("domain") or prompt["domain"],
        "description": parsed.get("description") or prompt.get("description", ""),
        "fields": fields,
        "example_queries": example_queries[: args.examples_per_schema],
        "source_prompt": prompt,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    prompts = load_prompts(args.prompts)
    specs = [synthesize_spec(prompt, args, rng) for prompt in prompts]
    write_jsonl(args.out, specs)
    print(f"Wrote {len(specs)} specs to {args.out}")


if __name__ == "__main__":
    main()
