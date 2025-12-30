"""Evaluate Gemma 3 270M student adapters against the generated dataset."""

from __future__ import annotations

import unsloth
import argparse
import builtins
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import _bootstrap  # noqa: F401
import torch
from datasets import load_dataset
from jsonschema import Draft7Validator
from peft import PeftModel
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

from lib.io import canonical_json, write_jsonl
from lib.llm import run_ollama
from lib.parsing import extract_json_object
from model_config import default_model

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
    default_teacher = default_model()
    parser = argparse.ArgumentParser(description="Evaluate student LoRA adapters on held-out samples.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/d_03_dataset.jsonl"),
        help="JSONL produced by generate_dataset.py.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("outputs/student_runs/gemma3-270m/checkpoint-final"),
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-3-270m-it",
        help="Base model id used during training.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--max-samples", type=int, default=200, help="How many rows to evaluate.")
    parser.add_argument("--include-invalid", action="store_true", help="Include invalid targets during eval.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Generation top-p.")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load student in 4-bit for eval (default: on).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/student_runs/eval"), help="Where to store logs.")
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=default_teacher,
        help="Ollama model to compare against (teacher baseline). Defaults to model from .llmrc/LLM_MODEL.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print progress every N samples.",
    )
    return parser.parse_args()


def load_student(
    base_model: str, adapter_dir: Path, max_seq_length: int, load_in_4bit: bool
) -> Tuple[Any, Any]:
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    if not hasattr(builtins, "VARIANT_KWARG_KEYS"):
        builtins.VARIANT_KWARG_KEYS = {"adapter_name"}
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def extract_json_block(text: str) -> Dict[str, Any]:
    return extract_json_object(text)


def call_teacher(prompt: str, model: str) -> str:
    return run_ollama(prompt, model)


def build_prompt(schema_json: str, query: str, current_date: str) -> str:
    return PROMPT_TEMPLATE.format(schema_json=schema_json, query=query, current_date=current_date or "N/A")


def grade_with_teacher(schema_json: str, query: str, student_ast: Dict[str, Any], model: str) -> Tuple[bool, str]:
    """Ask the teacher model to judge if the student's AST satisfies the query."""
    payload = canonical_json(student_ast)
    prompt = f"""
You are the authoritative teacher for a JSON Schema-governed DSL.
Given the JSON Schema, the user's natural language query, and a candidate AST produced by a student model,
decide if the candidate AST correctly answers the query while respecting the schema.

Instructions:
- DO NOT rewrite or correct the AST; only judge it.
- Require that the AST is semantically consistent with the query (fields, operators, values, timeframe).
- If the AST misses key intent from the query, or contradicts it, mark it as not good.
- Respond ONLY with a JSON object of the form:
  {{"is_good": true/false, "reason": "short justification"}}

[SCHEMA]
{schema_json}
[/SCHEMA]
[QUERY]
{query}
[/QUERY]
[CANDIDATE_AST]
{payload}
[/CANDIDATE_AST]
"""
    answer = call_teacher(prompt, model)
    graded = extract_json_block(answer)
    is_good = bool(graded.get("is_good"))
    reason = str(graded.get("reason", "") or "")
    return is_good, reason


def semantic_signature(ast: Dict[str, Any]) -> str:
    """Canonicalize AST semantics (ignores prompt/name/description, sorts conditions)."""
    normalized = {"timeframe": ast.get("timeframe")}
    steps = []
    for step in ast.get("steps") or []:
        conds = []
        for cond in step.get("conditions") or []:
            conds.append(
                {
                    "field": cond.get("field"),
                    "operator": cond.get("operator"),
                    "value": cond.get("value"),
                }
            )
        conds = sorted(conds, key=lambda c: json.dumps(c, sort_keys=True))
        steps.append({"conditions": conds})
    normalized["steps"] = sorted(steps, key=lambda s: json.dumps(s, sort_keys=True))
    # stringify to make equality checks order-agnostic and cheap
    return canonical_json(normalized)


def evaluate_sample(
    sample: Dict[str, Any],
    model,
    tokenizer,
    validator: Draft7Validator,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    teacher_model: str | None = None,
) -> Dict[str, Any]:
    prompt_text = build_prompt(sample["schema_json"], sample["query"], sample.get("current_date", ""))
    messages = [
        {"role": "system", "content": "You are a JSON AST generator."},
        {"role": "user", "content": prompt_text},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix("<bos>")

    inputs = tokenizer(chat_prompt, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    gen_ids = generated[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    result: Dict[str, Any] = {
        "schema_id": sample.get("schema_id"),
        "query": sample["query"],
        "target_valid": sample.get("is_valid", True),
        "raw_output": decoded,
        "prompt_text": prompt_text,
    }

    try:
        parsed = extract_json_block(decoded)
        result["parsed"] = True
    except Exception as exc:  # noqa: BLE001
        result["parsed"] = False
        result["parse_error"] = str(exc)
        return result

    result["student_canonical"] = canonical_json(parsed)
    try:
        target = json.loads(sample["ast_json"])
        result["target_canonical"] = canonical_json(target)
        result["exact_match"] = result["student_canonical"] == result["target_canonical"]
    except json.JSONDecodeError as exc:
        result["exact_match"] = False
        result["target_error"] = str(exc)
        
    # semantic equivalence ignores prompt/name/description ordering and uses condition sets
    try:
        result["semantic_match_target"] = semantic_signature(parsed) == semantic_signature(json.loads(sample["ast_json"]))
    except Exception:
        result["semantic_match_target"] = False

    try:
        validator.validate(parsed)
        result["schema_valid"] = True
    except Exception as exc:  # noqa: BLE001
        result["schema_valid"] = False
        result["schema_error"] = str(exc)

    teacher_ok: bool | None = None
    if teacher_model:
        try:
            teacher_out = call_teacher(prompt_text, teacher_model)
            teacher_json = extract_json_block(teacher_out)
            result["teacher_canonical"] = canonical_json(teacher_json)
            result["matches_teacher"] = result["student_canonical"] == result["teacher_canonical"]
            result["semantic_match_teacher"] = semantic_signature(parsed) == semantic_signature(teacher_json)
            if result.get("schema_valid"):
                teacher_ok, reason = grade_with_teacher(sample["schema_json"], sample["query"], parsed, teacher_model)
                result["teacher_verdict"] = teacher_ok
                result["teacher_reason"] = reason
        except Exception as exc:  # noqa: BLE001
            result["matches_teacher"] = False
            result["semantic_match_teacher"] = False
            result["teacher_error"] = str(exc)

    if teacher_ok is not None:
        result["semantic_pass"] = bool(result.get("schema_valid") and teacher_ok)
    else:
        result["semantic_pass"] = bool(
            result.get("schema_valid")
            and (
                result.get("semantic_match_target")
                or result.get("semantic_match_teacher")
            )
        )

    return result


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    parsed = sum(1 for r in rows if r.get("parsed"))
    schema_valid = sum(1 for r in rows if r.get("schema_valid"))
    exact = sum(1 for r in rows if r.get("exact_match"))
    matches_teacher = sum(1 for r in rows if r.get("matches_teacher"))
    semantic_target = sum(1 for r in rows if r.get("semantic_match_target"))
    semantic_teacher = sum(1 for r in rows if r.get("semantic_match_teacher"))
    semantic_pass = sum(1 for r in rows if r.get("semantic_pass"))
    teacher_verdicts = [r.get("teacher_verdict") for r in rows if "teacher_verdict" in r]
    teacher_accept = sum(1 for v in teacher_verdicts if v)
    return {
        "total": total,
        "parsed_rate": parsed / total if total else 0.0,
        "schema_valid_rate": schema_valid / total if total else 0.0,
        "exact_match_rate": exact / total if total else 0.0,
        "teacher_agreement_rate": matches_teacher / total if total else 0.0,
        "semantic_match_target_rate": semantic_target / total if total else 0.0,
        "semantic_match_teacher_rate": semantic_teacher / total if total else 0.0,
        "teacher_accept_rate": (teacher_accept / len(teacher_verdicts)) if teacher_verdicts else 0.0,
        "semantic_pass_rate": semantic_pass / total if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    if not args.teacher_model:
        raise SystemExit("Teacher model is required; set LLM_MODEL or .llmrc to an ollama model id.")

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    if not args.include_invalid:
        dataset = dataset.filter(lambda x: bool(x.get("is_valid", True)))
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    if not args.teacher_model:
        print("[eval] teacher disabled (no --teacher-model provided and no .llmrc found)")
    print(
        f"[eval] samples={len(dataset)} 4bit={args.load_in_4bit} "
        f"teacher={args.teacher_model or 'none'} log_every={args.log_every}"
    )

    model, tokenizer = load_student(args.base_model, args.adapter, args.max_seq_length, args.load_in_4bit)
    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset, start=1):
        schema = json.loads(sample["schema_json"])
        validator = Draft7Validator(schema)
        result = evaluate_sample(
            sample,
            model,
            tokenizer,
            validator,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
            teacher_model=args.teacher_model,
        )
        results.append(result)
        if args.log_every and idx % args.log_every == 0:
            print(f"[eval] processed {idx}/{len(dataset)}")

    summary = summarize(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "evaluation_results.jsonl", results)
    (args.out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
