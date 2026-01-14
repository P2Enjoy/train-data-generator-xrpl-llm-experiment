"""Evaluate Gemma 3 270M student adapters against the generated dataset."""

from __future__ import annotations

import argparse
import builtins
import json
import time
import unsloth
from pathlib import Path
from typing import Any, Dict, List, Tuple

import _bootstrap  # noqa: F401
import torch
from datasets import Features, Sequence, Value, load_dataset
from jsonschema import Draft7Validator
from peft import PeftModel
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import canonical_json, load_jsonl
from lib.llm import run_ollama
from lib.parsing import extract_json_object
from model_config import default_model

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
    default_teacher = default_model()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to defaults JSON (config/defaults.json).",
    )
    config_args, remaining = config_parser.parse_known_args()
    training_defaults = load_section("training", config_args.config)
    data_defaults = load_section("dataset_generation", config_args.config)
    alignment_defaults = load_section("alignment", config_args.config)
    evaluation_defaults = load_section("evaluation", config_args.config)

    if not data_defaults.get("dataset_out"):
        raise SystemExit("dataset_generation.dataset_out must be set in config/defaults.json.")
    dataset_default = Path(data_defaults["dataset_out"])
    adapter_default = alignment_defaults.get("adapter")
    if adapter_default:
        adapter_default_path = Path(adapter_default)
    else:
        output_dir = training_defaults.get("output_dir")
        if not output_dir:
            raise SystemExit("training.output_dir must be set in config/defaults.json.")
        adapter_default_path = Path(output_dir) / "checkpoint-final"
    base_model_default = str(training_defaults.get("base_model", "unsloth/gemma-3-270m-it"))
    max_seq_length_default = int(training_defaults.get("max_seq_length", 2048))
    load_in_4bit_default = bool(training_defaults.get("load_in_4bit", True))
    eval_results_default = evaluation_defaults.get("eval_results") or alignment_defaults.get("eval_results")
    eval_summary_default = evaluation_defaults.get("eval_summary") or alignment_defaults.get("eval_summary")
    eval_checkpoint_path_default = evaluation_defaults.get("checkpoint_path")
    if not eval_checkpoint_path_default and eval_results_default:
        eval_checkpoint_path_default = str(Path(eval_results_default).parent / "evaluation_checkpoint.json")
    eval_checkpoint_every_default = int(evaluation_defaults.get("checkpoint_every", 25) or 25)
    eval_max_samples_default = int(evaluation_defaults.get("max_samples", 200) or 200)
    eval_max_new_tokens_default = int(evaluation_defaults.get("max_new_tokens", 256) or 256)
    eval_temperature_default = float(evaluation_defaults.get("temperature", 0.2) or 0.2)
    eval_top_p_default = float(evaluation_defaults.get("top_p", 0.9) or 0.9)
    eval_include_invalid_default = bool(evaluation_defaults.get("include_invalid", False))
    sample_seed_default = int(
        evaluation_defaults.get("sample_seed", training_defaults.get("seed", 3407)) or training_defaults.get("seed", 3407)
    )
    teacher_retries_default = int(evaluation_defaults.get("teacher_retries", 3) or 3)
    teacher_retry_wait_default = float(evaluation_defaults.get("teacher_retry_wait", 1.0) or 1.0)
    if not eval_results_default or not eval_summary_default:
        raise SystemExit("evaluation.eval_results/eval_summary (or alignment overrides) must be set in config/defaults.json.")
    eval_results_default_path = Path(eval_results_default)
    eval_summary_default_path = Path(eval_summary_default)

    parser = argparse.ArgumentParser(
        description="Evaluate student LoRA adapters on held-out samples.", parents=[config_parser]
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_default,
        help="JSONL produced by generate_dataset.py.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=adapter_default_path,
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=base_model_default,
        help="Base model id used during training.",
    )
    parser.add_argument("--max-seq-length", type=int, default=max_seq_length_default, help="Max sequence length.")
    parser.add_argument("--max-new-tokens", type=int, default=eval_max_new_tokens_default, help="Max tokens to generate.")
    parser.add_argument("--max-samples", type=int, default=eval_max_samples_default, help="How many rows to evaluate.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=sample_seed_default,
        help="Seed for shuffled sampling when capping with --max-samples (ensures diverse subset).",
    )
    parser.add_argument(
        "--include-invalid",
        action=argparse.BooleanOptionalAction,
        default=eval_include_invalid_default,
        help="Include invalid targets during eval.",
    )
    parser.add_argument("--temperature", type=float, default=eval_temperature_default, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=eval_top_p_default, help="Generation top-p.")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=load_in_4bit_default,
        help="Load student in 4-bit for eval (default: on).",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=eval_results_default_path,
        help="Where to write per-sample evaluation JSONL (default: alignment.eval_results from config).",
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=eval_summary_default_path,
        help="Where to write evaluation metrics JSON (default: alignment.eval_summary or alongside --eval-results).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(eval_checkpoint_path_default) if eval_checkpoint_path_default else None,
        help="Where to store progress for resuming evaluation. Disabled if not set.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=eval_checkpoint_every_default,
        help="Save checkpoint every N samples (requires --checkpoint-path).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume evaluation from an existing checkpoint.",
    )
    parser.add_argument(
        "--teacher-retries",
        type=int,
        default=teacher_retries_default,
        help="How many times to retry teacher calls when parsing fails.",
    )
    parser.add_argument(
        "--teacher-retry-wait",
        type=float,
        default=teacher_retry_wait_default,
        help="Seconds to wait between teacher retries.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Deprecated: directory to place evaluation outputs (overrides --eval-results parent).",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=default_teacher,
        help="Ollama model to compare against (teacher baseline). Defaults to model from LLM_MODEL env or built-in default.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print progress every N samples.",
    )
    args = parser.parse_args(remaining)
    args.config = config_args.config
    args.eval_results_default = eval_results_default_path
    args.eval_summary_default = eval_summary_default_path
    args.eval_checkpoint_default = Path(eval_checkpoint_path_default) if eval_checkpoint_path_default else None
    return args


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


def call_teacher_json(prompt: str, model: str, retries: int, wait: float, label: str = "teacher") -> Dict[str, Any]:
    """Call the teacher and retry if parsing fails; retries=-1 means infinite."""
    last_error: Exception | None = None
    attempt = 0
    max_label = "∞" if retries < 0 else str(max(retries, 1))
    while True:
        attempt += 1
        try:
            answer = call_teacher(prompt, model)
            return extract_json_block(answer)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[eval][{label}] attempt {attempt}/{max_label} failed: {exc}")
            if retries >= 0 and attempt >= max(retries, 1):
                break
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def build_prompt(schema_json: str, query: str, current_date: str) -> str:
    return PROMPT_TEMPLATE.format(schema_json=schema_json, query=query, current_date=current_date or "N/A")


def grade_with_teacher(
    schema_json: str,
    query: str,
    student_ast: Dict[str, Any],
    model: str,
    retries: int,
    wait: float,
) -> Tuple[bool, str]:
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
- If the query cannot be satisfied with the available fields/operators, a refusal object with a reason and suggestion is the correct response.
- If the candidate AST is a refusal but the query is answerable with the schema, mark it as not good.
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
    graded = call_teacher_json(prompt, model, retries=retries, wait=wait, label="grade")
    is_good = bool(graded.get("is_good"))
    reason = str(graded.get("reason", "") or "")
    return is_good, reason


def semantic_signature(ast: Dict[str, Any]) -> str:
    """Canonicalize AST semantics (ignores prompt/name/description, sorts conditions)."""
    if "refusal" in ast:
        return canonical_json({"refusal": True})
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
    teacher_retries: int = 3,
    teacher_retry_wait: float = 1.0,
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
            teacher_json = call_teacher_json(
                prompt_text,
                teacher_model,
                retries=teacher_retries,
                wait=teacher_retry_wait,
            )
            result["teacher_canonical"] = canonical_json(teacher_json)
            result["matches_teacher"] = result["student_canonical"] == result["teacher_canonical"]
            result["semantic_match_teacher"] = semantic_signature(parsed) == semantic_signature(teacher_json)
            if result.get("schema_valid"):
                teacher_ok, reason = grade_with_teacher(
                    sample["schema_json"],
                    sample["query"],
                    parsed,
                    teacher_model,
                    retries=teacher_retries,
                    wait=teacher_retry_wait,
                )
                result["teacher_verdict"] = teacher_ok
                result["teacher_reason"] = reason
        except Exception as exc:  # noqa: BLE001
            result["matches_teacher"] = False
            result["semantic_match_teacher"] = False
            result["teacher_error"] = str(exc)

    # Even when the student output is schema-invalid, keep a teacher verdict so alignment can learn from it.
    if teacher_model and "teacher_verdict" not in result and result.get("teacher_canonical"):
        result["teacher_verdict"] = False
        result.setdefault("teacher_reason", "schema invalid")

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


def read_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to read checkpoint at {path}: {exc}")


def write_checkpoint(
    path: Path,
    next_sample_index: int,
    eval_results: Path,
    eval_summary: Path,
    dataset_path: Path,
    total_samples: int,
    results_written: int,
    adapter: Path,
    teacher_model: str,
    sample_seed: int,
    include_invalid: bool,
) -> None:
    payload = {
        "next_sample_index": next_sample_index,
        "results_written": results_written,
        "eval_results": str(eval_results),
        "eval_summary": str(eval_summary),
        "dataset": str(dataset_path),
        "total_samples": total_samples,
        "adapter": str(adapter),
        "teacher_model": teacher_model,
        "sample_seed": sample_seed,
        "include_invalid": include_invalid,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1 when checkpointing is enabled.")
    if not args.teacher_model:
        raise SystemExit("Teacher model is required; set LLM_MODEL or pass --teacher-model to an Ollama model id.")

    if args.out_dir:
        eval_results_path = args.out_dir / "evaluation_results.jsonl"
        eval_summary_path = args.out_dir / "evaluation_summary.json"
    else:
        eval_results_path = args.eval_results
        eval_summary_path = args.eval_summary
        if (
            args.eval_summary == args.eval_summary_default
            and args.eval_results != args.eval_results_default
        ):
            eval_summary_path = args.eval_results.parent / "evaluation_summary.json"
    out_dir = eval_results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    derived_checkpoint = eval_results_path.parent / "evaluation_checkpoint.json"
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = derived_checkpoint
    elif args.eval_checkpoint_default and checkpoint_path == args.eval_checkpoint_default and args.eval_results != args.eval_results_default:
        checkpoint_path = derived_checkpoint
    args.checkpoint_path = checkpoint_path

    start_index = 1
    results_written = 0
    checkpoint: Dict[str, Any] = {}

    if args.resume:
        if not args.checkpoint_path or not args.checkpoint_path.exists():
            raise SystemExit(f"Checkpoint file not found: {args.checkpoint_path}")
        checkpoint = read_checkpoint(args.checkpoint_path)
        start_index = int(checkpoint.get("next_sample_index", 1))
        results_written = int(checkpoint.get("results_written", 0))
        checkpoint_results = checkpoint.get("eval_results")
        if checkpoint_results and Path(checkpoint_results) != eval_results_path:
            print(f"[warn] Checkpoint expects eval_results at {checkpoint_results}; using {eval_results_path}")
        checkpoint_dataset = checkpoint.get("dataset")
        if checkpoint_dataset and Path(checkpoint_dataset) != args.dataset:
            print(f"[warn] Checkpoint was created with dataset {checkpoint_dataset}; current --dataset is {args.dataset}")
        if not eval_results_path.exists():
            raise SystemExit(f"Expected eval results at {eval_results_path} when resuming, but the file is missing.")
        existing_lines = sum(1 for line in eval_results_path.open("r", encoding="utf-8") if line.strip())
        if existing_lines != results_written:
            print(
                f"[warn] Checkpoint recorded {results_written} results, "
                f"but found {existing_lines} lines in {eval_results_path}; using file count."
            )
            results_written = existing_lines
        print(f"[resume] Continuing from sample {start_index} with {results_written} results already written.")
    else:
        if eval_results_path.exists():
            print(f"[info] overwriting existing results at {eval_results_path} (pass --resume to continue).")
        eval_results_path.unlink(missing_ok=True)
        eval_summary_path.unlink(missing_ok=True)
        if args.checkpoint_path:
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if args.checkpoint_path.exists():
                print(f"[info] removing existing checkpoint at {args.checkpoint_path} (pass --resume to continue it).")
            args.checkpoint_path.unlink(missing_ok=True)

    # Explicit schema avoids Arrow inferring null-only columns (e.g., validation_error)
    features = Features(
        {
            "schema_id": Value("string"),
            "domain": Value("string"),
            "operators": Sequence(Value("string")),
            "schema_json": Value("string"),
            "query": Value("string"),
            "current_date": Value("string"),
            "ast_json": Value("string"),
            "is_valid": Value("bool"),
            "validation_error": Value("string"),
            "error_type": Value("string"),
        }
    )
    dataset = load_dataset("json", data_files=str(args.dataset), split="train", features=features)
    if not args.include_invalid:
        dataset = dataset.filter(lambda x: bool(x.get("is_valid", True)))
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.sample_seed).select(range(args.max_samples))
    total_samples = len(dataset)
    checkpoint_total = checkpoint.get("total_samples") if checkpoint else None
    if checkpoint_total and int(checkpoint_total) != total_samples:
        print(f"[warn] Checkpoint expected {checkpoint_total} samples; current dataset has {total_samples}.")
    if not args.teacher_model:
        print("[eval] teacher disabled (no --teacher-model provided and no LLM_MODEL env)")
    print(
        f"[eval] samples={len(dataset)} 4bit={args.load_in_4bit} "
        f"teacher={args.teacher_model or 'none'} log_every={args.log_every}"
    )

    if start_index > total_samples:
        print(f"[eval] Checkpoint covers all {total_samples} samples; rebuilding summary only.")
        if args.checkpoint_path:
            args.checkpoint_path.unlink(missing_ok=True)
        all_results = load_jsonl(eval_results_path) if eval_results_path.exists() else []
        summary = summarize(all_results)
        eval_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    model, tokenizer = load_student(args.base_model, args.adapter, args.max_seq_length, args.load_in_4bit)
    with eval_results_path.open("a", encoding="utf-8") as handle:
        for idx, sample in enumerate(dataset, start=1):
            if idx < start_index:
                continue
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
                teacher_retries=args.teacher_retries,
                teacher_retry_wait=args.teacher_retry_wait,
            )
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
            handle.flush()
            results_written += 1
            if args.log_every and idx % args.log_every == 0:
                print(f"[eval] processed {idx}/{len(dataset)}")
            if args.checkpoint_path and (idx % args.checkpoint_every == 0 or idx == total_samples):
                write_checkpoint(
                    args.checkpoint_path,
                    next_sample_index=idx + 1,
                    eval_results=eval_results_path,
                    eval_summary=eval_summary_path,
                    dataset_path=args.dataset,
                    total_samples=total_samples,
                    results_written=results_written,
                    adapter=args.adapter,
                    teacher_model=args.teacher_model,
                    sample_seed=args.sample_seed,
                    include_invalid=args.include_invalid,
                )

            if args.checkpoint_path:
                args.checkpoint_path.unlink(missing_ok=True)
    all_results = load_jsonl(eval_results_path) if eval_results_path.exists() else []
    summary = summarize(all_results)
    eval_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
