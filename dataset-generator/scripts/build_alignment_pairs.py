"""Build preference pairs for DPO/ORPO from evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import _bootstrap  # noqa: F401
from lib.io import load_jsonl, write_jsonl
from lib.config import DEFAULT_CONFIG_PATH, load_section


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to defaults JSON (config/defaults.json).",
    )
    config_args, remaining = config_parser.parse_known_args()
    defaults = load_section("alignment", config_args.config)

    parser = argparse.ArgumentParser(description="Create chosen/rejected pairs from evaluate_student outputs.", parents=[config_parser])
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=Path(defaults.get("eval_results", "outputs/student_runs/eval/evaluation_results.jsonl")),
        help="Per-sample JSONL produced by evaluate_student.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(defaults.get("pairs_out", "outputs/student_runs/alignment/dpo_pairs.jsonl")),
        help="Where to write the preference pairs.",
    )
    parser.add_argument(
        "--only-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows where the teacher rejected the student (teacher_verdict is False).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on how many pairs to emit.",
    )
    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args


def iter_pairs(rows: Iterable[Dict[str, object]], only_rejected: bool) -> Iterable[Dict[str, str]]:
    for row in rows:
        teacher_ast = row.get("teacher_canonical")
        student_ast = row.get("student_canonical") or row.get("raw_output")
        teacher_verdict = row.get("teacher_verdict")
        prompt = row.get("prompt_text")

        if not (teacher_ast and student_ast and prompt):
            continue
        if only_rejected and teacher_verdict is not False:
            continue
        if teacher_ast == student_ast:
            continue

        yield {
            "prompt": str(prompt),
            "chosen": str(teacher_ast),
            "rejected": str(student_ast),
            "schema_id": str(row.get("schema_id", "")),
            "teacher_verdict": bool(teacher_verdict),
        }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.eval_results)
    if not rows:
        raise SystemExit(f"No eval rows found in {args.eval_results}; run evaluate_student.py with --teacher-model first.")
    if not any("teacher_verdict" in r for r in rows):
        raise SystemExit(
            f"Eval results at {args.eval_results} lack teacher fields; rerun evaluate_student.py with --teacher-model."
        )
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: List[Dict[str, str]] = []
    for pair in iter_pairs(rows, args.only_rejected):
        pairs.append(pair)
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break

    write_jsonl(args.out, pairs)
    print(f"[pairs] wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
