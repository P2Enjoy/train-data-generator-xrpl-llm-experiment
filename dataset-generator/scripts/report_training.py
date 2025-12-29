"""Build a lightweight Markdown report with plots for student training/eval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training/eval report.")
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path("outputs/student_runs/gemma3-270m/training_metrics.jsonl"),
        help="JSONL produced by train_student_unsloth.py callback.",
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=Path("outputs/student_runs/eval/evaluation_summary.json"),
        help="Summary JSON produced by evaluate_student.py.",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=Path("outputs/student_runs/eval/evaluation_results.jsonl"),
        help="Per-sample JSONL produced by evaluate_student.py.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("outputs/student_runs/gemma3-270m/run_config.json"),
        help="Config emitted by train_student_unsloth.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory to write plots and report to.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="student_training_report.md",
        help="Filename for the generated Markdown report.",
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


def plot_training_curves(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: List[Path] = []
    if df.empty:
        return plots

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    if "loss" in df:
        ax.plot(df["step"], df["loss"], label="train_loss")
    if "eval_loss" in df:
        ax.plot(df["step"], df["eval_loss"], label="eval_loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    loss_path = out_dir / "training_loss.png"
    fig.tight_layout()
    fig.savefig(loss_path, dpi=200)
    plt.close(fig)
    plots.append(loss_path)

    if "learning_rate" in df:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df["step"], df["learning_rate"], label="learning_rate", color="tab:orange")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        lr_path = out_dir / "learning_rate.png"
        fig.tight_layout()
        fig.savefig(lr_path, dpi=200)
        plt.close(fig)
        plots.append(lr_path)

    return plots


def plot_eval_histogram(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if df.empty or "schema_valid" not in df:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="schema_valid", hue="exact_match", ax=ax)
    ax.set_xlabel("Schema valid")
    ax.set_ylabel("Count")
    hist_path = out_dir / "eval_outcomes.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    return hist_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(load_jsonl(args.training_metrics)) if args.training_metrics.exists() else pd.DataFrame()
    eval_df = pd.DataFrame(load_jsonl(args.eval_results)) if args.eval_results.exists() else pd.DataFrame()

    run_config = {}
    if args.run_config.exists():
        run_config = json.loads(args.run_config.read_text(encoding="utf-8"))

    eval_summary = {}
    if args.eval_summary.exists():
        eval_summary = json.loads(args.eval_summary.read_text(encoding="utf-8"))

    plot_dir = args.out_dir / "plots"
    train_plots = plot_training_curves(train_df, plot_dir) if not train_df.empty else []
    eval_plot = plot_eval_histogram(eval_df, plot_dir)

    lines: List[str] = []
    lines.append("# Gemma 3 270M Student Report")
    if run_config:
        lines.append("")
        lines.append("## Run configuration")
        lines.append(f"- Base model: `{run_config.get('base_model', 'unspecified')}`")
        lines.append(f"- Samples: train={run_config.get('train_samples')} eval={run_config.get('eval_samples')}")
        lines.append(f"- Max seq length: {run_config.get('max_seq_length')}")
        lines.append(f"- LoRA: r={run_config.get('lora_r')} alpha={run_config.get('lora_alpha')}")
        lines.append(f"- Batch size: {run_config.get('batch_size')} x grad_accum={run_config.get('grad_accum')}")

    if eval_summary:
        lines.append("")
        lines.append("## Evaluation KPIs")
        lines.append(f"- Parsed rate: {eval_summary.get('parsed_rate', 0):.3f}")
        lines.append(f"- Schema valid rate: {eval_summary.get('schema_valid_rate', 0):.3f}")
        lines.append(f"- Exact match rate: {eval_summary.get('exact_match_rate', 0):.3f}")
        if "teacher_agreement_rate" in eval_summary:
            lines.append(f"- Teacher agreement: {eval_summary.get('teacher_agreement_rate', 0):.3f}")

    if train_plots:
        lines.append("")
        lines.append("## Training curves")
        for path in train_plots:
            lines.append(f"- {path}")

    if eval_plot:
        lines.append("")
        lines.append("## Eval distribution")
        lines.append(f"- {eval_plot}")

    report_path = args.out_dir / args.report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
