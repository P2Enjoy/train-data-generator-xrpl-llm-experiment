"""Build a lightweight Markdown report with plots for student training/eval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.config import DEFAULT_CONFIG_PATH, load_section
from lib.io import load_jsonl


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to defaults JSON (config/defaults.json).",
    )
    config_args, remaining = config_parser.parse_known_args()

    training_defaults = load_section("training", config_args.config)
    alignment_defaults = load_section("alignment", config_args.config)
    evaluation_defaults = load_section("evaluation", config_args.config)
    reporting_defaults = load_section("reporting", config_args.config)

    training_output_dir = training_defaults.get("output_dir")
    eval_results_default = evaluation_defaults.get("eval_results") or alignment_defaults.get("eval_results")
    eval_summary_default = evaluation_defaults.get("eval_summary") or alignment_defaults.get("eval_summary")
    reports_dir_default = reporting_defaults.get("reports_dir")

    if not training_output_dir:
        raise SystemExit("training.output_dir must be set in config/defaults.json.")
    if not eval_results_default or not eval_summary_default:
        raise SystemExit("evaluation.eval_results/eval_summary (or alignment overrides) must be set in config/defaults.json.")
    if not reports_dir_default:
        raise SystemExit("reporting.reports_dir must be set in config/defaults.json.")

    parser = argparse.ArgumentParser(description="Generate training/eval report.", parents=[config_parser])
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path(training_output_dir) / "training_metrics.jsonl",
        help="JSONL produced by train_student_unsloth.py callback.",
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=Path(eval_summary_default),
        help="Summary JSON produced by evaluate_student.py.",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=Path(eval_results_default),
        help="Per-sample JSONL produced by evaluate_student.py.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path(training_output_dir) / "run_config.json",
        help="Config emitted by train_student_unsloth.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(reports_dir_default),
        help="Directory to write plots and report to.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="student_training_report.md",
        help="Filename for the generated Markdown report.",
    )
    return parser.parse_args(remaining)


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
    handles, labels = ax.get_legend_handles_labels()
    if labels:
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
    if df.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    def label_outcome(row: pd.Series) -> str:
        parsed = bool(row.get("parsed"))
        schema_ok = bool(row.get("schema_valid"))
        verdict = row.get("teacher_verdict")
        if not parsed:
            return "Parse failed"
        if not schema_ok:
            return "Schema invalid"
        if verdict is True:
            return "Teacher accepted"
        if verdict is False:
            return "Teacher rejected"
        return "No teacher verdict"

    plot_df = df.copy()
    plot_df["outcome"] = plot_df.apply(label_outcome, axis=1)

    order = [
        label
        for label in (
            "Teacher accepted",
            "Teacher rejected",
            "Schema invalid",
            "Parse failed",
            "No teacher verdict",
        )
        if label in set(plot_df["outcome"])
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=plot_df, x="outcome", order=order or None, ax=ax, color="tab:blue")
    ax.set_xlabel("Outcome (teacher + schema)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    hist_path = out_dir / "eval_outcomes.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    return hist_path


def plot_rate_bars(metrics: Dict[str, float], out_dir: Path, title: str, filename: str) -> Path | None:
    if not metrics:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(metrics.keys())
    values = list(metrics.values())
    sns.barplot(x=values, y=names, ax=ax, orient="h", color="tab:blue")
    ax.set_xlabel("Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    for idx, val in enumerate(values):
        ax.text(val + 0.01, idx, f"{val:.1%}", va="center")
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_teacher_confusion(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if df.empty or "teacher_verdict" not in df or "schema_valid" not in df:
        return None
    plot_df = df.copy()
    plot_df["teacher_verdict"] = plot_df["teacher_verdict"].fillna(False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=plot_df, x="teacher_verdict", hue="schema_valid", ax=ax)
    ax.set_xlabel("Teacher accepted")
    ax.set_ylabel("Count")
    ax.legend(title="Schema valid")
    path = out_dir / "teacher_vs_schema.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


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
    rate_plot: Path | None = None
    teacher_plot: Path | None = None

    # Build metric bar plot if we have summary
    if eval_summary:
        key_rates = {
            "Parsed": eval_summary.get("parsed_rate", 0.0),
            "Schema valid": eval_summary.get("schema_valid_rate", 0.0),
        }
        if "teacher_accept_rate" in eval_summary:
            key_rates["Teacher accept"] = eval_summary.get("teacher_accept_rate", 0.0)
        if "semantic_pass_rate" in eval_summary:
            key_rates["Semantic pass"] = eval_summary.get("semantic_pass_rate", 0.0)
        if "teacher_agreement_rate" in eval_summary:
            key_rates["Teacher agreement"] = eval_summary.get("teacher_agreement_rate", 0.0)
        if "exact_match_rate" in eval_summary:
            key_rates["Exact match"] = eval_summary.get("exact_match_rate", 0.0)
        rate_plot = plot_rate_bars(key_rates, plot_dir, "Evaluation rates", "eval_rates.png")

    # Teacher vs schema confusion plot
    teacher_plot = plot_teacher_confusion(eval_df, plot_dir) if not eval_df.empty else None

    lines: List[str] = []
    lines.append("# Gemma 3 270M Student Report")
    if run_config:
        lines.append("")
        lines.append("## Run configuration")
        lines.append(f"- Base model: `{run_config.get('base_model', 'unspecified')}`")
        lines.append(f"- Samples: train={run_config.get('train_samples')} eval={run_config.get('eval_samples')}")
        lines.append(f"- Max seq length: {run_config.get('max_seq_length')}")
        lines.append(f"- LoRA: r={run_config.get('lora_r')} alpha={run_config.get('lora_alpha')}")
        lines.append(
            f"- Batch size: {run_config.get('batch_size')} x grad_accum={run_config.get('grad_accum')} "
            f"(effective={int(run_config.get('batch_size', 0) or 0) * int(run_config.get('grad_accum', 0) or 0)})"
        )
        if run_config.get("max_steps") and int(run_config.get("max_steps")) > 0:
            lines.append(f"- Max steps: {run_config.get('max_steps')}")
        else:
            lines.append(f"- Epochs: {run_config.get('num_epochs')}")

    if eval_summary:
        lines.append("")
        lines.append("## Evaluation KPIs")
        lines.append(f"- Parsed: {eval_summary.get('parsed_rate', 0):.1%}")
        lines.append(f"- Schema valid: {eval_summary.get('schema_valid_rate', 0):.1%}")
        if "teacher_accept_rate" in eval_summary:
            lines.append(f"- Teacher acceptance: {eval_summary.get('teacher_accept_rate', 0):.1%}")
        if "semantic_pass_rate" in eval_summary:
            lines.append(f"- Semantic pass (schema valid + teacher OK): {eval_summary.get('semantic_pass_rate', 0):.1%}")
        if "teacher_agreement_rate" in eval_summary:
            lines.append(f"- Teacher agreement (canonical): {eval_summary.get('teacher_agreement_rate', 0):.1%}")
        if "semantic_match_target_rate" in eval_summary:
            lines.append(f"- Semantic match to target: {eval_summary.get('semantic_match_target_rate', 0):.1%}")
        if "semantic_match_teacher_rate" in eval_summary:
            lines.append(f"- Semantic match to teacher: {eval_summary.get('semantic_match_teacher_rate', 0):.1%}")
        if "exact_match_rate" in eval_summary:
            lines.append(f"- Exact match: {eval_summary.get('exact_match_rate', 0):.1%}")

    if train_plots:
        lines.append("")
        lines.append("## Training curves")
        for path in train_plots:
            lines.append(f"- {path}")

    if eval_plot:
        lines.append("")
        lines.append("## Eval distribution")
        lines.append(f"- {eval_plot}")
    if rate_plot:
        lines.append("")
        lines.append("## Eval rates")
        lines.append(f"- {rate_plot}")
    if teacher_plot:
        lines.append("")
        lines.append("## Teacher vs schema")
        lines.append(f"- {teacher_plot}")

    report_path = args.out_dir / args.report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
