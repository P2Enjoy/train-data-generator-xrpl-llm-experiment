#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

usage() {
  cat <<EOF
Usage: $0 [evaluate_student.py args...]

Runs evaluation then builds a report. To override defaults, pass args understood by
scripts/evaluate_student.py (e.g., --adapter, --dataset, --teacher-model, --out-dir).
EOF
}

EVAL_ARGS=()
EVAL_OUT="outputs/student_runs/eval"
ADAPTER="outputs/student_runs/gemma3-270m/checkpoint-final"

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --out-dir)
        EVAL_OUT="$2"
        EVAL_ARGS+=("$1" "$2")
        shift 2
        ;;
      --adapter)
        ADAPTER="$2"
        EVAL_ARGS+=("$1" "$2")
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        EVAL_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

parse_args "$@"

TRAIN_DIR="$(cd "$(dirname "$ADAPTER")" && pwd)"

echo "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Installing / syncing dependencies with uv..."
uv sync

echo "Running evaluation..."
uv run python scripts/evaluate_student.py "${EVAL_ARGS[@]}"

echo "Building report..."
uv run python scripts/report_training.py \
  --training-metrics "${TRAIN_DIR}/training_metrics.jsonl" \
  --eval-summary "${EVAL_OUT}/evaluation_summary.json" \
  --eval-results "${EVAL_OUT}/evaluation_results.jsonl" \
  --run-config "${TRAIN_DIR}/run_config.json" \
  --out-dir outputs/reports

echo "Evaluation + report complete."
