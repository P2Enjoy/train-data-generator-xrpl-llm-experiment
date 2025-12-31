#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export PYTHONUNBUFFERED=1

usage() {
  cat <<EOF
Usage: $0 [--config PATH] [evaluate_student.py args...]

Runs evaluation then builds a report. To override defaults, pass args understood by
scripts/evaluate_student.py (e.g., --adapter, --dataset, --teacher-model, --eval-results).
EOF
}

EVAL_ARGS=()
EVAL_RESULTS=""
EVAL_SUMMARY=""
EVAL_OUT=""
ADAPTER="outputs/student_runs/gemma3-270m/checkpoint-final"
CONFIG_PATH="config/defaults.json"
ADAPTER_SET=0
OUT_DIR_SET=0
EVAL_RESULTS_SET=0

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_PATH="$2"
        EVAL_ARGS+=("$1" "$2")
        shift 2
        ;;
      --adapter)
        ADAPTER="$2"
        EVAL_ARGS+=("$1" "$2")
        ADAPTER_SET=1
        shift 2
        ;;
      --out-dir)
        EVAL_OUT="$2"
        EVAL_ARGS+=("$1" "$2")
        OUT_DIR_SET=1
        shift 2
        ;;
      --eval-results)
        EVAL_RESULTS="$2"
        EVAL_ARGS+=("$1" "$2")
        EVAL_RESULTS_SET=1
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

if [[ -n "${CONFIG_PATH}" && ( ${ADAPTER_SET} -eq 0 || ${OUT_DIR_SET} -eq 0 || ${EVAL_RESULTS_SET} -eq 0 ) ]]; then
  config_defaults="$(python3 - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
defaults = {}
if config_path.exists():
    try:
        defaults = json.loads(config_path.read_text())
    except Exception:
        defaults = {}

training = defaults.get("training", {}) or {}
alignment = defaults.get("alignment", {}) or {}

adapter = alignment.get("adapter")
if not adapter:
    output_dir = training.get("output_dir", "outputs/student_runs/gemma3-270m")
    adapter = str(Path(output_dir) / "checkpoint-final")

eval_results = alignment.get("eval_results") or "outputs/student_runs/eval/evaluation_results.jsonl"
eval_summary = alignment.get("eval_summary") or str(Path(eval_results).parent / "evaluation_summary.json")

print(adapter)
print(eval_results)
print(eval_summary)
PY
  )"
  IFS=$'\n' read -r CONFIG_ADAPTER CONFIG_EVAL_RESULTS CONFIG_EVAL_SUMMARY <<< "${config_defaults}"
  if [[ ${ADAPTER_SET} -eq 0 ]]; then
    ADAPTER="${CONFIG_ADAPTER}"
    EVAL_ARGS+=("--adapter" "${ADAPTER}")
  fi
  if [[ ${OUT_DIR_SET} -eq 0 && ${EVAL_RESULTS_SET} -eq 0 ]]; then
    EVAL_RESULTS="${CONFIG_EVAL_RESULTS}"
  fi
  if [[ -z "${EVAL_SUMMARY}" ]]; then
    EVAL_SUMMARY="${CONFIG_EVAL_SUMMARY}"
  fi
fi

if [[ ${OUT_DIR_SET} -eq 1 ]]; then
  EVAL_RESULTS="${EVAL_OUT}/evaluation_results.jsonl"
  EVAL_SUMMARY="${EVAL_OUT}/evaluation_summary.json"
fi

if [[ -z "${EVAL_RESULTS}" ]]; then
  EVAL_RESULTS="outputs/student_runs/eval/evaluation_results.jsonl"
fi

if [[ ${EVAL_RESULTS_SET} -eq 0 && ${OUT_DIR_SET} -eq 0 ]]; then
  EVAL_ARGS+=("--eval-results" "${EVAL_RESULTS}")
fi

if [[ -z "${EVAL_SUMMARY}" ]]; then
  EVAL_SUMMARY="$(dirname "${EVAL_RESULTS}")/evaluation_summary.json"
fi

TRAIN_DIR="$(cd "$(dirname "$ADAPTER")" && pwd)"

echo "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Installing / syncing dependencies with uv..."
uv sync

echo "Running evaluation..."
PYTHONUNBUFFERED=1 uv run python -u scripts/evaluate_student.py "${EVAL_ARGS[@]}"

echo "Building report..."
PYTHONUNBUFFERED=1 uv run python -u scripts/report_training.py \
  --training-metrics "${TRAIN_DIR}/training_metrics.jsonl" \
  --eval-summary "${EVAL_SUMMARY}" \
  --eval-results "${EVAL_RESULTS}" \
  --run-config "${TRAIN_DIR}/run_config.json" \
  --out-dir outputs/reports

echo "Evaluation + report complete."
