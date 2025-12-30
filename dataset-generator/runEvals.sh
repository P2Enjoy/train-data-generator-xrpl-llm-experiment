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
scripts/evaluate_student.py (e.g., --adapter, --dataset, --teacher-model, --out-dir).
EOF
}

EVAL_ARGS=()
EVAL_OUT="outputs/student_runs/eval"
ADAPTER="outputs/student_runs/gemma3-270m/checkpoint-final"
CONFIG_PATH="config/defaults.json"
ADAPTER_SET=0
OUT_DIR_SET=0

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_PATH="$2"
        EVAL_ARGS+=("$1" "$2")
        shift 2
        ;;
      --out-dir)
        EVAL_OUT="$2"
        EVAL_ARGS+=("$1" "$2")
        OUT_DIR_SET=1
        shift 2
        ;;
      --adapter)
        ADAPTER="$2"
        EVAL_ARGS+=("$1" "$2")
        ADAPTER_SET=1
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

if [[ -n "${CONFIG_PATH}" && ( ${ADAPTER_SET} -eq 0 || ${OUT_DIR_SET} -eq 0 ) ]]; then
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

eval_results = alignment.get("eval_results")
eval_out = str(Path(eval_results).parent) if eval_results else "outputs/student_runs/eval"

print(adapter)
print(eval_out)
PY
  )"
  IFS=$'\n' read -r CONFIG_ADAPTER CONFIG_EVAL_OUT <<< "${config_defaults}"
  if [[ ${ADAPTER_SET} -eq 0 ]]; then
    ADAPTER="${CONFIG_ADAPTER}"
    EVAL_ARGS+=("--adapter" "${ADAPTER}")
  fi
  if [[ ${OUT_DIR_SET} -eq 0 ]]; then
    EVAL_OUT="${CONFIG_EVAL_OUT}"
    EVAL_ARGS+=("--out-dir" "${EVAL_OUT}")
  fi
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
  --eval-summary "${EVAL_OUT}/evaluation_summary.json" \
  --eval-results "${EVAL_OUT}/evaluation_results.jsonl" \
  --run-config "${TRAIN_DIR}/run_config.json" \
  --out-dir outputs/reports

echo "Evaluation + report complete."
