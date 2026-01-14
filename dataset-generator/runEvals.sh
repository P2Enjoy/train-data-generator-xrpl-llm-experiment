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
ORIGINAL_ARGS=("$@")

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
reporting = defaults.get("reporting", {}) or {}

adapter = alignment.get("adapter")
if not adapter:
    output_dir = training.get("output_dir", "outputs/student_runs/gemma3-270m")
    adapter = str(Path(output_dir) / "checkpoint-final")

eval_results = alignment.get("eval_results") or "outputs/student_runs/eval/evaluation_results.jsonl"
eval_summary = alignment.get("eval_summary") or str(Path(eval_results).parent / "evaluation_summary.json")

aligned_out_dir = alignment.get("output_dir", "")
aligned_adapter = str(Path(aligned_out_dir) / "checkpoint-final") if aligned_out_dir else ""
aligned_eval_results = alignment.get("aligned_eval_results") or ""
aligned_eval_summary = alignment.get("aligned_eval_summary") or ""
if aligned_eval_results and not aligned_eval_summary:
    aligned_eval_summary = str(Path(aligned_eval_results).parent / "evaluation_summary.json")

reports_dir = reporting.get("reports_dir", "outputs/reports")

print(adapter)
print(eval_results)
print(eval_summary)
print(aligned_adapter)
print(aligned_eval_results)
print(aligned_eval_summary)
print(reports_dir)
PY
  )"
  IFS=$'\n' read -r CONFIG_ADAPTER CONFIG_EVAL_RESULTS CONFIG_EVAL_SUMMARY CONFIG_ALIGNED_ADAPTER CONFIG_ALIGNED_EVAL_RESULTS CONFIG_ALIGNED_EVAL_SUMMARY CONFIG_REPORTS_DIR <<< "${config_defaults}"
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

HAS_CHECKPOINT_ARG=0
for arg in "${EVAL_ARGS[@]}"; do
  if [[ "$arg" == "--checkpoint-path" ]]; then
    HAS_CHECKPOINT_ARG=1
    break
  fi
done

if [[ ${HAS_CHECKPOINT_ARG} -eq 0 ]]; then
  EVAL_CHECKPOINT="$(dirname "${EVAL_RESULTS}")/evaluation_checkpoint.json"
  EVAL_ARGS+=("--checkpoint-path" "${EVAL_CHECKPOINT}")
  if [[ -f "${EVAL_CHECKPOINT}" ]]; then
    echo "Found eval checkpoint at ${EVAL_CHECKPOINT}; resuming."
    EVAL_ARGS+=("--resume")
  fi
fi

TRAIN_DIR="$(cd "$(dirname "$ADAPTER")" && pwd)"
ALIGNED_ADAPTER="${CONFIG_ALIGNED_ADAPTER:-}"
ALIGNED_EVAL_RESULTS="${CONFIG_ALIGNED_EVAL_RESULTS:-}"
ALIGNED_EVAL_SUMMARY="${CONFIG_ALIGNED_EVAL_SUMMARY:-}"
REPORTS_DIR="${CONFIG_REPORTS_DIR:-outputs/reports}"

if [[ -z "${ALIGNED_EVAL_RESULTS}" && -n "${ALIGNED_ADAPTER}" ]]; then
  ALIGNED_EVAL_RESULTS="$(dirname "${ALIGNED_ADAPTER}")/../eval-dpo/evaluation_results.jsonl"
fi
if [[ -z "${ALIGNED_EVAL_SUMMARY}" && -n "${ALIGNED_EVAL_RESULTS}" ]]; then
  ALIGNED_EVAL_SUMMARY="$(dirname "${ALIGNED_EVAL_RESULTS}")/evaluation_summary.json"
fi

# Filter user args to reuse for aligned eval (drop adapter/output overrides)
COMMON_ARGS=()
idx=0
while [[ ${idx} -lt ${#ORIGINAL_ARGS[@]} ]]; do
  arg="${ORIGINAL_ARGS[$idx]}"
  case "$arg" in
    --adapter|--out-dir|--eval-results|--eval-summary)
      ((idx+=2))
      continue
      ;;
  esac
  COMMON_ARGS+=("$arg")
  ((idx++))
done

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
  --out-dir "${REPORTS_DIR}"

echo "Evaluation + report complete."

if [[ -n "${ALIGNED_ADAPTER}" && -d "${ALIGNED_ADAPTER}" ]]; then
  echo "Found alignment adapter at ${ALIGNED_ADAPTER}; running aligned evaluation..."
  if [[ -f "${ALIGNED_EVAL_RESULTS}" && -f "${ALIGNED_EVAL_SUMMARY}" ]]; then
    echo "Skipping aligned evaluation; outputs already exist:"
    echo "  ${ALIGNED_EVAL_RESULTS}"
    echo "  ${ALIGNED_EVAL_SUMMARY}"
  else
    HAS_ALIGNED_CHECKPOINT=0
    for arg in "${COMMON_ARGS[@]}"; do
      if [[ "$arg" == "--checkpoint-path" ]]; then
        HAS_ALIGNED_CHECKPOINT=1
        break
      fi
    done
    ALIGNED_ARGS=()
    if [[ ${HAS_ALIGNED_CHECKPOINT} -eq 0 ]]; then
      ALIGNED_EVAL_CHECKPOINT="$(dirname "${ALIGNED_EVAL_RESULTS}")/evaluation_checkpoint.json"
      ALIGNED_ARGS+=("--checkpoint-path" "${ALIGNED_EVAL_CHECKPOINT}")
      if [[ -f "${ALIGNED_EVAL_CHECKPOINT}" ]]; then
        echo "Found aligned eval checkpoint at ${ALIGNED_EVAL_CHECKPOINT}; resuming."
        ALIGNED_ARGS+=("--resume")
      fi
    fi
    PYTHONUNBUFFERED=1 uv run python -u scripts/evaluate_student.py \
      --adapter "${ALIGNED_ADAPTER}" \
      --eval-results "${ALIGNED_EVAL_RESULTS}" \
      --eval-summary "${ALIGNED_EVAL_SUMMARY}" \
      "${ALIGNED_ARGS[@]}" \
      "${COMMON_ARGS[@]}"
  fi

  echo "Building aligned report..."
  PYTHONUNBUFFERED=1 uv run python -u scripts/report_training.py \
    --training-metrics "${ALIGNED_ADAPTER}/../training_metrics.jsonl" \
    --eval-summary "${ALIGNED_EVAL_SUMMARY}" \
    --eval-results "${ALIGNED_EVAL_RESULTS}" \
    --run-config "${ALIGNED_ADAPTER}/../run_config.json" \
    --out-dir "${REPORTS_DIR}" \
    --report-name alignment_report.md
  echo "Aligned evaluation + report complete."
else
  echo "No alignment adapter found (expected at ${ALIGNED_ADAPTER:-<unset>}); skipping aligned evaluation."
fi
