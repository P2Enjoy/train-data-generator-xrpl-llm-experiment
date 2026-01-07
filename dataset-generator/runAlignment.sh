#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="config/defaults.json"
ADAPTER_OVERRIDE=""
TEACHER_ARG=()
DPO_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [--config PATH] [--adapter DIR] [--teacher-model MODEL] [train_alignment_dpo args...]

Runs teacher-evaluated alignment:
  1) evaluate_student.py (uses the Ollama teacher)
  2) build_alignment_pairs.py (requires teacher verdicts)
  3) train_alignment_dpo.py (DPO/ORPO-style alignment)
  4) evaluate_student.py on the aligned adapter (writes to dedicated outputs)

Any extra args (after the known options) are forwarded to train_alignment_dpo.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --adapter)
      ADAPTER_OVERRIDE="$2"
      shift 2
      ;;
    --teacher-model)
      TEACHER_ARG=(--teacher-model "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      DPO_ARGS+=("$1")
      shift
      ;;
  esac
done

default_from_config() {
  python3 - <<'PY' "$CONFIG_PATH" "$1" "$2"
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
section = cfg.get(sys.argv[2], {})
print(section.get(sys.argv[3], ""))
PY
}

ADAPTER_DEFAULT="$(default_from_config "alignment" "adapter")"
EVAL_RESULTS_DEFAULT="$(default_from_config "alignment" "eval_results")"
EVAL_SUMMARY_DEFAULT="$(default_from_config "alignment" "eval_summary")"
PAIRS_OUT_DEFAULT="$(default_from_config "alignment" "pairs_out")"
ALIGNED_EVAL_RESULTS_DEFAULT="$(default_from_config "alignment" "aligned_eval_results")"
ALIGNED_EVAL_SUMMARY_DEFAULT="$(default_from_config "alignment" "aligned_eval_summary")"
ALIGN_OUTPUT_DIR_DEFAULT="$(default_from_config "alignment" "output_dir")"

if [[ -z "${EVAL_RESULTS_DEFAULT}" ]]; then
  EVAL_RESULTS_DEFAULT="outputs/student_runs/eval/evaluation_results.jsonl"
fi
if [[ -z "${EVAL_SUMMARY_DEFAULT}" ]]; then
  EVAL_SUMMARY_DEFAULT="$(dirname "${EVAL_RESULTS_DEFAULT}")/evaluation_summary.json"
fi
if [[ -z "${ALIGNED_EVAL_RESULTS_DEFAULT}" || -z "${ALIGNED_EVAL_SUMMARY_DEFAULT}" ]]; then
  echo "Aligned eval outputs must be set in config (alignment.aligned_eval_results / alignment.aligned_eval_summary)."
  exit 1
fi
if [[ -z "${ALIGN_OUTPUT_DIR_DEFAULT}" ]]; then
  echo "alignment.output_dir must be set in config."
  exit 1
fi

ADAPTER="${ADAPTER_OVERRIDE:-$ADAPTER_DEFAULT}"
ALIGNED_ADAPTER="${ALIGN_OUTPUT_DIR_DEFAULT%/}/checkpoint-final"

echo "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Installing / syncing dependencies with uv..."
uv sync

if [[ -f "${EVAL_RESULTS_DEFAULT}" && -f "${EVAL_SUMMARY_DEFAULT}" ]]; then
  echo "Skipping evaluation; outputs already exist:"
  echo "  ${EVAL_RESULTS_DEFAULT}"
  echo "  ${EVAL_SUMMARY_DEFAULT}"
else
  echo "Running teacher evaluation (required for alignment pairs)..."
  PYTHONUNBUFFERED=1 uv run python -u scripts/evaluate_student.py \
    --config "${CONFIG_PATH}" \
    --adapter "${ADAPTER}" \
    --eval-results "${EVAL_RESULTS_DEFAULT}" \
    --eval-summary "${EVAL_SUMMARY_DEFAULT}" \
    "${TEACHER_ARG[@]}"
fi

echo "Building preference pairs..."
PYTHONUNBUFFERED=1 uv run python -u scripts/build_alignment_pairs.py --config "${CONFIG_PATH}" \
  --eval-results "${EVAL_RESULTS_DEFAULT}" \
  --out "${PAIRS_OUT_DEFAULT}"

echo "Starting alignment (DPO)..."
PYTHONUNBUFFERED=1 uv run python -u scripts/train_alignment_dpo.py --config "${CONFIG_PATH}" \
  --pairs "${PAIRS_OUT_DEFAULT}" \
  --adapter "${ADAPTER}" \
  "${DPO_ARGS[@]}"

if [[ -f "${ALIGNED_EVAL_RESULTS_DEFAULT}" && -f "${ALIGNED_EVAL_SUMMARY_DEFAULT}" ]]; then
  echo "Skipping post-alignment evaluation; outputs already exist:"
  echo "  ${ALIGNED_EVAL_RESULTS_DEFAULT}"
  echo "  ${ALIGNED_EVAL_SUMMARY_DEFAULT}"
else
  echo "Running post-alignment evaluation on aligned adapter..."
  PYTHONUNBUFFERED=1 uv run python -u scripts/evaluate_student.py \
    --config "${CONFIG_PATH}" \
    --adapter "${ALIGNED_ADAPTER}" \
    --eval-results "${ALIGNED_EVAL_RESULTS_DEFAULT}" \
    --eval-summary "${ALIGNED_EVAL_SUMMARY_DEFAULT}" \
    "${TEACHER_ARG[@]}"
fi

echo "Alignment complete."
