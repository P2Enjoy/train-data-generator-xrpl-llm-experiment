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
  python3 - <<'PY' "$CONFIG_PATH" "$1"
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
section = cfg.get("alignment", {})
print(section.get(sys.argv[2], ""))
PY
}

ADAPTER_DEFAULT="$(default_from_config "adapter")"
EVAL_RESULTS_DEFAULT="$(default_from_config "eval_results")"
PAIRS_OUT_DEFAULT="$(default_from_config "pairs_out")"

ADAPTER="${ADAPTER_OVERRIDE:-$ADAPTER_DEFAULT}"

echo "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Installing / syncing dependencies with uv..."
uv sync

echo "Running teacher evaluation (required for alignment pairs)..."
PYTHONUNBUFFERED=1 uv run python -u scripts/evaluate_student.py --adapter "${ADAPTER}" "${TEACHER_ARG[@]}"

echo "Building preference pairs..."
PYTHONUNBUFFERED=1 uv run python -u scripts/build_alignment_pairs.py --config "${CONFIG_PATH}" \
  --eval-results "${EVAL_RESULTS_DEFAULT}" \
  --out "${PAIRS_OUT_DEFAULT}"

echo "Starting alignment (DPO)..."
PYTHONUNBUFFERED=1 uv run python -u scripts/train_alignment_dpo.py --config "${CONFIG_PATH}" \
  --pairs "${PAIRS_OUT_DEFAULT}" \
  --adapter "${ADAPTER}" \
  "${DPO_ARGS[@]}"

echo "Alignment complete."
