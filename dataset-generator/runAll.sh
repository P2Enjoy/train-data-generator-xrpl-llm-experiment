#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG_DIR="${ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/runAll_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee "${LOG_FILE}") 2>&1

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

log() {
  echo "[$(date '+%F %T')] $*"
}

WITH_TRAINING=0
WITH_EVALS=0
WITH_ALIGNMENT=0
MODEL_ARG=()
CONFIG_PATH="config/defaults.json"

usage() {
  cat <<EOF
Usage: $0 [--model MODEL] [--with-training] [--with-evals] [--with-alignment]

This wraps the per-stage runners:
  - runDatasetGeneration.sh (always)
  - runTraining.sh (when --with-training)
  - runEvals.sh (when --with-evals)
  - runAlignment.sh (when --with-alignment; implies teacher-evaluated pairs + DPO stage)

Pass model override to the dataset generator with --model.
Use the individual runner scripts for finer control over arguments.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_ARG=(--model "$2")
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --with-training)
      WITH_TRAINING=1
      shift
      ;;
    --with-evals)
      WITH_EVALS=1
      shift
      ;;
    --with-alignment)
      WITH_ALIGNMENT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
    exit 1
      ;;
  esac
done

path_info="$(
python3 - "$CONFIG_PATH" <<'PY'
import json
import re
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
defaults = {}
if cfg_path.exists():
    try:
        defaults = json.loads(cfg_path.read_text())
    except Exception:
        defaults = {}

training = defaults.get("training", {}) or {}
alignment = defaults.get("alignment", {}) or {}

def latest_checkpoint(base: Path) -> str:
    if not base.exists():
        return ""
    best = None
    best_step = -1
    for path in base.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best = path
    return str(best) if best else ""

train_dir = Path(training.get("output_dir", "outputs/student_runs/gemma3-270m"))
align_dir = Path(alignment.get("output_dir", "outputs/student_runs/gemma3-270m-dpo"))

print(train_dir)
print(latest_checkpoint(train_dir))
print(align_dir)
print(latest_checkpoint(align_dir))
PY
)"
IFS=$'\n' read -r TRAIN_OUTPUT_DIR TRAIN_LATEST_CKPT ALIGN_OUTPUT_DIR ALIGN_LATEST_CKPT <<< "${path_info}"

log "Logging to ${LOG_FILE}"
log "Starting dataset generation with config=${CONFIG_PATH}"
"$ROOT/runDatasetGeneration.sh" "${MODEL_ARG[@]}" --config "${CONFIG_PATH}"

if [[ $WITH_TRAINING -eq 1 ]]; then
  TRAIN_RESUME_ARGS=()
  if [[ -n "${TRAIN_LATEST_CKPT}" && -d "${TRAIN_LATEST_CKPT}" ]]; then
    log "Auto-resuming training from ${TRAIN_LATEST_CKPT}"
    TRAIN_RESUME_ARGS+=(--resume-from-checkpoint "${TRAIN_LATEST_CKPT}")
  fi
  log "Starting training stage"
  "$ROOT/runTraining.sh" --config "${CONFIG_PATH}" "${TRAIN_RESUME_ARGS[@]}"
fi

if [[ $WITH_EVALS -eq 1 ]]; then
  log "Starting evals stage"
  "$ROOT/runEvals.sh" --config "${CONFIG_PATH}"
fi

if [[ $WITH_ALIGNMENT -eq 1 ]]; then
  ALIGN_RESUME_ARGS=()
  if [[ -n "${ALIGN_LATEST_CKPT}" && -d "${ALIGN_LATEST_CKPT}" ]]; then
    log "Auto-resuming alignment from ${ALIGN_LATEST_CKPT}"
    ALIGN_RESUME_ARGS+=(--resume-from-checkpoint "${ALIGN_LATEST_CKPT}")
  fi
  log "Starting alignment stage"
  "$ROOT/runAlignment.sh" --config "${CONFIG_PATH}" "${ALIGN_RESUME_ARGS[@]}"
fi

log "runAll complete."
