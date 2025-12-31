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

log "Logging to ${LOG_FILE}"
log "Starting dataset generation with config=${CONFIG_PATH}"
"$ROOT/runDatasetGeneration.sh" "${MODEL_ARG[@]}" --config "${CONFIG_PATH}"

if [[ $WITH_TRAINING -eq 1 ]]; then
  log "Starting training stage"
  "$ROOT/runTraining.sh" --config "${CONFIG_PATH}"
fi

if [[ $WITH_EVALS -eq 1 ]]; then
  log "Starting evals stage"
  "$ROOT/runEvals.sh" --config "${CONFIG_PATH}"
fi

if [[ $WITH_ALIGNMENT -eq 1 ]]; then
  log "Starting alignment stage"
  "$ROOT/runAlignment.sh" --config "${CONFIG_PATH}"
fi

log "runAll complete."
