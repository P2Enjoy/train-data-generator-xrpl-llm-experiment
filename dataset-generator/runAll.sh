#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

WITH_TRAINING=0
WITH_EVALS=0
MODEL_ARG=()
CONFIG_PATH="config/defaults.json"

usage() {
  cat <<EOF
Usage: $0 [--model MODEL] [--with-training] [--with-evals]

This wraps the per-stage runners:
  - runDatasetGeneration.sh (always)
  - runTraining.sh (when --with-training)
  - runEvals.sh (when --with-evals)

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

"$ROOT/runDatasetGeneration.sh" "${MODEL_ARG[@]}" --config "${CONFIG_PATH}"

if [[ $WITH_TRAINING -eq 1 ]]; then
  "$ROOT/runTraining.sh" --config "${CONFIG_PATH}"
fi

if [[ $WITH_EVALS -eq 1 ]]; then
  "$ROOT/runEvals.sh"
fi

echo "runAll complete."
