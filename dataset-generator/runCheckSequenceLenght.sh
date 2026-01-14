#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="config/defaults.json"
DATASET=""

usage() {
  cat <<EOF
Usage: $0 [--config PATH] [--dataset PATH]

Checks the longest line in a JSONL dataset and reports its character length and an approximate token count.
Defaults come from dataset_generation.dataset_out in the config.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET}" ]]; then
  DATASET="$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("dataset_generation", {}).get("dataset_out", ""))
PY
)"
fi

if [[ -z "${DATASET}" ]]; then
  echo "Dataset path not found in config and not provided via --dataset."
  exit 1
fi

echo "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Checking dataset: ${DATASET}"

uv sync
PYTHONUNBUFFERED=1 uv run python -u scripts/check_sequence_length.py --dataset "${DATASET}" --tokenizer=tiktoken