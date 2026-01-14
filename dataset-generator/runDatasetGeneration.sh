#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export PYTHONUNBUFFERED=1

log() {
  echo "[$(date '+%F %T')] $*"
}

TEACHER_MODEL=""

CONFIG_PATH="config/defaults.json"

usage() {
  echo "Usage: $0 [--teacher-model MODEL] [--config CONFIG_PATH]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-model)
      TEACHER_MODEL="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
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

mkdir -p outputs

log "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
log "Installing / syncing dependencies with uv..."
uv sync

env_exports=$(
  UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python - "$CONFIG_PATH" <<'PY'
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

dg = defaults.get("dataset_generation", {}) or {}

def emit(name: str, fallback: str) -> None:
    value = dg.get(name, fallback)
    print(f'{name}="{value}"')

emit("schema_specs_out", "outputs/d_01_domain_specs.jsonl")
emit("final_schemas_out", "outputs/d_02_final_schemas.jsonl")
emit("dataset_out", "outputs/d_03_dataset.jsonl")
emit("training_corpus_out", "outputs/d_04_training_corpus.jsonl")
emit("dataset_checkpoint", "outputs/checkpoints/dataset_generation.json")
emit("teacher_model", "")
PY
)

eval "${env_exports}"
log "Resolved dataset outputs: specs=${schema_specs_out}, schemas=${final_schemas_out}, dataset=${dataset_out}, corpus=${training_corpus_out}, checkpoint=${dataset_checkpoint}"
if [[ -z "${TEACHER_MODEL}" ]]; then
  TEACHER_MODEL="${teacher_model:-}"
fi
if [[ -z "${TEACHER_MODEL}" ]]; then
  echo "Teacher model is required; set dataset_generation.teacher_model in config or pass --teacher-model."
  exit 1
fi
log "Using teacher model: ${TEACHER_MODEL}"

run_step() {
  local name="$1"
  local condition="$2"
  shift 2
  if eval "$condition"; then
    log "Skipping ${name}: output already exists."
    return
  fi

  local tries=0
  local max_tries=2
  while true; do
    local tmpfile
    tmpfile="$(mktemp)"
    tries=$((tries + 1))
    log "Running ${name} (attempt ${tries})..."
    if ! PYTHONUNBUFFERED=1 uv run python -u "$@" 2>&1 | tee "${tmpfile}"; then
      echo "[error] command failed"
      rm -f "${tmpfile}"
      exit 1
    fi
    if grep -q "\[warn\]" "${tmpfile}" && [[ ${tries} -lt ${max_tries} ]]; then
      log "[retry] detected warning, re-running ${name}"
      rm -f "${tmpfile}"
      continue
    fi
    rm -f "${tmpfile}"
    break
  done
}

run_step \
  "domain spec synthesis" \
  "[ -s \"${schema_specs_out}\" ]" \
scripts/generate_domain_specs.py \
    --config "${CONFIG_PATH}" \
    --teacher-model "${TEACHER_MODEL}"

run_step \
  "schema build" \
  "[ -s \"${final_schemas_out}\" ]" \
  scripts/build_schemas.py \
    --config "${CONFIG_PATH}"

DATASET_RESUME_ARGS=()
DATASET_SKIP_COND="[ -s \"${dataset_out}\" ] && { [ -z \"${dataset_checkpoint}\" ] || [ ! -f \"${dataset_checkpoint}\" ]; }"
if [[ -n "${dataset_checkpoint}" && -f "${dataset_checkpoint}" ]]; then
  log "Dataset checkpoint found at ${dataset_checkpoint}; resuming generation."
  DATASET_RESUME_ARGS+=(--resume)
fi

run_step \
  "dataset generation" \
  "${DATASET_SKIP_COND}" \
  scripts/generate_dataset.py \
    --config "${CONFIG_PATH}" \
    --teacher-model "${TEACHER_MODEL}" \
    "${DATASET_RESUME_ARGS[@]}"

run_step \
  "training corpus export" \
  "[ -s \"${training_corpus_out}\" ]" \
  scripts/build_training_corpus.py \
    --config "${CONFIG_PATH}"

PRETTY_DIR="outputs/pretty"
mkdir -p "$PRETTY_DIR"

pretty_pairs=(
  "${schema_specs_out}|$PRETTY_DIR/$(basename "${schema_specs_out%.jsonl}").json"
  "${final_schemas_out}|$PRETTY_DIR/$(basename "${final_schemas_out%.jsonl}").json"
  "${dataset_out}|$PRETTY_DIR/$(basename "${dataset_out%.jsonl}").json"
  "${training_corpus_out}|$PRETTY_DIR/$(basename "${training_corpus_out%.jsonl}").json"
)

for pair in "${pretty_pairs[@]}"; do
  IFS="|" read -r input output <<< "$pair"
  if [ -s "$output" ]; then
    log "Skipping prettify for $input"
    continue
  fi
  log "Prettifying $input → $output"
  uv run python scripts/pretty_jsonl.py --input "$input" --out "$output"
done

log "Dataset generation complete."
