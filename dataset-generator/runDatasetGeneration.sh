#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

MODEL_DEFAULT="gpt-oss:120b"
if [ -f .llmrc ]; then
  MODEL_DEFAULT="$(<.llmrc)"
fi
MODEL="${LLM_MODEL:-$MODEL_DEFAULT}"

usage() {
  echo "Usage: $0 [--model MODEL]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
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

echo "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Installing / syncing dependencies with uv..."
uv sync

run_step() {
  local name="$1"
  local condition="$2"
  shift 2
  if eval "$condition"; then
    echo "Skipping ${name}: output already exists."
    return
  fi

  local tries=0
  local max_tries=2
  while true; do
    tries=$((tries + 1))
    echo "Running ${name} (attempt ${tries})..."
    local cmd_output
    if ! cmd_output="$(uv run python "$@" 2>&1)"; then
      echo "${cmd_output}"
      echo "[error] command failed"
      exit 1
    fi
    echo "${cmd_output}"
    if [[ "${cmd_output}" == *"[warn]"* && ${tries} -lt ${max_tries} ]]; then
      echo "[retry] detected warning, re-running ${name}"
      continue
    fi
    break
  done
}

run_step \
  "domain spec synthesis" \
  "[ -s outputs/d_01_domain_specs.jsonl ]" \
  scripts/generate_domain_specs.py \
    --prompts data/domain_prompts.jsonl \
    --out outputs/d_01_domain_specs.jsonl \
    --model "${MODEL}"

run_step \
  "schema build" \
  "[ -s outputs/d_02_final_schemas.jsonl ]" \
  scripts/build_schemas.py \
    --domain-specs outputs/d_01_domain_specs.jsonl \
    --base-template data/base_schema_template.json \
    --out outputs/d_02_final_schemas.jsonl

run_step \
  "example query generation" \
  "[ -s outputs/d_03_schema_queries.jsonl ]" \
  scripts/generate_example_queries.py \
    --schemas outputs/d_02_final_schemas.jsonl \
    --out outputs/d_03_schema_queries.jsonl \
    --per-schema 6 \
    --model "${MODEL}"

run_step \
  "dataset generation" \
  "[ -s outputs/d_04_dataset.jsonl ]" \
  scripts/generate_dataset.py \
    --schemas outputs/d_02_final_schemas.jsonl \
    --out outputs/d_04_dataset.jsonl \
    --positives-per-schema 8 \
    --negative-ratio 0.4

run_step \
  "training corpus export" \
  "[ -s outputs/d_05_training_corpus.jsonl ]" \
  scripts/build_training_corpus.py \
    --dataset outputs/d_04_dataset.jsonl \
    --out outputs/d_05_training_corpus.jsonl

PRETTY_DIR="outputs/pretty"
mkdir -p "$PRETTY_DIR"

pretty_pairs=(
  "outputs/d_01_domain_specs.jsonl|$PRETTY_DIR/d_01_domain_specs.json"
  "outputs/d_02_final_schemas.jsonl|$PRETTY_DIR/d_02_final_schemas.json"
  "outputs/d_03_schema_queries.jsonl|$PRETTY_DIR/d_03_schema_queries.json"
  "outputs/d_04_dataset.jsonl|$PRETTY_DIR/d_04_dataset.json"
  "outputs/d_05_training_corpus.jsonl|$PRETTY_DIR/d_05_training_corpus.json"
)

for pair in "${pretty_pairs[@]}"; do
  IFS="|" read -r input output <<< "$pair"
  if [ -s "$output" ]; then
    echo "Skipping prettify for $input"
    continue
  fi
  echo "Prettifying $input → $output"
  uv run python scripts/pretty_jsonl.py --input "$input" --out "$output"
done

echo "Dataset generation complete."
