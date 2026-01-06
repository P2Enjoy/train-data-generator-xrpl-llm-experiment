#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$ROOT/config/defaults.json"

usage() {
  cat <<EOF
Reset generated artifacts using paths from a config file.

Usage: $(basename "$0") [--config path] [--dataset] [--evaluations] [--training] [--alignment] [--report] [--all]

Without flags, no action is taken. Flags are mutually independent except:
  - --all cannot be combined with any specific reset flag.
  - --config selects the config file (default: config/defaults.json).
EOF
  exit "${1:-1}"
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required tool: $1"; exit 1; }
}

resolve_path() {
  local path="$1"
  if [[ -z "$path" ]]; then
    return
  fi
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT" "$path"
  fi
}

safe_rm() {
  local target="$1"
  if [[ -z "$target" ]]; then
    return
  fi
  # Avoid deleting outside the repo root.
  case "$target" in
    "$ROOT"/*) rm -rf "$target" ;;
    *) echo "Skipping unsafe path outside repo: $target" ;;
  esac
}

confirm_all() {
  echo "WARNING: --all will remove every configured output path."
  read -r -p "Type 'yes' to continue: " first
  [[ "$first" == "yes" ]] || { echo "Aborted."; exit 1; }
  read -r -p "Type 'ALL' to confirm: " second
  [[ "$second" == "ALL" ]] || { echo "Aborted."; exit 1; }
}

DATASET=false
EVALS=false
TRAINING=false
ALIGNMENT=false
REPORT=false
ALL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift
      ;;
    --dataset) DATASET=true ;;
    --evaluations) EVALS=true ;;
    --training) TRAINING=true ;;
    --alignment) ALIGNMENT=true ;;
    --report) REPORT=true ;;
    --all) ALL=true ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown arg: $1"; usage 1 ;;
  esac
  shift
done

if ! "$DATASET" && ! "$EVALS" && ! "$TRAINING" && ! "$ALIGNMENT" && ! "$REPORT" && ! "$ALL"; then
  echo "No reset flags provided; nothing to do."
  exit 0
fi

if "$ALL" && ( "$DATASET" || "$EVALS" || "$TRAINING" || "$ALIGNMENT" || "$REPORT" ); then
  echo "--all cannot be combined with specific reset flags."
  exit 1
fi

require_tool jq
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

# Collect paths from config sections.
dataset_paths=($(jq -r '[
  .dataset_generation.schema_specs_out,
  .dataset_generation.final_schemas_out,
  .dataset_generation.dataset_out,
  .dataset_generation.training_corpus_out
] | map(select(.!=null)) | .[]' "$CONFIG"))

evaluation_paths=($(jq -r '[
  .evaluation.eval_results,
  .evaluation.eval_summary,
  .alignment.eval_results,
  .alignment.eval_summary
] | map(select(.!=null)) | .[]' "$CONFIG"))

training_paths=($(jq -r '[
  .training.output_dir,
  .training.training_corpus
] | map(select(.!=null)) | .[]' "$CONFIG"))

alignment_paths=($(jq -r '[
  .alignment.pairs_out,
  .alignment.output_dir,
  .alignment.adapter
] | map(select(.!=null)) | .[]' "$CONFIG"))

report_paths=($(jq -r '[
  .reporting.reports_dir
] | map(select(.!=null)) | .[]' "$CONFIG"))

if "$ALL"; then
  confirm_all
  DATASET=true
  EVALS=true
  TRAINING=true
  ALIGNMENT=true
  REPORT=true
fi

echo "Using config: $CONFIG"

if "$DATASET"; then
  echo "[reset] dataset_generation outputs..."
  for p in "${dataset_paths[@]}"; do
    safe_rm "$(resolve_path "$p")"
  done
fi

if "$TRAINING"; then
  echo "[reset] training outputs..."
  for p in "${training_paths[@]}"; do
    safe_rm "$(resolve_path "$p")"
  done
fi

if "$EVALS"; then
  echo "[reset] evaluation outputs..."
  for p in "${evaluation_paths[@]}"; do
    safe_rm "$(resolve_path "$p")"
  done
fi

if "$ALIGNMENT"; then
  echo "[reset] alignment outputs..."
  for p in "${alignment_paths[@]}"; do
    safe_rm "$(resolve_path "$p")"
  done
fi

if "$REPORT"; then
  echo "[reset] reports..."
  for p in "${report_paths[@]}"; do
    safe_rm "$(resolve_path "$p")"
  done
fi

echo "Reset complete."
