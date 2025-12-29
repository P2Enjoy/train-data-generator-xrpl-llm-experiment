#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

usage() {
  echo "Usage: $0 [train_student_unsloth.py args...]"
  echo "Example: $0 --output-dir outputs/student_runs/gemma3-270m --max-steps 500"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

echo "Using UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "Installing / syncing dependencies with uv..."
uv sync

echo "Starting student training..."
uv run python scripts/train_student_unsloth.py "$@"
echo "Training finished."
