#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export PYTHONUNBUFFERED=1

export CUDA_HOME=/usr/local/cuda-13.0
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="12.1a"

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
PYTHONUNBUFFERED=1 uv run python -u scripts/train_student_unsloth.py "$@"
echo "Training finished."
