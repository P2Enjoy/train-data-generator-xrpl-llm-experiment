#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "Cleaning generated artifacts..."
rm -rf outputs

echo "Outputs removed; next run the pipeline (./runAll.sh)."
