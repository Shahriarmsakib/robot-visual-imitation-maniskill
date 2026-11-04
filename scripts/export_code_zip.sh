#!/usr/bin/env bash
# Zip the committed code (no data, checkpoints or venv) for upload as a private Kaggle Dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p dist
OUT="dist/robot-visual-imitation-maniskill-$(git rev-parse --short HEAD).zip"
git archive --format=zip --prefix=robot-visual-imitation-maniskill/ -o "$OUT" HEAD
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1)). Uncommitted changes are NOT included."
