#!/usr/bin/env bash
# One-time local setup on macOS (Apple Silicon): Python 3.11 venv, pinned packages,
# and a project-local MoltenVK so SAPIEN can render through Metal.
# Nothing is installed system-wide; delete .venv/ and .tools/ to undo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOLTENVK_VERSION="v1.4.2"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  UV="$(python3 -m site --user-base)/bin/uv"
  [ -x "$UV" ] || python3 -m pip install --user --quiet uv
else
  UV="$(command -v uv)"
fi

"$UV" venv --python 3.11 .venv
"$UV" pip install --python .venv/bin/python -r requirements.txt

ICD=".tools/MoltenVK/MoltenVK/dynamic/dylib/macOS/MoltenVK_icd.json"
if [ ! -f "$ICD" ]; then
  mkdir -p .tools
  curl -fsSL -o .tools/MoltenVK-macos.tar \
    "https://github.com/KhronosGroup/MoltenVK/releases/download/${MOLTENVK_VERSION}/MoltenVK-macos.tar"
  tar -xf .tools/MoltenVK-macos.tar -C .tools
  rm .tools/MoltenVK-macos.tar
fi

echo "Setup done. Before running anything: source scripts/env_macos.sh"
