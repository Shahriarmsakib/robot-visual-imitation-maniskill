# Source this (do not execute) in each new terminal on macOS:
#   source scripts/env_macos.sh
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
export VK_ICD_FILENAMES="$_ROOT/.tools/MoltenVK/MoltenVK/dynamic/dylib/macOS/MoltenVK_icd.json"
export MVK_CONFIG_LOG_LEVEL=0
source "$_ROOT/.venv/bin/activate"
unset _ROOT
