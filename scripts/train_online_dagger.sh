#!/usr/bin/env bash
# General entrypoint: forward arguments to train_online_dagger.py unchanged.
# Use the train_online_dagger_lerobot*.sh scripts for task presets and ablations.
# Examples:
#   bash scripts/train_online_dagger.sh --help
#   bash scripts/train_online_dagger.sh --config-name YOUR_CONFIG --exp-name YOUR_RUN
# Set OPENPI_PYTHON to use another environment, or OPENPI_DRY_RUN=1 to preview.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${OPENPI_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found or not executable: ${python_bin}" >&2
    exit 1
fi

# With no arguments, show the Python CLI help instead of starting an experiment.
if [[ "$#" == "0" ]]; then
    set -- --help
fi
command=("${python_bin}" "${repo_root}/scripts/train_online_dagger.py" "$@")

if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
