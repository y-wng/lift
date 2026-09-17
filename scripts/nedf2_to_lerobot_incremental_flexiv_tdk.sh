#!/usr/bin/env bash
# Convert completed NEDF2 episodes into local LeRobot episode datasets.
# SOURCE_DIR contains raw recordings; OUTPUT_PATH is a separate output root.
# Choose CONFIG_PATH for your task (default: towelv3_online.yaml):
#   Book:  preprocess_data/configs/book_insertion_v3_online.yaml
#   Hanoi: preprocess_data/configs/hanoi_v1_online.yaml
#   Towel: preprocess_data/configs/towelv3_online.yaml
# Set OPENPI_DRY_RUN=1 to inspect the command without reading/writing data.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'USAGE'
Usage:
  SOURCE_DIR=/path/to/raw/episodes OUTPUT_PATH=/path/to/converted/episodes \
  bash scripts/nedf2_to_lerobot_incremental_flexiv_tdk.sh [additional Python CLI options]

Optional: CONFIG_PATH, POLL_INTERVAL (seconds, default: 1), FPS (default: 10),
          OPENPI_PYTHON, OPENPI_DRY_RUN=1.
Source and output directories must differ. Use a new output root for a new task.
USAGE
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${OPENPI_PYTHON:-${repo_root}/.venv/bin/python}"

config_path="${CONFIG_PATH:-${repo_root}/preprocess_data/configs/towelv3_online.yaml}"
source_dir="${SOURCE_DIR:?Set SOURCE_DIR to your raw NEDF2 episode directory}"
output_path="${OUTPUT_PATH:?Set OUTPUT_PATH to a separate LeRobot output directory}"
poll_interval="${POLL_INTERVAL:-1}"
fps="${FPS:-10}"

if [[ "$(realpath -m -- "${source_dir}")" == "$(realpath -m -- "${output_path}")" ]]; then
    echo "SOURCE_DIR and OUTPUT_PATH must be different directories." >&2
    exit 1
fi

command=(
    "${python_bin}" "${repo_root}/preprocess_data/nedf2_to_lerobot_incremental_flexiv_tdk.py"
    "${config_path}"
    --source-dir "${source_dir}"
    --output-path "${output_path}"
    --poll-interval "${poll_interval}"
    --fps "${fps}"
    "$@"
)

if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
