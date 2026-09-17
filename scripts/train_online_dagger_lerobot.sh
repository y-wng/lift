#!/usr/bin/env bash
# Original π₀.₅ online-DAGGER launcher.
#
# Quick start:
#   OPENPI_CHECKPOINT_ROOT=/path/to/checkpoints \
#   OPENPI_INIT_CHECKPOINT=/path/to/checkpoints/30000/params \
#   bash scripts/train_online_dagger_lerobot.sh
#
# Set OPENPI_DRY_RUN=1 to print the final command without starting training.
# Set OPENPI_LOCAL_LEROBOT_DATA_ROOT to read local online episodes. Leave it
# empty when the online data-cloud service is configured instead.
#
# Uses the original vision-only policy: this launcher does not request wrench
# inputs or LIFT's intervention chunk filter. It therefore consumes all
# supplied online frames; prepare a correction-only source for comparisons
# that should exclude autonomous rollout frames.
# Data layout and shared environment overrides are documented in the header
# of train_online_dagger_lerobot_reactive.sh. Its force requirements do not
# apply here. Choose a matching non-reactive OPENPI_CONFIG_NAME.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${OPENPI_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found or not executable: ${python_bin}" >&2
    exit 1
fi

# Paths and runtime defaults. Override any of these with environment variables.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
export OPENPI_CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-${repo_root}/checkpoints}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

# Experiment identity and model initialization.
config_name="${OPENPI_CONFIG_NAME:-pi05_iPhoneSingle_book_insertion_v3_100}"
exp_name="${OPENPI_EXP_NAME:-${config_name#pi05_iPhoneSingle_}_online_$(date -u +%Y%m%d_%H%M%S)}"
checkpoint_root="${OPENPI_CHECKPOINT_ROOT}"
init_checkpoint="${OPENPI_INIT_CHECKPOINT:-${checkpoint_root}/30000/params}"
gpu_ids="${OPENPI_CUDA_VISIBLE_DEVICES-${CUDA_VISIBLE_DEVICES-0,1}}"

# Online data source. A local root should contain LeRobot episode directories.
local_data_root="${OPENPI_LOCAL_LEROBOT_DATA_ROOT:-}"
online_repo_id="${OPENPI_ONLINE_REPO_ID:-flexiv/${config_name#pi05_iPhoneSingle_}_online_$(date -u +%Y%m%d_%H%M%S)}"
datacloud_endpoint="${OPENPI_DATACLOUD_ENDPOINT:-}"
identifier="${OPENPI_IDENTIFIER:-}"
fetch_interval="${OPENPI_FETCH_INTERVAL:-1}"
robot_type="${OPENPI_ROBOT_TYPE:-single_iphone_flexiv}"
fps="${OPENPI_FPS:-10}"
task_description="${OPENPI_TASK_DESCRIPTION:-}"

# Training and loader controls. Empty step/batch values use config defaults.
save_interval="${OPENPI_SAVE_INTERVAL:-1000}"
log_interval="${OPENPI_LOG_INTERVAL:-5}"
prefetch_batches="${OPENPI_PREFETCH_BATCHES:-2}"
sampler_update_interval="${OPENPI_SAMPLER_UPDATE_INTERVAL:-10}"
num_train_steps="${OPENPI_NUM_TRAIN_STEPS:-}"
batch_size="${OPENPI_BATCH_SIZE:-}"
min_online_ratio="${OPENPI_MIN_ONLINE_RATIO:-0.5}"
max_online_ratio="${OPENPI_MAX_ONLINE_RATIO:-0.5}"
initial_online_weight="${OPENPI_INITIAL_ONLINE_WEIGHT:-0.5}"

# New runs are timestamped; existing output is never overwritten by default.
# Set OPENPI_EXP_NAME with OPENPI_RESUME=1 or OPENPI_OVERWRITE=1 deliberately.
run_mode_args=()
if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
    run_mode_args=(--resume)
elif [[ "${OPENPI_OVERWRITE:-0}" == "1" ]]; then
    run_mode_args=(--overwrite)
fi
if [[ "${OPENPI_RESUME:-0}" == "1" && "${OPENPI_OVERWRITE:-0}" == "1" ]]; then
    echo "OPENPI_RESUME and OPENPI_OVERWRITE cannot both be 1." >&2
    exit 1
fi

# Only add optional CLI flags when the corresponding variable is set.
optional_args=()
if [[ -n "${OPENPI_FSDP_DEVICES:-}" ]]; then
    optional_args+=(--fsdp-devices "${OPENPI_FSDP_DEVICES}")
fi
if [[ "${OPENPI_ALLOW_OFFLINE_WARM_START:-1}" == "0" ]]; then
    optional_args+=(--no-allow-offline-warm-start)
fi
if [[ -n "${num_train_steps}" ]]; then
    optional_args+=(--num-train-steps "${num_train_steps}")
fi
if [[ -n "${batch_size}" ]]; then
    optional_args+=(--batch-size "${batch_size}")
fi

if [[ ! -d "${init_checkpoint}" ]]; then
    echo "Initial checkpoint params directory not found: ${init_checkpoint}" >&2
    echo "Set OPENPI_INIT_CHECKPOINT or OPENPI_CHECKPOINT_ROOT." >&2
    exit 1
fi

# Keep the command in an array so paths and task text remain shell-safe.
command=(
    env CUDA_VISIBLE_DEVICES="${gpu_ids}"
    "${python_bin}" "${repo_root}/scripts/train_online_dagger.py"
    --config-name "${config_name}"
    --exp-name "${exp_name}"
    --checkpoint-base-dir "${checkpoint_root}"
    --init-checkpoint "${init_checkpoint}"
    "${run_mode_args[@]}"
    "${optional_args[@]}"
    --datacloud-endpoint "${datacloud_endpoint}"
    --identifier "${identifier}"
    --local-lerobot-data-root "${local_data_root}"
    --online-repo-id "${online_repo_id}"
    --fetch-interval "${fetch_interval}"
    --robot-type "${robot_type}"
    --fps "${fps}"
    --task-description "${task_description}"
    --save-interval "${save_interval}"
    --log-interval "${log_interval}"
    --prefetch-batches "${prefetch_batches}"
    --sampler-update-interval "${sampler_update_interval}"
    --min-online-ratio "${min_online_ratio}"
    --max-online-ratio "${max_online_ratio}"
    --initial-online-weight "${initial_online_weight}"
    "$@"
)

if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
