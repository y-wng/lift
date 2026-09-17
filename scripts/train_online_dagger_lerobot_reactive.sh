#!/usr/bin/env bash
# LIFT reactive online-DAGGER launcher.
#
# Quick start:
#   OPENPI_CHECKPOINT_ROOT=/path/to/checkpoints \
#   OPENPI_INIT_CHECKPOINT=/path/to/checkpoints/30000/params \
#   OPENPI_LOCAL_LEROBOT_DATA_ROOT=/path/to/online/episodes \
#   bash scripts/train_online_dagger_lerobot_reactive.sh
#
# The default sampler is a fixed 1:1 offline:online mixture. Use one of the
# ratio wrapper scripts for the documented sampling ablations. Set
# OPENPI_DRY_RUN=1 to inspect the final command without training.
#
# Data layout:
#   HF_LEROBOT_HOME/<offline-repo-id>/{meta,data}/ contains offline data.
#   OPENPI_LOCAL_LEROBOT_DATA_ROOT/<dataset>/{meta,data}/ contains corrections.
# Each immediate child is a complete LeRobot dataset. Incremental children
# named episode_* or ep_* must also have a .done marker after export completes.
# Online samples need left_wrist_img, state, actions, left_wrench (6D), and
# control_flag. Only full action chunks whose control_flag values match OPENPI_INTERVENTION_VALUE (default -1)
# enter the LIFT intervention sampler. See training/online_data_fetcher.py
# and training/data_loader.py under src/openpi/ for ingestion and filtering.
# OPENPI_ONLINE_REPO_ID is a disposable cache recreated at startup, even on
# resume; never point it at offline data or a source dataset. The input folder
# can stay fixed across runs, but choose a fresh cache ID for each process.
# Without local data or a service endpoint, batches are offline-only: a ratio
# ablation is not meaningful until eligible online samples have been ingested.
#
# Task and runtime overrides:
#   OPENPI_CONFIG_NAME / OPENPI_TASK_DESCRIPTION select a matching task preset.
#   OPENPI_INIT_CHECKPOINT points to offline params; normalization assets still
#     come from the selected config's AssetsConfig.
#   OPENPI_CUDA_VISIBLE_DEVICES selects GPUs; OPENPI_BATCH_SIZE is global.
#   OPENPI_NUM_TRAIN_STEPS, OPENPI_SAVE_INTERVAL, OPENPI_LOG_INTERVAL count steps.
#   OPENPI_FETCH_INTERVAL polls completed datasets every N training steps.
#   OPENPI_FPS and OPENPI_ROBOT_TYPE must agree with the source dataset schema.
#   OPENPI_PREFETCH_BATCHES and OPENPI_SAMPLER_UPDATE_INTERVAL tune loader work.
#   OPENPI_PYTHON defaults to this repository's .venv/bin/python.
#   WANDB_MODE defaults to offline; use disabled to disable logging.
#   OPENPI_DATACLOUD_ENDPOINT / OPENPI_IDENTIFIER select an optional service;
#     neither is needed for local LeRobot data or supplied by this repository.
#
# New experiments are timestamped by default. To resume, reuse OPENPI_EXP_NAME
# and OPENPI_CHECKPOINT_ROOT with OPENPI_RESUME=1. OPENPI_OVERWRITE=1 instead
# deletes that experiment; never enable both modes. To change the sampling
# ratio directly, set MIN/MAX_ONLINE_RATIO and INITIAL_ONLINE_WEIGHT (all with
# OPENPI_ prefixes) to the same online fraction. A 1:2 offline:online mix is 2/3.
# Unequal bounds enable the adaptive sampler rather than a fixed-ratio ablation.
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

# Experiment identity and pretrained π₀.₅ initialization.
config_name="${OPENPI_CONFIG_NAME:-pi05_iPhoneSingle_book_insertion_v3_100_reactive}"
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

# Training, fetching, and sampler controls.
save_interval="${OPENPI_SAVE_INTERVAL:-1000}"
log_interval="${OPENPI_LOG_INTERVAL:-5}"
prefetch_batches="${OPENPI_PREFETCH_BATCHES:-2}"
sampler_update_interval="${OPENPI_SAMPLER_UPDATE_INTERVAL:-10}"
num_train_steps="${OPENPI_NUM_TRAIN_STEPS:-}"
batch_size="${OPENPI_BATCH_SIZE:-}"

# The ratio values are online fractions: 0.5 means 1:1 offline:online.
window_size="${OPENPI_WINDOW_SIZE:-200}"
boost_factor="${OPENPI_BOOST_FACTOR:-1.5}"
min_online_ratio="${OPENPI_MIN_ONLINE_RATIO:-0.5}"
max_online_ratio="${OPENPI_MAX_ONLINE_RATIO:-0.5}"
initial_online_weight="${OPENPI_INITIAL_ONLINE_WEIGHT:-0.5}"

# Ratio ablations are available as companion launchers:
# train_online_dagger_lerobot_reactive_ratio_0to1.sh, ratio_1to2.sh
# The single-frame-force ablation is available as:
# train_online_dagger_lerobot_reactive_no_force_history.sh

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

# Optional overrides are omitted so the config can provide its own defaults.
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
if [[ "${OPENPI_DISABLE_FORCE_HISTORY:-0}" == "1" ]]; then
    optional_args+=(--disable-force-history)
fi

if [[ ! -d "${init_checkpoint}" ]]; then
    echo "Initial checkpoint params directory not found: ${init_checkpoint}" >&2
    echo "Set OPENPI_INIT_CHECKPOINT or OPENPI_CHECKPOINT_ROOT." >&2
    exit 1
fi
if [[ -z "${local_data_root}" && -z "${datacloud_endpoint}" ]]; then
    echo "No online data source configured; this command will train from offline data only." >&2
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
    --online-use-wrench
    --online-intervention-only
    --intervention-value "${OPENPI_INTERVENTION_VALUE:--1}"
    --save-interval "${save_interval}"
    --log-interval "${log_interval}"
    --prefetch-batches "${prefetch_batches}"
    --sampler-update-interval "${sampler_update_interval}"
    --window-size "${window_size}"
    --boost-factor "${boost_factor}"
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
