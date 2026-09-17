#!/usr/bin/env bash
# LIFT sampling ablation: online-only batches (0:1 offline:online).
#
# All paths and model settings are inherited from the main reactive launcher.
# Add OPENPI_DRY_RUN=1 to inspect the assembled command.
set -euo pipefail

export OPENPI_ALLOW_OFFLINE_WARM_START="${OPENPI_ALLOW_OFFLINE_WARM_START:-0}"

export OPENPI_MIN_ONLINE_RATIO="1.0"
export OPENPI_MAX_ONLINE_RATIO="1.0"
export OPENPI_INITIAL_ONLINE_WEIGHT="1.0"
config_name="${OPENPI_CONFIG_NAME:-pi05_iPhoneSingle_book_insertion_v3_100_reactive}"
export OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-${config_name}_ratio_0to1_$(date -u +%Y%m%d_%H%M%S)}"

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/train_online_dagger_lerobot_reactive.sh" "$@"
