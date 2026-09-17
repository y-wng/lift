#!/usr/bin/env bash
# LIFT default sampling ratio: 1 offline sample for every 1 online sample.
#
# This wrapper makes the paper/default setting explicit while inheriting all
# paths and model settings from the main reactive launcher.
# Add OPENPI_DRY_RUN=1 to inspect the assembled command.
set -euo pipefail

export OPENPI_ALLOW_OFFLINE_WARM_START="${OPENPI_ALLOW_OFFLINE_WARM_START:-0}"

export OPENPI_MIN_ONLINE_RATIO="0.5"
export OPENPI_MAX_ONLINE_RATIO="0.5"
export OPENPI_INITIAL_ONLINE_WEIGHT="0.5"
config_name="${OPENPI_CONFIG_NAME:-pi05_iPhoneSingle_book_insertion_v3_100_reactive}"
export OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-${config_name}_ratio_1to1_$(date -u +%Y%m%d_%H%M%S)}"

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/train_online_dagger_lerobot_reactive.sh" "$@"
