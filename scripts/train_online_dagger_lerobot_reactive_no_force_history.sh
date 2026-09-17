#!/usr/bin/env bash
# LIFT force-history ablation.
#
# This keeps the reactive model and data pipeline unchanged, but permits only
# the first force-memory token (k == 0) through the latency-aligned causal mask.
# It does not remove the wrench input or select each action's latest force.
# Configure the same paths as the main reactive launcher, then run this wrapper.
# Set Pi0Config.use_force_history=False in the config used for inference too;
# the training CLI override is not stored as a model-config file in checkpoints.
#
# Example:
#   OPENPI_DRY_RUN=1 bash scripts/train_online_dagger_lerobot_reactive_no_force_history.sh
set -euo pipefail

export OPENPI_DISABLE_FORCE_HISTORY="1"
config_name="${OPENPI_CONFIG_NAME:-pi05_iPhoneSingle_book_insertion_v3_100_reactive}"
export OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-${config_name}_no_force_history_$(date -u +%Y%m%d_%H%M%S)}"

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/train_online_dagger_lerobot_reactive.sh" "$@"
