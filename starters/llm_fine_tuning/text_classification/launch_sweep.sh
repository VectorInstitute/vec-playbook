#!/usr/bin/env bash
# Parameter sweep launcher using experiment config file

TRAINER="starters.llm_fine_tuning.text_classification.trainer:HFTextClassificationTrainer"
CONFIG_FILE="$(dirname "$0")/sweep_config.yaml"

echo "Launching sweep with config: $CONFIG_FILE"

vec-tool sweep "$TRAINER" \
    --experiment-config "$CONFIG_FILE"
