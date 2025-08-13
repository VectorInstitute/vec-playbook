#!/usr/bin/env bash
# Single job launcher using experiment config file

TRAINER="starters.llm_fine_tuning.text_classification.trainer:HFTextClassificationTrainer"
CONFIG_FILE="$(dirname "$0")/config.yaml"

echo "Launching single job with config: $CONFIG_FILE"

vec-tool submit "$TRAINER" \
    --experiment-config "$CONFIG_FILE"
