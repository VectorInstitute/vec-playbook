#!/usr/bin/env bash
# Single job launcher using experiment config file

TRAINER="starters.vlm_fine_tuning.image_captioning.trainer:ImageCaptioningTrainer"
CONFIG_FILE="$(dirname "$0")/config.yaml"

echo "Launching single job with config: $CONFIG_FILE"

vec-tool submit "$TRAINER" \
    --experiment-config "$CONFIG_FILE"
