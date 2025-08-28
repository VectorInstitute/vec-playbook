#!/usr/bin/env bash
# Simple launcher for non-checkpointable trainer

TRAINER="starters.simple_nn.single_not_checkpointable.trainer:SimpleTrainerNoCheckpoint"

vec-tool submit "$TRAINER" \
    --input-dim 10 \
    --hidden-dim 64 \
    --num-classes 3 \
    --batch-size 32 \
    --lr 1e-3 \
    --num-epochs 50
