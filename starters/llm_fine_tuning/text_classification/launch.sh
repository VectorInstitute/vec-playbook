#!/usr/bin/env bash

TRAINER="starters.llm_fine_tuning.text_classification.trainer:HFTextClassificationTrainer"

vec-tool submit "$TRAINER" \
  --job-dir "./vec_jobs" \
  --nodes 1 \
  --gpus-per-node 1 \
  --gpu-type a40 \
  --time "1:00:00" \
  --name "hf-textcls" \
  --kw "num_epochs=2 batch_size=16 lr=5e-5 save_steps=200 eval_steps=200"
