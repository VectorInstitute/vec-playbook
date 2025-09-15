# Scripts

## RLVR

```bash
uv run starters/llm_fine_tuning/rlvr/grpo_trainer.py
```

## vec-inf Placeholder

```bash
singularity exec --nv  --bind /model-weights/Qwen3-0.6B --containall /model-weights/vec-inf-shared/vector-inference_0.10.0.sif \
vllm serve /model-weights/Qwen3-0.6B \
--served-model-name Qwen3-0.6B \
--host "0.0.0.0" \
--port 8000 \
--max-model-len 40960 \
--max-num-seqs 256 \
--enable-auto-tool-choice \
--tool-call-parser hermes
```
