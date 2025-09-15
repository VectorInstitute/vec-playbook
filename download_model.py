"""
Download model from Hub to a local path.

Usage:
uv run download_model.py \
--repo_name Qwen/Qwen3-0.6B \
--local_path /model-weights/Qwen3-0.6B

uv run download_model.py \
--repo_name Qwen/Qwen3-4B-Thinking-2507 \
--local_path /model-weights/Qwen3-4B-Thinking-2507

uv run download_model.py \
--repo_name Qwen/Qwen3-1.7B \
--local_path /model-weights/Qwen3-1.7B
"""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--repo_name", required=True)
parser.add_argument("--local_path", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    AutoTokenizer.from_pretrained(args.repo_name).save_pretrained(args.local_path)
    AutoModelForCausalLM.from_pretrained(args.repo_name).save_pretrained(
        args.local_path
    )
