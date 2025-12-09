"""Test Local Submitit vLLM executor."""

import logging
import os

import pytest
import submitit
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import EngineArgs

# "best practice" to use absolute imports for pytest
from templates.src.rl.rlvr.submitit_vllm import ExecutorConfig, SubmititVLLM


MODEL = os.environ.get("MODEL_NAME", "/model-weights/Qwen3-8B")


@pytest.fixture
def executor_configs():
    """Get config for one local executor."""
    local_executor = submitit.LocalExecutor("/tmp/.submitit/")
    visible_gpus = list(
        map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    )
    logging.info(f"visible_gpus: {visible_gpus}")
    local_executor.update_parameters(
        visible_gpus=visible_gpus, gpus_per_node=len(visible_gpus), timeout_min=60
    )
    return [
        ExecutorConfig(
            name="local", executor=local_executor, num_replicas=1, concurrency=128
        )
    ]


@pytest.mark.asyncio
async def test_submitit_vllm_local(executor_configs: list[ExecutorConfig]):
    """Run integration test."""
    cache_dir = f"/tmp/.cache/vllm_compiled_graphs/{MODEL}"
    logging.basicConfig(level=logging.DEBUG)
    with SubmititVLLM(
        logger_name_prefix="submitit_vllm.policy",
        executor_configs=executor_configs,
        engine_args=EngineArgs(
            model=MODEL,
            compilation_config=CompilationConfig(cache_dir=cache_dir),
            max_model_len=1024,
        ),
    ) as submitit_vllm:
        async with submitit_vllm.get_client() as client:
            response = await client.responses.create(
                model=MODEL, input="Why is the sky blue?", max_output_tokens=128
            )
            print(response.model_dump_json(indent=2))

    with SubmititVLLM(
        logger_name_prefix="submitit_vllm.policy",
        executor_configs=executor_configs,
        engine_args=EngineArgs(
            model="/model-weights/Qwen3-8B",
            compilation_config=CompilationConfig(cache_dir=cache_dir),
            max_model_len=1024,
        ),
    ) as submitit_vllm:
        async with submitit_vllm.get_client() as client:
            response = await client.responses.create(
                model=MODEL, input="Why is the sky blue?", max_output_tokens=128
            )
            print(response.model_dump_json(indent=2))
