"""Typing and validator for GRPO config."""

import datetime
from pathlib import Path

import pydantic


class SubmititArgs(pydantic.BaseModel):
    """Submitit args."""

    job_name: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    mem: str | None = "72GB"
    cpus_per_task: str | None = "16"
    gres: str | None = "gpu:1"
    time: str = "1:00:00"
    use_srun: bool = False

    def to_submitit_parameters(self) -> dict[str, int | str]:
        """Produce submit-compatible dict consisting of non-None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    @property
    def time_in_minutes(self) -> float:
        """Return self.time as a float."""
        d = 0
        time = self.time
        if "-" in time:
            _days, time = time.split("-")
            d = int(_days)

        h, m, s = map(int, time.split(":"))
        return (
            datetime.timedelta(days=d, hours=h, minutes=m, seconds=s).total_seconds()
            / 60
        )


class GRPOHyperparameters(pydantic.BaseModel):
    """Hyperparameters for GRPO."""

    max_model_len: int

    batch_size_forward: int
    batch_size_backprop: int

    grad_acc_steps: int

    learning_rate: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_weight_decay: float = 0.0


class VLLMConfigs(pydantic.BaseModel):
    """Configs for vLLM and SLURM rollout for one model.

    E.g., one config for policy, and one for evaluator.
    """

    max_model_len: int
    num_replicas: int
    concurrency_per_replica: int

    cache_dir: Path

    uv_venv: Path
    submitit_args: SubmititArgs
    submitit_python: str | None = None


class LLMJudgeConfig(VLLMConfigs):
    """Configs for evaluator model for LLM as a judge."""

    model_name: str


class DataConfig(pydantic.BaseModel):
    """Configs for GRPO RLVR data."""

    dataset_name: str
    subset: str | None = None
    train_split: str = "train"
    test_split: str = "test"

    query_column: str = "question"
    target_column: str = "answer"
    target_regexp: str | None = None


class GRPOConfig(pydantic.BaseModel):
    """Typing for GRPO config."""

    base_model: str
    tokenizer_name: str
    checkpoint_folder: Path
    keep_all_checkpoints: bool

    submitit_logs_folder: Path

    rollout_vllm: VLLMConfigs
    llm_judge_vllm: LLMJudgeConfig

    data: DataConfig
    hyperparameters: GRPOHyperparameters

    optimizer_folder: Path | None = None
    num_epochs: int