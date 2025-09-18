"""Launch script for checkpointable distributed finetuning with Hydra + Submitit."""

from __future__ import annotations

import os
import subprocess
import sys

import submitit
from omegaconf import OmegaConf


def _under_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ


def _running_inside_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def _slurm_world():
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    node_rank = int(os.environ.get("SLURM_NODEID", "0"))
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        master_addr = "127.0.0.1"
    else:
        out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
        master_addr = out.decode().splitlines()[0].strip()
    master_port = os.environ.get("MASTER_PORT", "29500")
    return nnodes, node_rank, master_addr, master_port


def _resolve_work_dir(cfg) -> str:
    env_dir = os.environ.get("HYDRA_LAUNCHER_RUN_DIR") or os.environ.get(
        "HYDRA_RUN_DIR"
    )
    if env_dir:
        return env_dir
    work_dir = getattr(cfg, "work_dir", None)
    if isinstance(work_dir, str) and "${" not in work_dir:
        return work_dir
    return os.getcwd()


def _save_resolved_config(cfg, work_dir: str) -> str:
    OmegaConf.set_struct(cfg, False)
    cfg.work_dir = work_dir
    if "paths" in cfg:
        cfg.paths["work_dir"] = work_dir
        cfg.paths["work_root"] = os.path.dirname(work_dir)
    base = os.path.basename(work_dir)
    if base.isdigit():
        cfg.experiment_name = os.path.basename(os.path.dirname(work_dir))
    else:
        cfg.experiment_name = base
    cfg_path = os.path.join(work_dir, "_fsdp_cfg.yaml")
    OmegaConf.save(cfg, cfg_path)
    return cfg_path


def _launch_torchrun(cfg, world_size: int, nproc_per_node: int) -> int:
    if world_size <= 1 or _under_torchrun() or not _running_inside_slurm():
        return 0
    nnodes, node_rank, master_addr, master_port = _slurm_world()
    work_dir = _resolve_work_dir(cfg)
    os.makedirs(work_dir, exist_ok=True)
    cfg_path = _save_resolved_config(cfg, work_dir)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--module",
        "llm.finetune_distributed.train",
        "--config",
        cfg_path,
    ]
    return subprocess.run(cmd, check=False).returncode


class DistributedLauncher(submitit.helpers.Checkpointable):
    """Submitit helper that spins up torchrun or falls back to a local run."""

    def __call__(self, cfg):
        """Dispatch the training job based on the selected distributed mode."""
        nnodes = int(getattr(cfg.compute, "nodes", 1))
        gpn = int(getattr(cfg.compute, "gpus_per_node", 1))
        world_size = nnodes * gpn

        if getattr(cfg.dist, "mode", "none") in {"ddp", "fsdp"}:
            return _launch_torchrun(cfg, world_size, gpn)

        work_dir = _resolve_work_dir(cfg)
        os.makedirs(work_dir, exist_ok=True)
        cfg_path = _save_resolved_config(cfg, work_dir)
        cmd = [
            sys.executable,
            "-m",
            "llm.finetune_distributed.train",
            "--config",
            cfg_path,
        ]
        return subprocess.run(cmd, check=False).returncode

    def checkpoint(self, *args, **kwargs):
        """Checkpoint the launcher so Submitit can requeue the job."""
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


__all__ = ["DistributedLauncher"]
