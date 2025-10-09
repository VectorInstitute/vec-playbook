"""Launch script for checkpointable distributed finetuning with Hydra + Submitit."""

import os

import hydra
from hydra.core.hydra_config import HydraConfig
from llm.finetune_distributed.distributed_launcher import DistributedLauncher
from omegaconf import DictConfig, OmegaConf


_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../configs")
)


@hydra.main(config_path=_CONFIG_PATH, config_name="_global", version_base=None)
def main(cfg: DictConfig):
    """Hydra entrypoint that merges configs before launching training."""
    local_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    local_cfg = OmegaConf.load(local_cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(local_cfg, cfg)

    hydra_run_dir = HydraConfig.get().runtime.output_dir
    if hydra_run_dir is not None:
        cfg.work_dir = hydra_run_dir
        if "paths" in cfg:
            cfg.paths.work_dir = hydra_run_dir
            cfg.paths.work_root = os.path.dirname(hydra_run_dir)

    if "trainer" in cfg:
        trainer_cfg = cfg.trainer
        cfg = OmegaConf.merge(cfg, trainer_cfg)
        del cfg.trainer

    runner = DistributedLauncher()
    return runner(cfg)


if __name__ == "__main__":
    main()
