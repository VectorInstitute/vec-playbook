"""Launch script for DDP MLP training with Hydra + Submitit."""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from .train import DDPMLPTrainer


_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../configs")
)


@hydra.main(config_path=_CONFIG_PATH, config_name="_global", version_base=None)
def main(cfg: DictConfig):
    """Hydra entrypoint that merges local config and runs the Trainer."""
    local_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, local_cfg)

    if "trainer" in cfg:
        trainer_cfg = cfg.trainer
        cfg = OmegaConf.merge(cfg, trainer_cfg)
        del cfg.trainer

    ddp_trainer = DDPMLPTrainer()
    return ddp_trainer(cfg)


if __name__ == "__main__":
    main()
