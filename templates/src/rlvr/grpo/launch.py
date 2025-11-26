"""Launch script for GRPO, Hydra + Submitit."""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from templates.src.rlvr.grpo.config import GRPOConfig
from templates.src.rlvr.grpo.train import GRPOTrainer


_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../configs")
)
log = logging.getLogger(__name__)


@hydra.main(config_path=_CONFIG_PATH, config_name="_global", version_base=None)
def main(cfg: DictConfig):
    """Run entrypoint that merges local config and runs the Trainer."""
    local_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
    cfg = OmegaConf.merge(local_cfg, cfg)  # type: ignore
    OmegaConf.set_struct(cfg, False)

    if "trainer" in cfg:
        trainer_cfg = cfg.trainer
        cfg = OmegaConf.merge(cfg, trainer_cfg)  # type: ignore

    grpo_config = GRPOConfig.model_validate(cfg.__dict__["_content"]["trainer"])
    trainer = GRPOTrainer(grpo_config)
    metrics = trainer(grpo_config)
    for _epoch, (_item, _reward) in metrics:
        print(f"epoch: {_epoch}, reward: {_reward:.03f}, metrics: {_item}")


if __name__ == "__main__":
    main()
