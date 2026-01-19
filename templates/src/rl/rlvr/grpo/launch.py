"""Launch script for GRPO, Hydra + Submitit."""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from .config import GRPOConfig
from .train import GRPOTrainer


logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run entrypoint that merges local config and runs the Trainer."""
    OmegaConf.set_struct(cfg, False)

    # Add output_directory for current run
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    cfg.paths.out_dir = str(os.path.join(hydra_config.runtime.output_dir, "outputs"))
    logger.info(f"Setting paths.out_dir to: {cfg.paths.out_dir}")

    # Save a resolved version of the hydra config
    save_path = os.path.join(
        hydra_config.runtime.output_dir,
        hydra_config.output_subdir,
        "hydra_resolved.yaml",
    )
    logger.info(f"Resolving hydra config for this run and saving to: {save_path}")
    OmegaConf.set_readonly(hydra_config, False)
    OmegaConf.resolve(hydra_config)
    OmegaConf.save(hydra_config, save_path)

    # Resolve the run config so interpolations are applied before logging/validation
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    grpo_config = GRPOConfig.model_validate(resolved_cfg["trainer"])  # type: ignore
    trainer = GRPOTrainer(grpo_config)
    metrics = trainer()
    for _epoch, (_item, _reward) in enumerate(metrics):
        print(f"epoch: {_epoch}, reward: {_reward:.03f}, metrics: {_item}")


if __name__ == "__main__":
    main()
