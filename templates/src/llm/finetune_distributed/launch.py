"""Launch script for checkpointable distributed finetuning with Hydra + Submitit."""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from .train import FinetuneDistributedTrainer


logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Hydra entrypoint that updates paths, saves config, and launches training."""
    # Turn of struct mode so that we can modify DictConfig
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

    # Run the trainer with the run config
    trainer = FinetuneDistributedTrainer()
    return trainer(cfg)


if __name__ == "__main__":
    main()
