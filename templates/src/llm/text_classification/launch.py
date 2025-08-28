"""Launch script to run template with Hydra + Submitit."""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from .train import TextClassificationTrainer


_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../configs")
)


@hydra.main(config_path=_CONFIG_PATH, config_name="_global", version_base=None)
def main(cfg: DictConfig):
    """Hydra entrypoint that merges local config and runs the Trainer."""
    local_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
    cfg = OmegaConf.merge(cfg, local_cfg)

    text_classification_trainer = TextClassificationTrainer()
    return text_classification_trainer(cfg)


if __name__ == "__main__":
    main()
