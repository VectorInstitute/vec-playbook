"""Entrypoint for training."""

import importlib

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="_global", version_base=None)
def main(cfg: DictConfig):
    """Initialize the runner and run the training."""
    # Dynamically import the correct starter.
    mod = importlib.import_module(cfg.starter.module)
    runner_cls = getattr(mod, cfg.starter.entry)
    runner = runner_cls()
    return runner(cfg)


if __name__ == "__main__":
    main()
