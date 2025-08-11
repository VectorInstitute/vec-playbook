"""Ready-to-run training examples launchable via vec-tool."""

# Only import what's actually needed for the HF trainer
from .llm_fine_tuning import HFTextClassificationTrainer


__all__ = ["HFTextClassificationTrainer"]
