"""Multi-Layer Perceptron training templates."""

# Allow extra time for Submitit result pickles to land on shared filesystems.
import submitit


submitit.core.core.Job._results_timeout_s = max(
    submitit.core.core.Job._results_timeout_s, 120
)
