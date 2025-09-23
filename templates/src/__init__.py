"""Templates training ML models workloads on Vector cluster using Hydra and Submitit."""

# Give Submitit extra time for result pickles to land on slow filesystems.
import submitit


submitit.core.core.Job._results_timeout_s = max(
    submitit.core.core.Job._results_timeout_s, 120
)
