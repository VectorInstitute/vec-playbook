# Killarney Slurm Examples: Timeout Requeue

All jobs on Vector's Killarney cluster are subject to time limits. You need to specify a limit when submitting a job, for example by using `--time=2:00:00` to set a limit of 2 hours, after which your job will be automatically stopped by the Slurm scheduler. The maximum time limit is based on the compute resources requested. To see a list of the various tiers, run `sinfo --summarize`.

This example shows how to set up a job that automatically requeues after hitting its time limit, while also saving its current progress to a checkpoint.

## Submit the job

Submit using `sbatch`:
```
sbatch timeout-requeue.sh
```
