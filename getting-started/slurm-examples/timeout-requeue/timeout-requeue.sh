#!/bin/bash
#SBATCH --job-name=timeout-requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --output=timeout-requeue.%j.out
# Send this batch script a SIGUSR1 60 seconds before we hit our time limit
#SBATCH --signal=B:USR1@60

# Signal handler: do any cleanup necessary, then requeue the job
handler()
{
	echo "SIGUSR1 function handler called at $(date)"
	# Do any cleanup you want here; checkpoint, sync, etc
	# ...
	# (Optional) Also forward the signal to the python script
	# In this example, the python script is responsible for checkpoints
	scancel --signal=USR1 "$SLURM_JOB_ID"
	# Now requeue the job
	sbatch ${BASH_SOURCE[0]}
}

# Register signal handler
trap handler SIGUSR1

# Now run the actual work
srun ./time-loop.py &

wait
