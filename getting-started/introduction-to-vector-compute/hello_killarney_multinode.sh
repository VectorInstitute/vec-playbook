#!/bin/bash
#SBATCH --job-name=hello_killarney_multinode
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --nodes=4 
#SBATCH --ntasks-per-node=1
#SBATCH --output=hello_killarney_multinode.%j.out
#SBATCH --error=hello_killarney_multinode.%j.err

# Prepare your environment here
module load python/3.12.4
echo "Python version:"
python3 --version

# Host name:
echo "Hello Killarney Multinode! I am running on host $(hostname)"

# GPU configuration:
echo "This is the GPU configuration on my head node:"
nvidia-smi

# The following needs to specify the exact path to srun on the worker nodes
SRUN_BIN="/cm/shared/apps/slurm/current/bin/srun"

# Important note about multinode configuration: even though we've been allocated 4 different nodes, this current sbatch script will only run on the head node. We now need to use srun to get work running across all 4 nodes:
for index in $(seq 0 $((SLURM_JOB_NUM_NODES-1))); do
    $SRUN_BIN -lN$index --mem=1G --gres=gpu:1 -c 4 -N 1 -n 1 -r $index --output hello_killarney_multinode.$SLURM_JOB_ID.worker-$index.out bash -c "echo Hello Killarney Multinode Worker! I am running on host:; hostname" &
done

wait

