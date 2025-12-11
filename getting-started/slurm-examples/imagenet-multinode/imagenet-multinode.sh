#!/bin/bash
#SBATCH --job-name=imagenet-multinode
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=2:00:00
#SBATCH --output=imagenet-multinode.%j.out
#SBATCH --error=imagenet-multinode.%j.err

echo "Imagenet multinode training job starting at $(date +"%D %T")"

# We need the hostname address and an ephemeral port on the head (rank 0) node, which other workers will use to communicate with
export MASTER_ADDR=$(hostname --fqdn)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')

# The following needs to specify the exact path to srun on the worker nodes
SRUN_BIN="/cm/shared/apps/slurm/current/bin/srun"

# Important note about multinode configuration: even though we've been allocated 4 different nodes, this current sbatch script will only run on the head node. 
# We now need to use srun to get work running across all 4 nodes:
for index in $(seq 0 $((SLURM_JOB_NUM_NODES-1))); do
    $SRUN_BIN -lN$index --mem=64G --gres=gpu:l40s:2 -c 4 -N 1 -n 1 -r $index --output imagenet-multinode.$SLURM_JOB_ID.worker-$index.out bash -c "
	echo Hello Imagenet Multinode Worker! I am running on host:; hostname; \
    	source imagenet-multinode-venv/bin/activate; \
	python3 main.py -a resnet18 --dist-url 'tcp://$MASTER_ADDR:$MASTER_PORT' --dist-backend 'nccl' --epochs 3 --workers $SLURM_CPUS_ON_NODE --world-size $SLURM_NTASKS --rank $index --multiprocessing-distributed /datasets/imagenet" &
done

# The trailing & character after the srun command is very important. This allows the following `wait` command to block until all srun jobs are complete.
wait

echo "Imagenet multinode training job ending at $(date +"%D %T")"
