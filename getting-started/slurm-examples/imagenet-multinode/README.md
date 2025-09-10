# Killarney Slurm Examples: Imagenet Multinode Training

This is a simple job for training a model using the Imagenet dataset, using a simple example provided by PyTorch (https://github.com/pytorch/examples/blob/main/imagenet/main.py).

In this example, we train for 3 epochs across 4x nodes, each using 2x L40S GPUs.

## Setup

Start by building your virtual environment under the `imagenet-multinode-venv` folder:
```
python3 -m venv imagenet-multinode-venv
source imagenet-multinode-venv
python3 -m pip install -r requirements.txt
```

Download the example script:
```
wget https://raw.githubusercontent.com/pytorch/examples/refs/heads/main/imagenet/main.py
```

## Submit the job

Submit using `sbatch`:
```
sbatch imagenet-multinode.sh
```
