# Killarney Slurm Examples: Imagenet Training

This is a simple job for training the a model using the Imagenet dataset, using a simple example provided by PyTorch (https://github.com/pytorch/examples/blob/main/imagenet/main.py).

In this example, we train for a single epoch on a single node using 2x L40S GPUs.

## Setup

Start by building your virtual environment under the `imagenet-venv` folder:
```
python3 -m venv imagenet-venv
source imagenet-venv
python3 -m pip install -r requirements.txt
```

Download the example script:
```
wget https://raw.githubusercontent.com/pytorch/examples/refs/heads/main/imagenet/main.py
```

## Submit the job

Submit using `sbatch`:
```
sbatch imagenet.sh
```
