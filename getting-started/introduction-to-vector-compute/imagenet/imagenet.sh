#!/bin/bash
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=2:00:00
#SBATCH --output=imagenet.%j.out
#SBATCH --error=imagenet.%j.err

echo "Imagenet training job starting at $(date +"%D %T")"

# Activate your virtual environment (make sure to build this first from the requirements.txt file!)
source imagenet-venv/bin/activate

# Run the training script. There are many optional arguments which are easy to find in the main.py file.
python3 main.py -a resnet18 --workers 8 --epochs 1 /datasets/imagenet

echo "Imagenet training job ending at $(date +"%D %T")"
