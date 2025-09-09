#!/bin/bash
#SBATCH --job-name=hello-killarney
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --output=hello-killarney.%j.out
#SBATCH --error=hello-killarney.%j.err

# Prepare your environment here
module load python/3.12.4
echo "Python version:"
python3 --version

# Use python to write "Hello Killarney"
python3 -c 'print("Hello Killarney")'
