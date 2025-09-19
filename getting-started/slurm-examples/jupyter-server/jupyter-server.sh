#!/bin/bash
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2:00:00
#SBATCH --output=jupyter-server.%j.out

# Activate your virtual environment (make sure to build this first from the requirements.txt file!)
source jupyter-server-venv/bin/activate

# Start the jupyter server
jupyter notebook --ip $(hostname --fqdn) --no-browser
