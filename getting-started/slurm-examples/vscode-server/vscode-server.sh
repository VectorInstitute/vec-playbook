#!/bin/bash
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2:00:00
#SBATCH --output=vscode-server.%j.out

# Load the required module
module load code-server

# Start the VS Code server
# I've arbitrarily used port 9001 in this example. Any high number port should work.
code-server --bind-addr $(hostname --fqdn):9001 --auth none
