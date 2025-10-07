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
# BUG: The module code is running under /bin/sh which is wrong. Let's manually call /bin/bash until this gets fixed
#code-server --bind-addr $(hostname --fqdn):9001 --auth none
/bin/bash /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/code-server/4.101.2/bin/code-server --bind-addr $(hostname --fqdn):9001 --auth none
