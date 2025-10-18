# Killarney Slurm Examples: Jupyter Server

This is a basic script for setting up a Jupyter server in a Slurm job.

## Setup

Start by building your virtual environment under the `jupyter-server-venv` folder:
```
python3 -m venv jupyter-server-venv
source jupyter-server-venv
python3 -m pip install -r requirements.txt
```

## Submit the job

Submit using `sbatch`:
```
sbatch jupyter-server.sh
```

## Connect to the server

First, we need to know the hostname of the compute node where the server is running. This will be published to the .out file once the server is up and running:
```
cat jupyter*.out | grep vectorinstitute
```

This will reveal a URL path with a hostname that looks like `kn###.paice.vectorinstitute.ai`. We need to set up a SSH tunnel to that `kn###` host.

From your local workstation, use the following command to start the tunnel. In this example I'm connecting to `kn055`:
```
ssh username@killarney.alliancecan.ca -L 8888:kn055:8888
```

Now retrieve the local URL for the Jupyter server:
```
cat jupyter*.out | grep "127.0.0.1"
```

This will return a URL that looks something like `http://127.0.0.1:8888/tree?token=83b4c044abaf37fc9d31e9efd99922b2a3c7a8eae35aa8d9`. Open this in a web browser on your local workstation to access the Jupyter environment.

