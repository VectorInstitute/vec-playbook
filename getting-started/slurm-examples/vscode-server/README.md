# Killarney Slurm Examples: VS Code Server

This is a basic script for setting up a VS Code server in a Slurm job.

It leverages the `code-server` module that is available on the Killarney cluster, so no software installation is needed.

## Submit the job

Submit using `sbatch`:
```
sbatch vscode-server.sh
```

## Connect to the server

First, we need to know the hostname of the compute node where the server is running. This will be published to the .out file once the server is up and running:
```
cat vscode-server*.out | grep "HTTP server listening"
```

This will reveal a URL path that looks like http://10.1.1.97:9001/. We need to set up a SSH tunnel to that host.

From your local workstation, use the following command to start the tunnel. In this example I'm connecting to the above host, `10.1.1.97`:
```
ssh username@killarney.alliancecan.ca -L 9001:10.1.1.97:9001
```

Now you can access the server at http://localhost:9001

