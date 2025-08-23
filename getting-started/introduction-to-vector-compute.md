# Introduction to Vector Compute

This guide covers important details and examples for accessing and using Vector's research compute environment. As of September 2025, our main research environment is the [Killarney cluster](https://docs.alliancecan.ca/wiki/Killarney) 


# Logging onto Killarney

The Alliance documentation provides lots of general information about accessing the Killarney cluster: [https://docs.alliancecan.ca/wiki/SSH](https://docs.alliancecan.ca/wiki/SSH)

## Getting an Account

Please read the [user account guide](https://support.vectorinstitute.ai/Killarney?action=AttachFile&do=view&target=User+Guide+to+Killarney+for+Vector+Researchers.pdf) for full information about getting a Killarney account.

## SSH Access

From a terminal, use the `ssh` command to log onto the cluster via [killarney.alliancecan.ca](killarney.alliancecan.ca):


```
username@my-desktop:~$ ssh killarney_username@killarney.alliancecan.ca
Duo two-factor login for username

Enter a passcode or select one of the following options:

 1. Duo Push to Phone

Passcode or option (1-1): 1
Success. Logging you in...

#####################################################################
 _  ___ _ _
| |/ (_) | | __ _ _ __ _ __   ___ _   _
| ' /| | | |/ _' | '__| '_ \ / _ \ | | |
| . \| | | | (_| | |  | | | |  __/ |_| |
|_|\_\_|_|_|\__,_|_|  |_| |_|\___|\__, |
                                  |___/
     Killarney AI Compute Cluster
          Support: support@tech.alliancecan.ca
          Documentation: https://docs.alliancecan.ca

####################################################################

username@klogin02:~$
```


The hostname **killarney.alliancecan.ca** load balances ssh connections across the **klogin01**, **klogin02**, **klogin03** and **klogin04** machines. This provides for spreading out load and some redundancy.

**NOTE:** These "klogin" nodes are shared across all users and do not have GPUs or a lot of compute power. Please do not run any training jobs or compile code here! In the following sections we will go through how to run your jobs on the cluster.


## Public Key Setup

To avoid entering a password every time you connect, you can optionally set up a SSH public key in your Alliance Canada account. 

For some operations, such as migrating data from the Bon Echo (Vaughan) cluster, the SSH key is mandatory.

On the computer you'll be connecting from, generate a SSH key pair with the following command. When prompted, use the default file name and leave the passphrase empty.

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```


Output the key into your terminal window and output the key:


```
cat ~/.ssh/id_ed25519.pub
```


The key will look something like the following. Copy it to your clipboard:


```
ssh-ed25519 AAAAC3NzaC1lZDI1GTE5AA6AIEX4zziRB7OZOZ7NR58NUi3iLC0fgCArMe9KZyS8z6Dz username@vectorinstitute.ai
```


Next, open the SSH Keys page in your Alliance account: [https://ccdb.alliancecan.ca/ssh_authorized_keys](https://ccdb.alliancecan.ca/ssh_authorized_keys). Paste your key into the SSH Key field, give it a name (typically the host name of the computer where you generated it) and hit Add Key.


# Killarney File System Intro


## Home directories

When you first log onto the Killarney cluster, you will land in your home directory. This can be accessed at: `/home/username `or just` ~/`

Home directories have 50 GB of storage space. To check the amount of free space in your home directory, use the` diskusage_report `command:


```
username@klogin02:~$ diskusage_report
                            Description                Space         # of files
                  /home (user username)         0  B/  50GiB          13 / 500K
               /scratch (user username)        51GiB/2048GiB         341K/  10M
```



## Scratch space

In addition to your home directory, you have a minimum of additional 250 GB scratch space (up to 2 TB, depending on your user level) available in the following location: `/scratch/$USER` or simply` $SCRATCH.`

**Unlike your home directory, this scratch space is temporary. It will get automatically purged of files that have not been accessed in 60 days.**

A detailed description of the scratch purging policy is available on the Alliance Canada website: [https://docs.alliancecan.ca/wiki/Scratch_purging_policy](https://docs.alliancecan.ca/wiki/Scratch_purging_policy)


## Shared projects

For collaborative projects where many people need access to the same files, you need a shared project space. These are generally stored at `/project`

To set up a shared project space, send a request to [ops-help@vectorinstitute.ai](mailto:ops-help@vectorinstitute.ai). Describe what the project is about, which users need access, how much disk space you need, also an end date when it can be removed.


## Shared datasets

To reduce the storage footprint for each user, we've made various commonly-used datasets like MIMIC and IMAGENET available for everyone to use. These are generally stored at /datasets

Instead of copying these datasets on your home directory, you can create a symlink via


```
ln -s /dataset/PATH_TO_DATASET ~/PATH_OF_LINK # path of link can be some place in your home directory so that PyTorch/TF can pick up the dataset to these already downloaded directories.
```


For a list of available datasets please see [Current Datasets](https://support.vectorinstitute.ai/CurrentDatasets)


## Shared model weights

Similar to datasets, model weights are typically very large and can be shared among many users. We've made various common model weights such as Llama3, Mixtral and Stable Diffusion available at /`model-weights`


## Training checkpoints

Unlike the legacy Bon Echo (Vaughan) cluster, there is no dedicated checkpoint space in the Killarney cluster. Now that the $SCRATCH space has been greatly expanded, please use this for any training checkpoints.


# Migration from legacy Vaughan (Bon Echo) Cluster

The easiest way to migrate data from the legacy Vaughan (Bon Echo) Cluster to Killarney is by usinghttps://support.vectorinstitute.ai/Killarney?action=AttachFile&do=view&target=User+Guide+to+Killarney+for+Vector+Researchers.pdfk rsync.

Start by connecting via shttps://support.vectorinstitute.ai/Killarney?action=AttachFile&do=view&target=User+Guide+to+Killarney+for+Vector+Researchers.pdfsh into the legacy Vaughan cluster:


```
username@mark-desktop:~$ ssh v.vectorinstitute.ai
Password:
Duo two-factor login for username

Enter a passcode or select one of the following options:

 1. Duo Push to XXX-XXX-3089
 2. SMS passcodes to XXX-XXX-3089

Passcode or option (1-2): 1
Success. Logging you in...
Welcome to the Vector Institute HPC - Vaughan Cluster

Login nodes are shared among many users and therefore
must not be used to run computationally intensive tasks.
Those should be submitted to the slurm scheduler which
will dispatch them on compute nodes.

For more information, please consult the wiki at
  https://support.vectorinstitute.ai/Computing
For issues using this cluster, please contact us at
  ops-help@vectorinstitute.ai
If you forget your password, please visit our self-
  service portal at https://password.vectorinstitute.ai.

Last login: Mon Aug 18 07:28:24 2025 from 184.145.46.175
```


Next, use the `rsync` command to copy files across to the Killarney cluster. In the following example, I'm copying the contents of a folder called `my_projects` to my Killarney home directory.


```
username@v4:~$ cd ~/my_projects
username@v4:~/my_projects$ rsync -avz * killarney_username@killarney.alliancecan.ca:~/my_projects~
Duo two-factor login for username

Enter a passcode or select one of the following options:

 1. Duo Push to Phone

Passcode or option (1-1): 1
Success. Logging you in...
sending incremental file list
[...]
```

# Killarney GPU resources

There are two main types of GPU resources on the Killarney cluster: capacity GPUs (NVIDIA L40S) and high-performance GPUs (NVIDIA H100). 

A full description of compute resources is available on the Alliance website: [https://docs.alliancecan.ca/wiki/Killarney#Killarney_hardware_specifications](https://docs.alliancecan.ca/wiki/Killarney#Killarney_hardware_specifications)

Since the cluster has many users and limited resources, we use the Slurm job scheduler ([https://slurm.schedmd.com/documentation.html](https://slurm.schedmd.com/documentation.html)) to schedule and run user-requested jobs in a "fair" way.


# Using Slurm

The Alliance documentation provides lots of general information about submitting jobs using the Slurm job scheduler: [https://docs.alliancecan.ca/wiki/Running_jobs](https://docs.alliancecan.ca/wiki/Running_jobs)	


## View jobs in the Slurm cluster (squeue)

To view all the jobs currently in the cluster, either running, pending or failed, use squeue ([https://slurm.schedmd.com/squeue.html](https://slurm.schedmd.com/squeue.html))


```
username@klogin01:~$ squeue
          JOBID     USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) 
         505480 username     aip-acct        jupyter   R      41:31     1    4 gres/gpu:l    300G kn141 (None) 
       504615_8 username     aip-acct         My_JOB   R      50:05     1   32 gres/gpu:l     64G kn054 (None) 
       504615_9 username     aip-acct         My_JOB   R      53:07     1   32 gres/gpu:l     64G kn038 (None) 
      504615_10 username     aip-acct         My_JOB   R      54:07     1   32 gres/gpu:l     64G kn039 (None) 
         499132 username     aip-acct SFBDF_cifar10_   R      54:45     1   32 gres/gpu:3     60G kn135 (None) 
         505622 username     aip-acct        tus-rec   R      58:39     1   10 gres/gpu:l    128G kn001 (None) 
      504615_11 username     aip-acct         My_JOB   R    1:00:48     1   32 gres/gpu:l     64G kn041 (None) 
      504615_12 username     aip-acct         My_JOB   R    1:03:37     1   32 gres/gpu:l     64G kn008 (None) 
         504859 username     aip-acct         sbatch   R    1:06:59     1   16 gres/gpu:l     64G kn034 (None)
[...]
```


There are many different options for the squeue command. For example, to view jobs that are pending and waiting to start running, use the `-t PENDING` flag:


```
username@login01:~$ squeue -t PENDING
          JOBID     USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) 
  505518_[5-48] username     aip-acct         My_JOB  PD    4:00:00     1   32 gres/gpu:l     64G  (Resources) 
         505759 username     aip-acct grpg_qwen3_4b_  PD 1-00:00:00     1    8 gres/gpu:h     32G  (Resources) 
         505739 username     aip-acct lora_with_rgb_  PD   16:00:00     1   12 gres/gpu:h    256G  (Priority) 
         505714 username     aip-acct base_ff_96_fra  PD 1-00:00:00     1   12 gres/gpu:h    256G  (Priority) 
         505717 username     aip-acct base_ff_128_fr  PD 1-00:00:00     1   12 gres/gpu:h    256G  (Priority) 
         505300 username     aip-acct vanilla_vq_ima  PD 3-00:00:00     1   10 gres/gpu:h     50G  (Resources)
[...]
```


## Submit a new Slurm job (sbatch)

To ask Slurm to run your jobs in the background so you can have your job running, even after logging off, use sbatch https://slurm.schedmd.com/sbatch.html

To use sbatch, you need to create a file, specify the configurations within (you can also specify these on the command line) and then run `sbatch my_sbatch_slurm.sh` to get Slurm to schedule it.

Example Hello World sbatch file (hello_world.sh):


```
#!/bin/bash
#SBATCH --job-name=hello_world_example
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=hello_world.%j.out
#SBATCH --error=hello_world.%j.err

# Prepare your environment here
module load python/3.12.4

# Use python to write "Hello World"
python3 -c 'print("Hello World")'
```

Submit the job:

```
username@klogin02:~$ sbatch hello_world.sh
```

Note that the sbatch configurations all start with `#SBATCH`. The above script asks for 1 L40S GPU, 4 CPU cores, 32GB of memory and a one-hour hour time limit.

Since Slurm runs your job in the background, it becomes really difficult to see the standard output and standard error of your job in case anything crashes or your results are printed. So Slurm has an option to specify where to pipe the standard output via (`#SBATCH --output=hello_world.%j.out`) and standard error (`#SBATCH --error=hello_world.%j.err`). You can specify them to be any place and directory.

Note that the %j in output and error configuration tells Slurm to substitute the job ID where the %j is. So if your job ID is 1234 then your output file will be `hello_world.1234.out` and your error file will be `hello_world.1234.err`.


## Interactive sessions (srun)

To get an interactive session, you must use srun see documentation here.

A common configuration for interactive debugging is:

```
srun --gres=gpu:l40s:1 --mem=32GB --time=2:00:00 --pty bash
```

This tells Slurm you want 1 L40S GPU (`--gres=gpu:l40s:1`), 32GB of CPU ram (`--mem=32G`), for two hours (`--time=2:00:00`), using a pseudo-terminal (`--pty`), and you want to launch `bash` (the first argument to the srun command that isn't a valid Slurm option) on the compute node.

When srun returns, you should be able to see **username@kn###**:

```
username@klogin02:/project$ srun --gres=gpu:l40s:1 --mem=32GB --time=2:00:00 --account=aip-x-ksupport --pty bash
srun: job 501831 queued and waiting for resources
srun: job 501831 has been allocated resources
username@kn138:/project
```

After you see $USER@kn###, you can run your script interactively. It is also possible to 'attach' a new shell session to an existing jobID (either batch or interactive) using the the srun option of --overlap.


## Accessing specific GPUs



When SLURM schedules our jobs, it will give us an unique job ID. So it is possible to keep track of your jobs by querying SLURM.


## View cluster resource utilization (sinfo)

To see the availability in a more granular scale use sinfo ([https://slurm.schedmd.com/sinfo.html](https://slurm.schedmd.com/sinfo.html)). For example:


```
sinfo -N --Format=Partition,CPUsState,GresUsed,Gres
```

Partitions and number of resources:

```
username@klogin02:~$ sinfo -N --Format=Partition,CPUsState,GresUsed,Gres
PARTITION           CPUS(A/I/O/T)       GRES_USED           GRES
gpubase_l40s_b1     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b2     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b3     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b1     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b2     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b3     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b1     0/0/64/64           gpu:l40s:0(IDX:N/A) gpu:l40s:4
gpubase_l40s_b2     0/0/64/64           gpu:l40s:0(IDX:N/A) gpu:l40s:4
gpubase_l40s_b3     0/0/64/64           gpu:l40s:0(IDX:N/A) gpu:l40s:4
gpubase_l40s_b1     64/0/0/64           gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b2     64/0/0/64           gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b3     64/0/0/64           gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b1     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b2     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
gpubase_l40s_b3     32/32/0/64          gpu:l40s:4(IDX:0-3) gpu:l40s:4
[...]
```

## Jupyter notebooks

To run a Jupyter environment from the cluster, you can request an interactive session and start a Jupyter notebook from there.

First, log into the cluster via ssh to killarney.alliancecan.ca:

```
ssh username@killarney.alliancecan.ca
```

Jupyter is not installed by default. You need to install it yourself by running:

```
pip install jupyter
export PATH=$PATH:~/.local/bin
```

Make sure you have a .jupyter folder in your home directory:

```
mkdir -p ~/.jupyter
```

Next, use srun to request an interactive session. The following example asks for an A40 GPU and 32 GB of system memory.

```
srun --gres=gpu:l40s:1 --mem=32G --time 2:00:00 --pty bash
```

Now start the notebook:

```
username@kn135:~$ jupyter notebook --ip 0.0.0.0
[...]
[I 2025-08-20 11:52:22.038 ServerApp] Serving notebooks from local directory: /scratch/markcoat/slurm/helloworld
[I 2025-08-20 11:52:22.038 ServerApp] Jupyter Server 2.16.0 is running at:
[I 2025-08-20 11:52:22.038 ServerApp] http://kn135:8888/tree?token=3c2a490424359c8ee69d37c19c38aa27c5f02f61a1b77ecf
[I 2025-08-20 11:52:22.038 ServerApp]     http://127.0.0.1:8888/tree?token=3c2a490424359c8ee69d37c19c38aa27c5f02f61a1b77ecf
[I 2025-08-20 11:52:22.038 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2025-08-20 11:52:22.054 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///home/markcoat/.local/share/jupyter/runtime/jpserver-942891-open.html
    Or copy and paste one of these URLs:
        http://kn135:8888/tree?token=3c2a490424359c8ee69d37c19c38aa27c5f02f61a1b77ecf
        http://127.0.0.1:8888/tree?token=3c2a490424359c8ee69d37c19c38aa27c5f02f61a1b77ecf

You will need a VPN connection to access this notebook. Once you are connected to the VPN, visit the URL beginning with http://kn####, so in the example above this would be: http://kn135:8888/tree?token=3c2a490424359c8ee69d37c19c38aa27c5f02f61a1b77ecf
```


# Software Environments

The cluster comes with preinstalled software environments called **modules**. These will allow you to access many different versions of Python, VS Code Server, RStudio Server, NodeJS and many others. 

To see the available preinstalled environments, run:

```
module avail
```

To use an environment, use `module load`. For example, if you need to use Python 3.10, run the following:

```
module load python/3.10.12
```

If there isn't a preinstalled environment for your needs, you can use Poetry or python-venv. Here is a quick example of how to use python venv.

In the login node run the following:

```
python3 -m venv some_env
```

You can replace some_env with any path you want your environment to be installed in.

Then to activate this environment run

```
source some_env/bin/activate
```

and it should show the following

```
(some_env) username@kn130:~$
```

Now you can `pip install` anything you need and it will be contained in this environment.


# Time Limits and Checkpoints

The cluster uses time limits to ensure fair access to resources.

Preemption allows users to run high priority jobs (jobs in the deadline or high QOS) as soon as possible. If the cluster is full and there is no room for a job using one of these QOS to start, it will select the lowest priority job (or combination of jobs) that would allow the high priority job to run, that has run for at least two hours (the default PreemptExemptTime) and stop it, putting it back into the queue, to make room for the high priority job to start.

All jobs in our Slurm cluster have a time limit, after which they will get stopped. For longer running jobs which need more than a few hours, the Vaughan_slurm_changes document describes how to automatically restart these.

In order to avoid losing your work when your job is preempted, you will need to implement checkpoints - periodic snapshots of your work that you load from so you can stop and resume without much lost work.

Please see Checkpoint Example for more information.


# Useful Links and Resources

Computing parent page: https://support.vectorinstitute.ai/Computing

Vaughan valid partition/qos: https://support.vectorinstitute.ai/Vaughan_slurm_changes

Checkpointing: https://support.vectorinstitute.ai/CheckpointExample

Slurm Scheduler: https://support.vectorinstitute.ai/slurm_fairshare

FAQ: https://support.vectorinstitute.ai/FAQ%20about%20the%20cluster


# Support

For any cluster issues please email ops-help@vectorinstitute.ai.

For engineering/checkpointing related issues please check out the #computing channel on slack. You may also ask quick cluster questions here.