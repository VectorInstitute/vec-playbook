# Introduction to Vector Compute

This guide covers important details and examples for accessing and using Vector's research compute environment. As of September 2025, our main research environment is the [Killarney cluster](https://docs.alliancecan.ca/wiki/Killarney)

# Table of Contents

- [Introduction to Vector Compute](#introduction-to-vector-compute)
- [Table of Contents](#table-of-contents)
- [Logging onto Killarney](#logging-onto-killarney)
  - [Getting an Account](#getting-an-account)
  - [Public Key Setup](#public-key-setup)
  - [SSH Access](#ssh-access)
- [Killarney File System Intro](#killarney-file-system-intro)
  - [Home directories](#home-directories)
  - [Scratch space](#scratch-space)
  - [Shared projects](#shared-projects)
  - [Shared datasets](#shared-datasets)
  - [Shared model weights](#shared-model-weights)
  - [Training checkpoints](#training-checkpoints)
- [Migration from legacy Vaughan (Bon Echo) Cluster](#migration-from-legacy-vaughan-bon-echo-cluster)
- [Killarney GPU resources](#killarney-gpu-resources)
- [Using Slurm](#using-slurm)
  - [View jobs in the Slurm cluster (squeue)](#view-jobs-in-the-slurm-cluster-squeue)
  - [Submit a new Slurm job (sbatch)](#submit-a-new-slurm-job-sbatch)
  - [Interactive sessions (srun)](#interactive-sessions-srun)
  - [SSH to sbatch job](#ssh-to-sbatch-job)
  - [Accessing specific GPUs](#accessing-specific-gpus)
  - [View cluster resource utilization (sinfo)](#view-cluster-resource-utilization-sinfo)
- [Software Environments](#software-environments)
- [Time Limits](#time-limits)
  - [Tiers](#tiers)
  - [Automatic Restarts](#automatic-restarts)
  - [Checkpoints](#checkpoints)
- [Useful Links and Resources](#useful-links-and-resources)
- [Support](#support)

# Logging onto Killarney

The Alliance documentation provides lots of general information about accessing the Killarney cluster: [https://docs.alliancecan.ca/wiki/SSH](https://docs.alliancecan.ca/wiki/SSH)

## Getting an Account

Please read the [user account guide](https://support.vectorinstitute.ai/Killarney?action=AttachFile&do=view&target=User+Guide+to+Killarney+for+Vector+Researchers.pdf) for full information about getting a Killarney account.

## Public Key Setup

For SSH access, you need to add a public key in your Alliance Canada account.

On the computer you'll be connecting from, generate a SSH key pair with the following command. When prompted, use the default file name and leave the passphrase empty.

```
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Output the key into your terminal window:

```
cat ~/.ssh/id_ed25519.pub
```

The key will look something like the following. Copy it to your clipboard:

```
ssh-ed25519 AAAA5AA7OZOZ7NRB1acK54bB47h58N6AIEX4zDziR1r0nM41d3NCG0fgCArjUD45pr13578zo0z username@vectorinstitute.ai
```

Next, open the SSH Keys page in your Alliance account: [https://ccdb.alliancecan.ca/ssh_authorized_keys](https://ccdb.alliancecan.ca/ssh_authorized_keys). Paste your key into the SSH Key field, give it a name (typically the host name of the computer where you generated it) and hit Add Key.


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


# Killarney File System Intro


## Home directories

When you first log onto the Killarney cluster, you will land in your home directory. This can be accessed at: `/home/username `or just` ~/`

Home directories have 50 GB of storage space. To check the amount of free space in your home directory, use the` diskusage_report `command:


```
username@klogin02:~$ diskusage_report
                            Description                Space         # of files
                  /home (user username)         0  B/  50GiB          13 / 500K
               /scratch (user username)         51GiB/256GiB         341K/  10M
```

## Scratch space

In addition to your home directory, you have a minimum of additional 250 GB scratch space (up to 2 TB, depending on your user level) available in the following location: `/scratch/$USER` or simply` $SCRATCH.`

**⚠️ Unlike your home directory, this scratch space is temporary. It will get automatically purged of files that have not been accessed in 60 days.**

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

Unlike the legacy Bon Echo (Vaughan) cluster, there is no dedicated checkpoint space in the Killarney cluster. Now that the `$SCRATCH` space has been greatly expanded, please use this for any training checkpoints.


# Migration from legacy Vaughan (Bon Echo) Cluster

The easiest way to migrate data from the legacy Vaughan (Bon Echo) Cluster to Killarney is by using a file transfer command (likely `rsync` or `scp`) from an SSH session.

Start by connecting via https://support.vectorinstitute.ai/Killarney?action=AttachFile&do=view&target=User+Guide+to+Killarney+for+Vector+Researchers.pdfsh into the legacy Bon Echo (Vaughan) cluster:


```
username@my-desktop:~$ ssh v.vectorinstitute.ai
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

For some example Slurm workloads specific to the Killarney cluster (sbatch files, resource configurations, software environments, etc.) see the (../slurm-examples)[slurm-examples] provided in this repo.


## View jobs in the Slurm cluster (squeue)

To view all the jobs currently in the cluster, either running, pending or failed, use **squeue**: ([https://slurm.schedmd.com/squeue.html](https://slurm.schedmd.com/squeue.html))


```
username@klogin01:~$ squeue
          JOBID     USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) 
         505480    user1     aip-acct        jupyter   R      41:31     1    4 gres/gpu:l    300G kn141 (None) 
       504615_8    user2     aip-acct         My_JOB   R      50:05     1   32 gres/gpu:l     64G kn054 (None) 
       504615_9    user2     aip-acct         My_JOB   R      53:07     1   32 gres/gpu:l     64G kn038 (None) 
      504615_10    user2     aip-acct         My_JOB   R      54:07     1   32 gres/gpu:l     64G kn039 (None) 
         499132    user3     aip-acct SFBDF_cifar10_   R      54:45     1   32 gres/gpu:3     60G kn135 (None) 
         505622    user4     aip-acct        tus-rec   R      58:39     1   10 gres/gpu:l    128G kn001 (None) 
      504615_11    user5     aip-acct         My_JOB   R    1:00:48     1   32 gres/gpu:l     64G kn041 (None) 
      504615_12    user5     aip-acct         My_JOB   R    1:03:37     1   32 gres/gpu:l     64G kn008 (None) 
         504859    user6     aip-acct         sbatch   R    1:06:59     1   16 gres/gpu:l     64G kn034 (None)
[...]
```

There are many different options for the squeue command. For example, to view only jobs that belong to me, use the `--me` flag:

```
username@login01:~$ $ squeue --me
          JOBID     USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) 
         937344 username     aip-acct    imagenet.sh  PD    2:00:00     1    8 gres/gpu:l     64G  (Priority) 

[...]
```

Refer to the ([squeue manual page](https://slurm.schedmd.com/squeue.html)) for a full list of options.


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

If all you want is an interactive session on a GPU node (without the batch job), just use `srun` (https://slurm.schedmd.com/srun.html)

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

After you see $USER@kn###, you can run your script interactively.


## SSH to sbatch job

Any job submitted with sbatch is by definition a *batch* (non-interactive) job. It will sit in the Slurm scheduler queue until the appropriate compute resources become available, then it will run in the background. If you want to attach an interactive session to a running job, you can also do this using `srun`:

```
srun --pty --overlap --jobid <job-id> -w <hostname> /bin/bash
```

To obtain the values for `<job-id>` and `<hostname>`, use the `squeue` command as described above. In the following example, I query a list of my jobs and then use the above command to get an interactive shell session:

```
username@klogin01:~/scratch/imagenet$ sbatch imagenet.sh
Submitted batch job 937373
username@klogin01:~/scratch/imagenet$ squeue --me
          JOBID     USER      ACCOUNT           NAME  ST  TIME_LEFT NODES CPUS TRES_PER_N MIN_MEM NODELIST (REASON) 
         937373 username     aip-acct    imagenet.sh   R    1:52:20     1    8 gres/gpu:l     64G kn060 (None) 
username@klogin01:~/scratch/imagenet$ srun --pty --overlap --jobid 937373 -w kn060 /bin/bash
username@kn060:~/scratch/imagenet$
```


## Accessing specific GPUs

The Killarney cluster has both NVIDIA L40S and H100 GPUs available. To request a specific GPU type, use the `--gres=gpu` flag, for example:

```
# Request 2x L40S gpus
username@klogin01:/scratch$ srun --gres=gpu:l40s:2 --mem=32G --time=5:00 --pty bash
srun: job 581665 queued and waiting for resources
srun: job 581665 has been allocated resources
username@kn131:/scratch$ exit
exit

# Request 1x H100 gpu
username@klogin01:/scratch$ srun --gres=gpu:h100:1 --mem=32G --time=5:00 --pty bash
srun: job 581667 queued and waiting for resources
srun: job 581667 has been allocated resources
username@kn178:/scratch$
```


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


# Time Limits

The Killarney cluster uses time limits to ensure fair access to resources. These are groups into tiers (otherwise known as partitions) based on time length and resource availability.

## Tiers

There are 5 basic time tiers for cluster resources. When submitting a job, you need to specify a time limit, using the `--time=D-HH:MM:SS` argument. For example, ask for 1 hour using `--time=1:00:00`.

Your job will get automatically assigned to the correct partition depending on your request.

To view the various tiers:

```
username@klogin01:~$ sinfo --summarize
PARTITION       AVAIL  TIMELIMIT   NODES(A/I/O/T) NODELIST
gpubase_h100_b1    up    3:00:00         9/0/1/10 kn[169-178]
gpubase_h100_b2    up   12:00:00          8/0/0/8 kn[169-176]
gpubase_h100_b3    up 1-00:00:00          6/0/0/6 kn[171-176]
gpubase_h100_b4    up 3-00:00:00          4/0/0/4 kn[169-170,172,178]
gpubase_h100_b5    up 7-00:00:00          1/0/1/2 kn[171,177]
gpubase_l40s_b1    up    3:00:00     140/26/2/168 kn[001-168]
gpubase_l40s_b2    up   12:00:00      125/0/1/126 kn[001-126]
gpubase_l40s_b3    up 1-00:00:00       56/26/2/84 kn[001-042,127-168]
gpubase_l40s_b4    up 3-00:00:00        42/0/0/42 kn[043-084]
gpubase_l40s_b5    up 7-00:00:00        17/0/0/17 kn[085-101]
```

## Automatic Restarts

All jobs in our Slurm cluster have a time limit, after which they will get stopped. For longer running jobs which need more than a few hours, the [Vaughan Slurm Changes](https://support.vectorinstitute.ai/Computing?action=AttachFile&do=view&target=Vector+Vaughan+HPC+Changes+FAQ+2023.pdf) document describes how to automatically restart these.

## Checkpoints

In order to avoid losing your work when your job exits, you will need to implement checkpoints - periodic snapshots of your work that you load from so you can stop and resume without much lost work.

On the legacy Bon Echo cluster, there was a dedicated checkpoint space in the file system for checkpoints. **⚠️ In Killarney, there is no dedicated checkpoint space.** Users are expected to manage their own checkpoints under their `$SCRATCH` folder.


# Useful Links and Resources

Computing parent page: https://support.vectorinstitute.ai/Computing

Vaughan valid partition/qos: https://support.vectorinstitute.ai/Vaughan_slurm_changes

Checkpointing: https://support.vectorinstitute.ai/CheckpointExample

Slurm Scheduler: https://support.vectorinstitute.ai/slurm_fairshare

FAQ: https://support.vectorinstitute.ai/FAQ%20about%20the%20cluster


# Support

For any cluster issues please email ops-help@vectorinstitute.ai.

For engineering/checkpointing related issues please check out the #computing channel on Slack.
