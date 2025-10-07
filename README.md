# Vector Institute Compute Playbook

A comprehensive starter repository for researchers at the Vector Institute to get started with high-performance computing on **Bon Echo** and **Killarney** clusters. This playbook provides everything you need to run machine learning experiments at scale, from basic cluster usage to advanced distributed training workflows.

## 🚀 What's Inside

This repository provides two main components:

### 📚 **Getting Started Documentation**
- **Cluster Introduction**: Complete guide to connecting to and using Vector compute resources
- **Slurm Examples**: Real-world examples showing how to submit jobs, run distributed training, and use cluster services
- **Migration Guide**: Instructions for moving from legacy Bon Echo to the new Killarney cluster

### 🧪 **ML Training Templates**
- **Ready-to-run examples** for different ML domains (LLM, VLM, MLP, RL)
- **Hydra + Submitit integration** for configurable experiments and hyperparameter sweeps
- **Cluster-optimized configs** for different hardware setups (A40, A100, H100, L40S)
- **Checkpointing & requeue** support for long-running jobs

## 🏃‍♂️ Quick Start

### 1. Prerequisites
- Access to Vector Institute compute clusters (Bon Echo or Killarney)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager installed

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/VectorInstitute/vec-playbook.git
cd vec-playbook

# Install dependencies
uv sync
```

### 3. Configure Your Account
Edit `templates/configs/user.yaml` with your Slurm account details:
```yaml
user:
  slurm:
    account: YOUR_ACCOUNT
```

### 4. Run Your First Job
```bash
# Simple MLP training on Killarney L40S
uv run python -m mlp.single.launch compute=killarney/l40s_1x requeue=off --multirun
```

## 📖 Navigation Guide

### For New Users
1. **Start here**: [Getting Started Documentation](./getting-started/) - Learn the basics of Vector compute
2. **Try examples**: [Slurm Examples](./getting-started/slurm-examples/) - Run simple jobs to get familiar
3. **Use templates**: [Templates](./templates/) - Run ML training experiments

### For Experienced Users
- **Templates**: [templates/](./templates/) - Training workflows
- **Configs**: [templates/configs/](./templates/configs/) - Cluster and experiment configurations
- **Advanced**: [templates/README.md](./templates/README.md) - Detailed usage instructions

## 🖥️ Supported Hardware

### Bon Echo Cluster
- **A40 GPUs**: 1x, 4x configurations
- **A100 GPUs**: 1x, 4x configurations  

### Killarney Cluster
- **H100 GPUs**: 1x, 8x configurations
- **L40S GPUs**: 1x, 2x configurations

## 📚 Documentation Structure

```
vec-playbook/
├── getting-started/           # 📖 Learning resources
│   ├── introduction-to-vector-compute/  # Cluster basics
│   └── slurm-examples/        # 🧪 Hands-on examples
├── templates/                # 🧬 ML training templates
│   ├── src/                  # Template source code
│   └── configs/              # Cluster & experiment configs
└── README.md                 # This file
```



## 🤝 Contributing

We welcome contributions! Whether it's:
- New training templates
- Additional cluster configurations  
- Documentation improvements
- Bug fixes
