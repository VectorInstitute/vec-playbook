#!/bin/bash
source ~/.bashrc
source /opt/lmod/lmod/init/bash
export MODULEPATH=/opt/modulefiles:/pkgs/modulefiles:/pkgs/environment-modules


export NSS_WRAPPER_PASSWD=$HOME/.local/overlay/etc/passwd
export NSS_WRAPPER_GROUP=$HOME/.local/overlay/etc/group
export LD_PRELOAD="/h/jacobtian/.local/lib/libnss_wrapper.so"

whoami
id -u

module load singularity-ce
# # Ensure a minimal NSS config so passwd/group resolution works via files
# mkdir -p "$HOME/.local/overlay/etc"
# cat > "$HOME/.local/overlay/nsswitch.conf" <<'EOF'
# passwd:         files
# group:          files
# shadow:         files
# gshadow:        files
# hosts:          files dns
# networks:       files
# protocols:      files
# services:       files
# ethers:         files
# rpc:            files
# netgroup:       files
# EOF

# # Add passwd + group entries for your current UID/GID
# OVERLAY="$HOME/.local/overlay"
# uid=$(id -u)
# gid=$(id -g)
# user=${USER:-u$uid}
# grp=$(id -gn 2>/dev/null || echo "g$gid")
# home=${HOME:-/home/$user}
# shell=${SHELL:-/bin/sh}

# # Minimal files (enough for whoami and basic tools)
# cat > "$OVERLAY/etc/passwd" <<EOF
# $user:x:$uid:$gid:$user:$home:$shell
# EOF

# cat /etc/passwd >> $OVERLAY/etc/passwd

# getent group $(id -G) > "$OVERLAY/etc/group"
# cat /etc/group >> $OVERLAY/etc/group

# # Export env to be visible inside the container
# # - Ensure Slurm finds site config and binaries
# # - Ensure dynamic linker finds site libs
# # - Point libmunge to the correct host socket path
export SINGULARITYENV_SLURM_CONF=/opt/slurm/etc/slurm.conf
export SINGULARITYENV_PATH="/opt/slurm/bin:$PATH"
export SINGULARITYENV_LD_LIBRARY_PATH="/opt/slurm/lib:/opt/slurm/lib64:/opt/munge/lib:/opt/munge/lib64:${LD_LIBRARY_PATH:-}"

# # Detect the host's munge socket path and pass it through
# if [ -S /opt/munge/var/run/munge/munge.socket.2 ]; then
#   export SINGULARITYENV_MUNGE_SOCKET=/opt/munge/var/run/munge/munge.socket.2
# elif [ -S /run/munge/munge.socket.2 ]; then
#   export SINGULARITYENV_MUNGE_SOCKET=/run/munge/munge.socket.2
# elif [ -S /var/run/munge/munge.socket.2 ]; then
#   export SINGULARITYENV_MUNGE_SOCKET=/var/run/munge/munge.socket.2
# fi
singularity exec \
--nv \
--bind /model-weights:/model-weights \
--bind /projects/llm:/projects/llm \
--bind /scratch/ssd004/scratch/jacobtian:/scratch/ssd004/scratch/ \
--bind /scratch/ssd004/scratch/jacobtian:/scratch/ \
--bind /opt/:/opt/ \
--bind /usr/bin/lua5.2:/usr/bin/lua5.2 \
--bind /opt/slurm/:/opt/slurm \
--bind /opt/munge/var/run/munge:/opt/munge/var/run/munge \
--bind "$HOME/.local/overlay/nsswitch.conf:/etc/nsswitch.conf" \
--bind $OVERLAY/etc/passwd:/etc/passwd \
--bind $OVERLAY/etc/group:/etc/group \
--bind /var/run/munge:/var/run/munge \
--bind /etc/munge:/etc/munge \
--bind $HOME:$HOME \
--bind $SCRATCH:$SCRATCH \
/projects/llm/unsloth-vllm-trl-latest.sif \
bash run_in_venv.sh $@