#!/bin/bash
#SBATCH --job-name=swarm_cyclamen
#SBATCH --array=0-9
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=batch
#SBATCH --output=logs/%x_%A_%a_master.out
#SBATCH --error=logs/%x_%A_%a_master.err

set -euo pipefail

# Train 10 independent cyclamen runs on HPC via Apptainer + Isaac Lab.
#
# Submit:
#   cd /home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab
#   sbatch scripts/hpc/train_cyclamen.slurm

PROJECT_DIR="/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab"
CONTAINER="/srv/apps/shared/containers/isaacsim.sif"
OVERLAY="$GLOBALSCRATCH/isaacsim_overlay.img"
RUN_SUFFIX="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="runs/DirGate_cyclamen_hpc_${RUN_SUFFIX}"
CHECKPOINT_DIR="checkpoints/DirGate_cyclamen_hpc_${RUN_SUFFIX}"

mkdir -p "$PROJECT_DIR/logs"

echo "=========================================="
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
echo "  Array task:   $RUN_SUFFIX"
echo "  Date:         $(date)"
echo "  Project:      $PROJECT_DIR"
echo "  Container:    $CONTAINER"
echo "  Run dir:      $RUN_DIR"
echo "  Checkpoints:  $CHECKPOINT_DIR"
echo "=========================================="

nvidia-smi || true

apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --env ACCEPT_EULA=Y \
    "$CONTAINER" \
    bash -c "
        source /root/isaac_env/bin/activate && \
        export LD_LIBRARY_PATH=$PROJECT_DIR/.syslibs/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH && \
        export PYTHONPATH=$PROJECT_DIR/source/SwarmACB_isaac:\$PYTHONPATH && \
        cd $PROJECT_DIR && \
        python scripts/train.py \
            --config configs/DirGate_cyclamen.yaml \
            --headless \
            --log_dir $RUN_DIR \
            --checkpoint_dir $CHECKPOINT_DIR
    "

echo "Training finished at $(date)"
