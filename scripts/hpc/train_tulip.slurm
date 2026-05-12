#!/bin/bash
#SBATCH --job-name=swarm_tulip
#SBATCH --array=0-9
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=batch
#SBATCH --output=logs/%x_%A_%a_master.out
#SBATCH --error=logs/%x_%A_%a_master.err


# ═══════════════════════════════════════════════════════════════════════
#  SwarmACB — Train tulip variant on HPC via Apptainer + Isaac Lab
# ═══════════════════════════════════════════════════════════════════════
#
#  Prerequisites (run once, interactively):
#    apptainer overlay create --sparse --size 32768 $GLOBALSCRATCH/isaacsim_overlay.img
#
#  Submit:
#    cd /home/ulb/iridia_robo/igharbi/SwarmACB_isaac
#    sbatch scripts/hpc/train_tulip.slurm
#
#  This submits 10 independent runs via SLURM array tasks 0..9.
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab"
CONTAINER="/srv/apps/shared/containers/isaacsim.sif"
OVERLAY="$GLOBALSCRATCH/isaacsim_overlay.img"
RUN_SUFFIX="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="runs/DirGate_tulip_hpc_${RUN_SUFFIX}"
CHECKPOINT_DIR="checkpoints/DirGate_tulip_hpc_${RUN_SUFFIX}"

# ── Create log directory ──────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"

# ── Print job info ────────────────────────────────────────────────────
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

# ── Run training inside Apptainer ─────────────────────────────────────
# Key flags:
#   --nv           : expose NVIDIA GPU drivers into the container
#   --overlay :ro  : shared prepared overlay, mounted read-only for array jobs
#   --writable-tmpfs: per-job temporary writes inside the container
#   --bind         : mount the project directory so the code is visible inside
#   --env          : pass environment variables into the container
#   --headless     : Isaac Lab flag — no GUI, no rendering (essential for HPC)
#
# The training script will:
#   1. Bootstrap Isaac Lab via AppLauncher (headless)
#   2. Create the kinematic SwarmACB environment (no USD physics needed)
#   3. Train POCA for 120M steps

apptainer exec \
    --nv \
    --overlay "${OVERLAY}:ro" \
    --writable-tmpfs \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --env ACCEPT_EULA=Y \
    "$CONTAINER" \
    bash -c "
        source /root/isaac_env/bin/activate && \
        export LD_LIBRARY_PATH=$PROJECT_DIR/.syslibs/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH && \
        export PYTHONPATH=$PROJECT_DIR/source/SwarmACB_isaac:\$PYTHONPATH && \
        cd $PROJECT_DIR && \
        python scripts/train.py \
            --config configs/DirGate_tulip.yaml \
            --headless \
            --log_dir $RUN_DIR \
            --checkpoint_dir $CHECKPOINT_DIR
    "

echo "Training finished at $(date)"
