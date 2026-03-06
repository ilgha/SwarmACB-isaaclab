#!/bin/bash
#SBATCH --job-name=swarm_tulip
#SBATCH --array=0-1
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
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab"
CONTAINER="/srv/apps/shared/containers/isaacsim.sif"
OVERLAY="$GLOBALSCRATCH/isaacsim_overlay.img"

# ── Create log directory ──────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/logs"

# ── Print job info ────────────────────────────────────────────────────
echo "=========================================="
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
echo "  Date:         $(date)"
echo "  Project:      $PROJECT_DIR"
echo "  Container:    $CONTAINER"
echo "=========================================="

nvidia-smi || true

# ── Run training inside Apptainer ─────────────────────────────────────
# Key flags:
#   --nv           : expose NVIDIA GPU drivers into the container
#   --overlay      : writable filesystem layer (Isaac Sim needs to create files)
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
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --env ACCEPT_EULA=Y \
    "$CONTAINER" \
    bash -c "
        cd $PROJECT_DIR && \
        python3.11 scripts/train.py \
            --config configs/DirGate_tulip.yaml \
            --headless \
            --log_dir runs/DirGate_tulip_hpc \
            --checkpoint_dir checkpoints/DirGate_tulip_hpc
    "

echo "Training finished at $(date)"
