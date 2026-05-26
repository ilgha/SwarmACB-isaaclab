#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:?Usage: run_training_common.sh CONFIG_PATH RUN_NAME}"
RUN_NAME="${2:?Usage: run_training_common.sh CONFIG_PATH RUN_NAME}"

PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
CONTAINER="${CONTAINER:-/srv/apps/shared/containers/isaacsim.sif}"
OVERLAY="${OVERLAY:-${GLOBALSCRATCH:?GLOBALSCRATCH must be set}/isaacsim_overlay.img}"
RUN_SUFFIX="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="runs/${RUN_NAME}_hpc_${RUN_SUFFIX}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}_hpc_${RUN_SUFFIX}"

mkdir -p "$PROJECT_DIR/logs"

echo "=========================================="
echo "  Job ID:       ${SLURM_JOB_ID:-local}"
echo "  Node:         $(hostname)"
echo "  GPUs:         ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Array task:   $RUN_SUFFIX"
echo "  Date:         $(date)"
echo "  Project:      $PROJECT_DIR"
echo "  Container:    $CONTAINER"
echo "  Config:       $CONFIG_PATH"
echo "  Run dir:      $RUN_DIR"
echo "  Checkpoints:  $CHECKPOINT_DIR"
echo "=========================================="

nvidia-smi || true

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
            --config $CONFIG_PATH \
            --headless \
            --log_dir $RUN_DIR \
            --checkpoint_dir $CHECKPOINT_DIR
    "

echo "Training finished at $(date)"
