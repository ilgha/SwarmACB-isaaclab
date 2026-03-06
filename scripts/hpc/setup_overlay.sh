#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  One-time setup: install SwarmACB_isaac into the Apptainer overlay
# ═══════════════════════════════════════════════════════════════════════
#
#  Run this ONCE (interactively or via srun) before submitting training jobs.
#  It pip-installs the SwarmACB_isaac package into the writable overlay
#  so that 'import SwarmACB_isaac.tasks' works inside the container.
#
#  Usage:
#    srun --gres=gpu:1 --mem=16G --cpus-per-task=4 --time=00:30:00 \
#         bash scripts/hpc/setup_overlay.sh
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab"
CONTAINER="/srv/apps/shared/containers/isaacsim.sif"
OVERLAY="$GLOBALSCRATCH/isaacsim_overlay.img"

echo "=== Creating overlay (if not exists) ==="
if [ ! -f "$OVERLAY" ]; then
    apptainer overlay create --sparse --size 32768 "$OVERLAY"
    echo "Created $OVERLAY"
else
    echo "Overlay already exists at $OVERLAY"
fi

echo ""
echo "=== Installing missing system libraries into overlay ==="
# Isaac Sim plugins require libX11, libgomp, libGLU, libXt even in headless mode.
# These are missing from the base container — install them into the writable overlay.
apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    "$CONTAINER" \
    bash -c '
        # Check if libgomp is already installed
        if ! ldconfig -p 2>/dev/null | grep -q libgomp; then
            echo "Installing missing system libraries..."
            apt-get update -qq && \
            apt-get install -y --no-install-recommends \
                libx11-6 libgomp1 libglu1-mesa libxt6 libxrender1 libxext6 && \
            ldconfig && \
            echo "System libraries installed successfully."
        else
            echo "System libraries already installed (found libgomp)."
        fi
    '

echo ""
echo "=== Verifying SwarmACB_isaac import (via PYTHONPATH) ==="
apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    bash -c "
        source /root/isaac_env/bin/activate && \
        export PYTHONPATH=$PROJECT_DIR/source:\$PYTHONPATH && \
        python -c '
import SwarmACB_isaac
print(\"SwarmACB_isaac imported OK from:\", SwarmACB_isaac.__file__)
'
    "

echo ""
echo "=== Quick headless test ==="
apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    bash -c "
        source /root/isaac_env/bin/activate && \
        export PYTHONPATH=$PROJECT_DIR/source:\$PYTHONPATH && \
        python $PROJECT_DIR/scripts/hpc/check_env.py
    "

echo ""
echo "=== Setup complete ==="
echo "You can now submit training jobs with:"
echo "  sbatch scripts/hpc/train_tulip.slurm"
