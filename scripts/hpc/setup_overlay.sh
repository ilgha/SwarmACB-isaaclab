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

PROJECT_DIR="/home/ulb/iridia_robo/igharbi/SwarmACB_isaac"
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
echo "=== Installing SwarmACB_isaac package into overlay ==="
apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    bash -c "
        cd $PROJECT_DIR/source/SwarmACB_isaac && \
        python3.11 -m pip install --no-deps -e . && \
        echo '' && \
        echo '=== Verifying installation ===' && \
        python3.11 -c 'import SwarmACB_isaac; print(\"SwarmACB_isaac imported OK from:\", SwarmACB_isaac.__file__)'
    "

echo ""
echo "=== Quick headless test ==="
apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    python3.11 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('All good!')
"

echo ""
echo "=== Setup complete ==="
echo "You can now submit training jobs with:"
echo "  sbatch scripts/hpc/train_tulip.slurm"
