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
echo "=== Installing missing system libraries (no root needed) ==="
# Isaac Sim plugins require libX11, libgomp, libGLU, libXt even in headless mode.
# These are missing from the base container. Since we can't apt-get install
# (requires root), we download the .deb packages and extract them manually.
LIBDIR="$PROJECT_DIR/.syslibs"

apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    bash -c "
        LIBDIR=$LIBDIR
        LIBPATH=\$LIBDIR/usr/lib/x86_64-linux-gnu

        # Skip if already extracted
        if [ -f \"\$LIBPATH/libX11.so.6\" ] && [ -f \"\$LIBPATH/libgomp.so.1\" ]; then
            echo 'System libraries already extracted — skipping.'
            exit 0
        fi

        echo 'Downloading and extracting .deb packages...'
        mkdir -p \$LIBDIR /tmp/_debs
        cd /tmp/_debs

        # Download all needed debs (and their dependencies) — no root required
        apt-get download \
            libx11-6 libgomp1 libglu1-mesa libxt6 libxrender1 libxext6 \
            libxau6 libxcb1 libxdmcp6 libbsd0 libmd0 \
            libglvnd0 libopengl0 libice6 libsm6 2>/dev/null

        # Extract each into our custom syslibs prefix — no root required
        for deb in *.deb; do
            dpkg-deb -x \"\$deb\" \$LIBDIR
        done

        rm -rf /tmp/_debs

        # Verify key libraries are present
        for lib in libX11.so.6 libgomp.so.1 libGLU.so.1 libXt.so.6 libXrender.so.1; do
            if [ -f \"\$LIBPATH/\$lib\" ]; then
                echo \"  OK: \$lib\"
            else
                echo \"  MISSING: \$lib\"
            fi
        done
        echo 'Library extraction complete.'
    "

echo ""
echo "=== Verifying SwarmACB_isaac import (via PYTHONPATH) ==="
apptainer exec \
    --nv \
    --overlay "$OVERLAY" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    bash -c "
        source /root/isaac_env/bin/activate && \
        export LD_LIBRARY_PATH=$PROJECT_DIR/.syslibs/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH && \
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
        export LD_LIBRARY_PATH=$PROJECT_DIR/.syslibs/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH && \
        export PYTHONPATH=$PROJECT_DIR/source:\$PYTHONPATH && \
        python $PROJECT_DIR/scripts/hpc/check_env.py
    "

echo ""
echo "=== Setup complete ==="
echo "You can now submit training jobs with:"
echo "  sbatch scripts/hpc/train_tulip.slurm"
