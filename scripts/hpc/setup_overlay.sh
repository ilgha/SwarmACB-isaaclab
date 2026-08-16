#!/bin/bash
# Legacy filename retained for compatibility. The runtime no longer needs a
# persistent ext3 overlay: project code and compatibility libraries are bound
# directly from the repository.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
CONTAINER="${CONTAINER:-/srv/apps/shared/containers/isaacsim.sif}"
LIBDIR="$PROJECT_DIR/.syslibs"

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory does not exist: $PROJECT_DIR" >&2
    exit 2
fi
if [[ ! -f "$CONTAINER" ]]; then
    echo "Container does not exist: $CONTAINER" >&2
    exit 2
fi

echo "=== Preparing HPC compatibility libraries ==="
echo "Persistent overlay: disabled"

apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --env "SWARM_PROJECT_DIR=$PROJECT_DIR" \
    --env "SWARM_LIBDIR=$LIBDIR" \
    "$CONTAINER" \
    bash -lc '
        set -euo pipefail
        libpath="$SWARM_LIBDIR/usr/lib/x86_64-linux-gnu"

        if [[ -f "$libpath/libX11.so.6" && -f "$libpath/libgomp.so.1" ]]; then
            echo "System libraries already extracted; skipping download."
        else
            echo "Downloading and extracting compatibility packages..."
            mkdir -p "$SWARM_LIBDIR" /tmp/swarmacb_debs
            cd /tmp/swarmacb_debs
            apt-get download \
                libx11-6 libgomp1 libglu1-mesa libxt6 libxrender1 libxext6 \
                libxau6 libxcb1 libxdmcp6 libbsd0 libmd0 \
                libglvnd0 libopengl0 libice6 libsm6 2>/dev/null

            for deb in ./*.deb; do
                dpkg-deb -x "$deb" "$SWARM_LIBDIR"
            done
            rm -rf /tmp/swarmacb_debs
        fi

        for library in libX11.so.6 libgomp.so.1 libGLU.so.1 libXt.so.6 libXrender.so.1; do
            if [[ ! -f "$libpath/$library" ]]; then
                echo "Missing required compatibility library: $library" >&2
                exit 1
            fi
            echo "  OK: $library"
        done

        source /root/isaac_env/bin/activate
        export LD_LIBRARY_PATH="$libpath:${LD_LIBRARY_PATH:-}"
        export PYTHONPATH="$SWARM_PROJECT_DIR/source/SwarmACB_isaac:${PYTHONPATH:-}"
        python "$SWARM_PROJECT_DIR/scripts/hpc/check_env.py"
    '

echo "=== Setup complete ==="
echo "Training launchers now run without the legacy ext3 overlay."
