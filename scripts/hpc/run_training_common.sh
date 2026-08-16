#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:?Usage: run_training_common.sh CONFIG_PATH RUN_NAME}"
RUN_NAME="${2:?Usage: run_training_common.sh CONFIG_PATH RUN_NAME}"

PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
CONTAINER="${CONTAINER:-/srv/apps/shared/containers/isaacsim.sif}"
RUN_SUFFIX="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
RUN_DIR="runs/${RUN_NAME}_hpc_${RUN_SUFFIX}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}_hpc_${RUN_SUFFIX}"

# The project and compatibility libraries are bind-mounted from the host, so
# the old ext3 overlay is unnecessary. It can still be enabled explicitly for
# a custom cluster image, but keeping it off avoids fuse2fs mount timeouts.
HPC_USE_OVERLAY="${HPC_USE_OVERLAY:-0}"
APPTAINER_LAUNCH_ATTEMPTS="${APPTAINER_LAUNCH_ATTEMPTS:-5}"
APPTAINER_RETRY_DELAY="${APPTAINER_RETRY_DELAY:-15}"

if ! [[ "$APPTAINER_LAUNCH_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "APPTAINER_LAUNCH_ATTEMPTS must be a positive integer" >&2
    exit 2
fi
if ! [[ "$APPTAINER_RETRY_DELAY" =~ ^[0-9]+$ ]]; then
    echo "APPTAINER_RETRY_DELAY must be a non-negative integer" >&2
    exit 2
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory does not exist: $PROJECT_DIR" >&2
    exit 2
fi
if [[ ! -f "$PROJECT_DIR/$CONFIG_PATH" ]]; then
    echo "Training config does not exist: $PROJECT_DIR/$CONFIG_PATH" >&2
    exit 2
fi
if [[ ! -f "$CONTAINER" ]]; then
    echo "Container does not exist: $CONTAINER" >&2
    exit 2
fi

mkdir -p "$PROJECT_DIR/logs"

START_MARKER="$PROJECT_DIR/logs/.container_started_${ARRAY_JOB_ID}_${RUN_SUFFIX}_$$"
rm -f "$START_MARKER"
trap 'rm -f "$START_MARKER"' EXIT

APPTAINER_ARGS=(
    exec
    --nv
    --writable-tmpfs
    --bind "$PROJECT_DIR:$PROJECT_DIR"
    --env ACCEPT_EULA=Y
    --env "HPC_START_MARKER=$START_MARKER"
    --env "SWARM_PROJECT_DIR=$PROJECT_DIR"
    --env "SWARM_CONFIG_PATH=$CONFIG_PATH"
    --env "SWARM_RUN_DIR=$RUN_DIR"
    --env "SWARM_CHECKPOINT_DIR=$CHECKPOINT_DIR"
    --env "SWARM_SEED=$RUN_SUFFIX"
)

OVERLAY_DESCRIPTION="disabled"
case "$HPC_USE_OVERLAY" in
    0|false|FALSE|no|NO)
        ;;
    1|true|TRUE|yes|YES)
        OVERLAY="${OVERLAY:-${GLOBALSCRATCH:?GLOBALSCRATCH must be set when HPC_USE_OVERLAY=1}/isaacsim_overlay.img}"
        if [[ ! -f "$OVERLAY" ]]; then
            echo "Requested overlay does not exist: $OVERLAY" >&2
            exit 2
        fi
        APPTAINER_ARGS+=(--overlay "${OVERLAY}:ro")
        OVERLAY_DESCRIPTION="$OVERLAY (read-only)"
        ;;
    *)
        echo "HPC_USE_OVERLAY must be 0/1, false/true, or no/yes" >&2
        exit 2
        ;;
esac

echo "=========================================="
echo "  Job ID:       ${SLURM_JOB_ID:-local}"
echo "  Array job:    $ARRAY_JOB_ID"
echo "  Node:         $(hostname)"
echo "  GPUs:         ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  Array task:   $RUN_SUFFIX"
echo "  Seed:         $RUN_SUFFIX"
echo "  Date:         $(date)"
echo "  Project:      $PROJECT_DIR"
echo "  Container:    $CONTAINER"
echo "  Overlay:      $OVERLAY_DESCRIPTION"
echo "  Config:       $CONFIG_PATH"
echo "  Run dir:      $RUN_DIR"
echo "  Checkpoints:  $CHECKPOINT_DIR"
echo "=========================================="

nvidia-smi || true

CONTAINER_COMMAND='
set -euo pipefail
touch "$HPC_START_MARKER"
source /root/isaac_env/bin/activate
export LD_LIBRARY_PATH="$SWARM_PROJECT_DIR/.syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SWARM_PROJECT_DIR/source/SwarmACB_isaac:${PYTHONPATH:-}"
cd "$SWARM_PROJECT_DIR"
exec python -u scripts/train.py \
    --config "$SWARM_CONFIG_PATH" \
    --headless \
    --seed "$SWARM_SEED" \
    --log_dir "$SWARM_RUN_DIR" \
    --checkpoint_dir "$SWARM_CHECKPOINT_DIR"
'

status=1
for ((attempt = 1; attempt <= APPTAINER_LAUNCH_ATTEMPTS; attempt++)); do
    rm -f "$START_MARKER"
    echo "[HPC] Starting Apptainer (attempt $attempt/$APPTAINER_LAUNCH_ATTEMPTS)..."

    set +e
    apptainer "${APPTAINER_ARGS[@]}" "$CONTAINER" bash -lc "$CONTAINER_COMMAND"
    status=$?
    set -e

    if [[ $status -eq 0 ]]; then
        break
    fi

    if [[ -f "$START_MARKER" ]]; then
        echo "[HPC] Training exited with status $status after container startup; not retrying." >&2
        exit "$status"
    fi

    if [[ $attempt -eq $APPTAINER_LAUNCH_ATTEMPTS ]]; then
        echo "[HPC] Apptainer failed before container startup after $attempt attempts." >&2
        exit "$status"
    fi

    delay=$((APPTAINER_RETRY_DELAY * attempt))
    echo "[HPC] Container did not start (status $status); retrying in ${delay}s..." >&2
    sleep "$delay"
done

echo "Training finished at $(date)"
