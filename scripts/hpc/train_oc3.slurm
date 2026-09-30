#!/bin/bash
#SBATCH --job-name=oc3_cyclamen
#SBATCH --array=0-9
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=batch
#SBATCH --output=logs/%x_%A_%a_master.out
#SBATCH --error=logs/%x_%A_%a_master.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
SCRIPT_DIR="$PROJECT_DIR/scripts/hpc"
HOLDOUT="${1:-}"
ARGS=()
RUN_NAME="OC3_cyclamen_v1"
if [[ -n "$HOLDOUT" ]]; then
    case "$HOLDOUT" in
        DirGate|XOR|Homing|Foraging|Sheltering) ;;
        *) echo "Unknown held-out mission: $HOLDOUT" >&2; exit 2 ;;
    esac
    RUN_NAME="${RUN_NAME}_without_${HOLDOUT}"
    ARGS+=(--exclude "$HOLDOUT")
fi
bash "$SCRIPT_DIR/run_training_common.sh" "configs/OC3_cyclamen.yaml" \
    "$RUN_NAME" "scripts/train_oc3.py" "${ARGS[@]}"
