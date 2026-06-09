#!/bin/bash
set -euo pipefail

# Submit all Directional Gate SwarmACB variants as 10-run SLURM arrays.
#
# Usage:
#   cd /home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab
#   bash scripts/hpc/submit_all_dirgate.sh

PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
SCRIPT_DIR="$PROJECT_DIR/scripts/hpc"

sbatch "$SCRIPT_DIR/train_dirgate_dandelion.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_daisy.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_lily.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_tulip.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_cyclamen.slurm"
