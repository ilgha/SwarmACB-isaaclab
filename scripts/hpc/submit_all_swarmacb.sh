#!/bin/bash
set -euo pipefail

# Submit all five benchmark missions for all five SwarmACB methods.
#
# This launches 25 SLURM arrays. Each array has 10 independent runs.
#
# Usage:
#   cd /home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab
#   bash scripts/hpc/submit_all_swarmacb.sh

PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
SCRIPT_DIR="$PROJECT_DIR/scripts/hpc"

sbatch "$SCRIPT_DIR/train_dirgate_dandelion.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_daisy.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_lily.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_tulip.slurm"
sbatch "$SCRIPT_DIR/train_dirgate_cyclamen.slurm"

sbatch "$SCRIPT_DIR/train_xor_dandelion.slurm"
sbatch "$SCRIPT_DIR/train_xor_daisy.slurm"
sbatch "$SCRIPT_DIR/train_xor_lily.slurm"
sbatch "$SCRIPT_DIR/train_xor_tulip.slurm"
sbatch "$SCRIPT_DIR/train_xor_cyclamen.slurm"

sbatch "$SCRIPT_DIR/train_homing_dandelion.slurm"
sbatch "$SCRIPT_DIR/train_homing_daisy.slurm"
sbatch "$SCRIPT_DIR/train_homing_lily.slurm"
sbatch "$SCRIPT_DIR/train_homing_tulip.slurm"
sbatch "$SCRIPT_DIR/train_homing_cyclamen.slurm"

sbatch "$SCRIPT_DIR/train_foraging_dandelion.slurm"
sbatch "$SCRIPT_DIR/train_foraging_daisy.slurm"
sbatch "$SCRIPT_DIR/train_foraging_lily.slurm"
sbatch "$SCRIPT_DIR/train_foraging_tulip.slurm"
sbatch "$SCRIPT_DIR/train_foraging_cyclamen.slurm"

sbatch "$SCRIPT_DIR/train_sheltering_dandelion.slurm"
sbatch "$SCRIPT_DIR/train_sheltering_daisy.slurm"
sbatch "$SCRIPT_DIR/train_sheltering_lily.slurm"
sbatch "$SCRIPT_DIR/train_sheltering_tulip.slurm"
sbatch "$SCRIPT_DIR/train_sheltering_cyclamen.slurm"
