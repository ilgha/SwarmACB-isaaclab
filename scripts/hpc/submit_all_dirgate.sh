#!/bin/bash
set -euo pipefail

# Submit all Directional Gate SwarmACB variants as 10-run SLURM arrays.
#
# Usage:
#   cd /home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab
#   bash scripts/hpc/submit_all_dirgate.sh

sbatch scripts/hpc/train_dirgate_dandelion.slurm
sbatch scripts/hpc/train_dirgate_daisy.slurm
sbatch scripts/hpc/train_dirgate_lily.slurm
sbatch scripts/hpc/train_dirgate_tulip.slurm
sbatch scripts/hpc/train_dirgate_cyclamen.slurm
