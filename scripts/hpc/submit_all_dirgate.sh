#!/bin/bash
set -euo pipefail

# Submit all Directional Gate variants as 10-run SLURM arrays.
#
# Usage:
#   cd /home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab
#   bash scripts/hpc/submit_all_dirgate.sh

sbatch scripts/hpc/train_dandelion.slurm
sbatch scripts/hpc/train_daisy.slurm
sbatch scripts/hpc/train_lily.slurm
sbatch scripts/hpc/train_tulip.slurm
sbatch scripts/hpc/train_cyclamen.slurm
