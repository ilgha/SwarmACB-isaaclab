#!/bin/bash
#SBATCH --job-name=oc_homing
#SBATCH --array=0-9
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=batch
#SBATCH --output=logs/%x_%A_%a_master.out
#SBATCH --error=logs/%x_%A_%a_master.err

set -euo pipefail

# Train 10 independent fixed-option Option-Critic runs on Homing.
#
# Submit:
#   cd /home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab
#   sbatch scripts/hpc/train_oc_homing.slurm

PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
SCRIPT_DIR="$PROJECT_DIR/scripts/hpc"
bash "$SCRIPT_DIR/run_training_common.sh" "configs/OC_Homing_cyclamen.yaml" "OC_Homing_cyclamen"
