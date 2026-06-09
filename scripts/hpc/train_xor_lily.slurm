#!/bin/bash
#SBATCH --job-name=xor_lily
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
PROJECT_DIR="${PROJECT_DIR:-/home/ulb/iridia_robo/igharbi/SwarmACB-isaaclab}"
SCRIPT_DIR="$PROJECT_DIR/scripts/hpc"
bash "$SCRIPT_DIR/run_training_common.sh" "configs/XOR_lily.yaml" "XOR_lily"
