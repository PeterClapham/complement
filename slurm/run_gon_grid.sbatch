#!/bin/bash
#SBATCH --job-name=gon-grid
#SBATCH --array=0-48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module purge
# Adjust these lines for your cluster.
# module load python/3.11
# module load cuda

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Example environment setup. Replace with your cluster's preferred workflow.
source .venv/bin/activate

python -u scripts/run_slurm_task.py --config configs/slurm.yaml
