#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --job-name=arc_07_05
#SBATCH --error=results/arc_07_05/%x.err
#SBATCH --output=results/arc_07_05/%x.out
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

module load mamba
source activate arc

RESULT_DIR="results/arc_07_05"
mkdir -p "$RESULT_DIR"   

echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python training.py > "$RESULT_DIR/results.out" 2> "$RESULT_DIR/errors.err"
