#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=smri_preprocessing
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/SFCN

export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

module load cuda/11.8
module load fsl/6.0.7   # NOTE: to use the MNI152 template

source /gpfs/home/dl5635/.bashrc
conda activate ukbb_deep_pretrain
cd $PROJECT_ROOT
mkdir -p logs/${SLURM_JOB_NAME}

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Start Time: $(date)"
echo "=================================================="

START_TIME=$(date +%s)

# python scripts/preprocess_smri.py --input_dir data/IXI/raw
python scripts/preprocess_smri.py --input_dir data/OASIS_3/raw

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
DAYS=$((ELAPSED/86400))
HOURS=$(((ELAPSED%86400)/3600))
MINUTES=$(((ELAPSED%3600)/60))
SECONDS=$((ELAPSED%60))

echo "=================================================="
echo "Preprocessing Duration: ${DAYS}d ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End Time: $(date)"
echo "=================================================="
