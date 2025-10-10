#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=IXI_preprocessing
#SBATCH --output=logs/IXI_preprocessing_%J.log

module load cuda/11.8
module load fsl/6.0.7   # NOTE: to use the MNI152 template

source /gpfs/home/dl5635/.bashrc
conda activate ukbb_deep_pretrain
cd /gpfs/data/fieremanslab/dayne/SFCN

PYTHONPATH=$PYTHONPATH:$(pwd) python scripts/preprocess.py --input_dir data/IXI/T1/sampled_age_44_80
