#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cabaak9609@ung.edu

cd /ocean/projects/cis250151p/baaklini/.conda

module load anaconda3
conda activate ml_project

python3 ann.py