#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=v100-31:1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cabaak9609@ung.edu

cd /jet/home/baaklini/Machine-Learning-Project/Project 2
module load anaconda3
conda activate ml_project

python3 ann.py