#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cnchap4107@ung.edu

cd /ocean/projects/cis250151p/cchapman/Project2
module load anaconda3
conda activate project2_mnist_ann

python3 ann.py