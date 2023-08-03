#!/bin/bash
#SBATCH -J gnn-model_new
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p matador
#SBATCH -N 2
#SBATCH --ntasks-per-node=40
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=amankel@ttu.edu

module load gcc python 
. $HOME/conda/etc/profile.d/conda.sh
conda activate jupyternotebook
#python preperation_normal.py
python normal_train.py
