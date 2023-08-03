#!/bin/bash
#SBATCH -J gnn-model_800k
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p matador
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=amankel@ttu.edu

module load gcc python 
. $HOME/conda/etc/profile.d/conda.sh
conda activate jupyternotebook
#python preperation_photon.py
#python adam_train.py
python pred_photon.py
python pred_pion.py
#comparison_plots_pp.py
#comparison_plots_pion_800k.py
