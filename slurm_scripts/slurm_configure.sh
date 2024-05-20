#!/bin/bash
#SBATCH --job-name=flexforecast_init
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu

# Setup conda env - ensure the .conda dir is located on my workir, and move it if not
[ -L ~/.conda ] && unlink ~/.conda
[ -d ~/.conda ] && mv -v ~/.conda $WORKDIR
[ ! -d $WORKDIR/.conda ] && mkdir $WORKDIR/.conda
ln -s $WORKDIR/.conda ~/.conda

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Create conda environment from the configuration file
conda env create -f slurm_scripts/config/environment_flexforecast.yml --force

# Activate the environment
source activate flexforecast

# Check whether the environment is activated
echo 'Check which environment is activated'
conda env list