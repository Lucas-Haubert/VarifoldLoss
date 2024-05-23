#!/bin/bash
#SBATCH --job-name=dilate_handling
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast

# Try DILATE loss on synthetic sequences
python loss/dilate/dilate_loss.py