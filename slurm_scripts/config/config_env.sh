#!/bin/bash
#SBATCH --job-name=config_env
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu_test


module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

conda create --name flexforecast -y
source activate flexforecast

pip install -r requirements.txt

conda env export > slurm_scripts/config/environment_flexforecast.yml