#!/bin/bash
#SBATCH --job-name=baby_exper
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu_test

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast

# Choose the model
# model_name=Informer

# # Run the experiment
# python -u run.py \
#   --is_training 1 \
#   --root_path ./data/ETT-small/ \
#   --data_path ETTh1_reduced.csv \
#   --model_id Informer_ETTh1_192_48 \
#   --model $model_name \
#   --loss 'MSE' \
#   --data custom \
#   --features MS \
#   --seq_len 192 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1

# Remove the previous results
# rm -r checkpoints/*
# rm -r plots_epochs_losses/*
# rm -r results/*
# rm -r test_results/*
