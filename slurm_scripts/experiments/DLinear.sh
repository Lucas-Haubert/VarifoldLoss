#!/bin/bash
#SBATCH --job-name=DLinear
#SBATCH --output=slurm_outputs/%x.job_%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpua100

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast

# Choose the model
model_name=DLinear



# traffic dataset: strong seasonality

# DLinear - traffic - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id DLinear_traffic_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

# DLinear - traffic - DILATE_05
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id DLinear_traffic_DILATE_05_96_96 \
  --model $model_name \
  --loss 'DILATE' \
  --train_epochs 20 \
  --patience 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --factor 3 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1



# ECL: medium seasonality

# DLinear - electricity - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id DLinear_electricity_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --factor 3 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1

# DLinear - electricity - DILATE_05
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id DLinear_electricity_DILATE_05_96_96 \
  --model $model_name \
  --loss 'DILATE' \
  --train_epochs 20 \
  --patience 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --factor 3 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1



# exchange_rate: weak seasonality

# DLinear - exchange_rate - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id DLinear_exchange_rate_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --factor 3 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1

# DLinear - exchange_rate - DILATE_05
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id DLinear_exchange_rate_DILATE_05_96_96 \
  --model $model_name \
  --loss 'DILATE' \
  --train_epochs 20 \
  --patience 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --factor 3 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1