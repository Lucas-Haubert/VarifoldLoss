#!/bin/bash
#SBATCH --job-name=SegRNN
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
model_name=SegRNN



# traffic dataset: strong seasonality

# SegRNN - traffic - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id SegRNN_traffic_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1

# SegRNN - traffic - DILATE_05
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id SegRNN_traffic_DILATE_05_96_96 \
  --model $model_name \
  --loss 'DILATE' \
  --train_epochs 20 \
  --patience 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1



# ECL: medium seasonality

# SegRNN - electricity - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id SegRNN_electricity_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1

# SegRNN - electricity - DILATE_05
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id SegRNN_electricity_DILATE_05_96_96 \
  --model $model_name \
  --loss 'DILATE' \
  --train_epochs 20 \
  --patience 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1



# exchange_rate: weak seasonality

# SegRNN - exchange_rate - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id SegRNN_exchange_rate_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --dropout 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1

# SegRNN - exchange_rate - DILATE_05
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id SegRNN_exchange_rate_DILATE_05_96_96 \
  --model $model_name \
  --loss 'DILATE' \
  --train_epochs 20 \
  --patience 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --dropout 0 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1