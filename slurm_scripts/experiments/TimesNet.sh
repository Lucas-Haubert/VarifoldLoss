#!/bin/bash
#SBATCH --job-name=TimesNet
#SBATCH --output=slurm_outputs/%x.job_%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpup100

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast




# Ratrapper les valeurs des métriques pour iTransformer sur MSE

# Choose the model
model_name=iTransformer

# iTransformer - traffic - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id iTransformer_traffic_VARIFOLD_7th_try_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

# # iTransformer - electricity - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id iTransformer_electricity_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.0005 \
#   --itr 1

# # # iTransformer - exchange_rate - MSE
# # python -u run.py \
# #   --is_training 1 \
# #   --root_path ./dataset/exchange_rate/ \
# #   --data_path exchange_rate.csv \
# #   --model_id iTransformer_exchange_rate_MSE_96_96 \
# #   --model $model_name \
# #   --loss 'MSE' \
# #   --train_epochs 10 \
# #   --patience 5 \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 96 \
# #   --e_layers 2 \
# #   --enc_in 8 \
# #   --dec_in 8 \
# #   --c_out 8 \
# #   --des 'Exp' \
# #   --d_model 128 \
# #   --d_ff 128 \
# #   --itr 1



# # Choose the model
# model_name=TimesNet

# # traffic dataset: strong seasonality

# # TimesNet - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TimesNet_traffic_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --top_k 5 \
#   --itr 1

# # TimesNet - traffic - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TimesNet_traffic_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --factor 3 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --top_k 5 \
#   --itr 1



# # ECL: medium seasonality

# # TimesNet - electricity - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TimesNet_electricity_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --factor 3 \
#   --e_layers 2 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 512 \
#   --top_k 5 \
#   --itr 1

# # TimesNet - electricity - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TimesNet_electricity_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --factor 3 \
#   --e_layers 2 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 512 \
#   --top_k 5 \
#   --itr 1



# # # exchange_rate: weak seasonality

# # # TimesNet - exchange_rate - MSE
# # python -u run.py \
# #   --is_training 1 \
# #   --root_path ./dataset/exchange_rate/ \
# #   --data_path exchange_rate.csv \
# #   --model_id TimesNet_exchange_rate_MSE_96_96 \
# #   --model $model_name \
# #   --loss 'MSE' \
# #   --train_epochs 10 \
# #   --patience 5 \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 96 \
# #   --factor 3 \
# #   --e_layers 2 \
# #   --enc_in 8 \
# #   --dec_in 8 \
# #   --c_out 8 \
# #   --des 'Exp' \
# #   --d_model 64 \
# #   --d_ff 64 \
# #   --top_k 5 \
# #   --itr 1

# # # TimesNet - exchange_rate - DILATE_05
# # python -u run.py \
# #   --is_training 1 \
# #   --root_path ./dataset/exchange_rate/ \
# #   --data_path exchange_rate.csv \
# #   --model_id TimesNet_exchange_rate_DILATE_05_96_96 \
# #   --model $model_name \
# #   --loss 'DILATE' \
# #   --train_epochs 20 \
# #   --patience 10 \
# #   --data custom \
# #   --features M \
# #   --seq_len 96 \
# #   --pred_len 96 \
# #   --factor 3 \
# #   --e_layers 2 \
# #   --enc_in 8 \
# #   --dec_in 8 \
# #   --c_out 8 \
# #   --des 'Exp' \
# #   --d_model 64 \
# #   --d_ff 64 \
# #   --top_k 5 \
# #   --itr 1