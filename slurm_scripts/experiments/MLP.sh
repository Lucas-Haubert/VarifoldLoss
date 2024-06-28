#!/bin/bash
#SBATCH --job-name=MLP_and_DLinear_MSE_and_VARIFOLD
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


# Choose the model
model_name=MLP

# # MLP - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id MLP_traffic_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 4 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1

# MLP - traffic - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id MLP_traffic_VARIFOLD_1_1_05_05_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 14.7 \
  --sigma_s_2 14.7 \
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



# MLP - ETTh1 - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id MLP_ETTh1_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

# MLP - ETTh1 - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id MLP_ETTh1_VARIFOLD_1_1_05_05_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 1.3 \
  --sigma_s_2 1.3 \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1



# MLP - exchange_rate - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id MLP_exchange_rate_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

# MLP - exchange_rate - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id MLP_exchange_rate_VARIFOLD_1_1_05_05_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 1.4 \
  --sigma_s_2 1.4 \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1






# Choose the model
model_name=DLinear

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

# DLinear - traffic - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id DLinear_traffic_VARIFOLD_1_1_05_05_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 14.7 \
  --sigma_s_2 14.7 \
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



# DLinear - ETTh1 - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id DLinear_ETTh1_MSE_96_96 \
  --model $model_name \
  --loss 'MSE' \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

# DLinear - ETTh1 - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id DLinear_ETTh1_VARIFOLD_1_1_05_05_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 1.3 \
  --sigma_s_2 1.3 \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1



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
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

# DLinear - exchange_rate - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id DLinear_exchange_rate_VARIFOLD_1_1_05_05_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 1.4 \
  --sigma_s_2 1.4 \
  --train_epochs 10 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1