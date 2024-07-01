#!/bin/bash
#SBATCH --job-name=search_B_and_lr_B_16_lr_0dot01_sigma_1_1_05sqrt_05sqrt_iTransformer_traffic_VARIFOLD_Gauss_96_96
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
model_name=iTransformer


# iTransformer - traffic - VARIFOLD - Gaussian
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id B_16_lr_0dot01_sigma_1_1_05sqrt_05sqrt_iTransformer_traffic_VARIFOLD_Gauss_96_96 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --or_kernel 'Gaussian' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 14.7 \
  --sigma_s_2 14.7 \
  --train_epochs 20 \
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
  --learning_rate 0.01 \
  --itr 1


# # iTransformer - traffic - DILATE - alpha
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id B_16_lr_0dot0001_alpha_08_iTransformer_traffic_DILATE_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.8 \
#   --train_epochs 20 \
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
#   --learning_rate 0.0001 \
#   --itr 1


# # iTransformer - electricity - DILATE - alpha
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id B_32_lr_0dot0001_alpha_1_iTransformer_electricity_DILATE_96_96_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 1 \
#   --train_epochs 20 \
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
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1




# # iTransformer - electricity - VARIFOLD - Gaussian
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id sigma_1_1_05sqrt_1_iTransformer_electricity_VARIFOLD_GaussGauss_96_96 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 8.9 \
#   --sigma_s_2 1 \
#   --train_epochs 20 \
#   --patience 10 \
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
#   --batch_size 32 \
#   --learning_rate 0.01 \
#   --itr 1









# # iTransformer - exchange_rate - DILATE - alpha
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id iTransformer_exchange_rate_DILATE_1_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 1 \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1


# # iTransformer - electricity - DILATE - alpha
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id iTransformer_electricity_DILATE_alpha_03_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.3 \
#   --train_epochs 20 \
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
#   --learning_rate 0.001 \
#   --itr 1


















# traffic dataset: strong seasonality

# # iTransformer - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id iTransformer_traffic_MSE_96_96 \
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

# # iTransformer - traffic - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id iTransformer_traffic_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
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



# ECL: medium seasonality

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

# # iTransformer - electricity - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id iTransformer_electricity_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
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



# exchange_rate: weak seasonality

# # iTransformer - exchange_rate - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id iTransformer_exchange_rate_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1

# # iTransformer - exchange_rate - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id iTransformer_exchange_rate_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1





