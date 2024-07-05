#!/bin/bash
#SBATCH --job-name=3Sum_time_1_03_01_space_05sqrt_015sqrt_005sqrt_SegRNN_electricity_VARIFOLD_B_32_lr_0dot0001
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


# For Tuesday meeting


# # traffic

# # SegRNN - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id H_336_SegRNN_traffic_MSE_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - traffic - DILATE_08
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TUESDAY_MEETING_SegRNN_traffic_DILATE_alpha_1_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.8 \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - traffic - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id 3Sum_time_1_03_01_space_05sqrt_015sqrt_005sqrt_SegRNN_traffic_VARIFOLD_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel '3Sum_Gaussian' \
#   --sigma_t_1_kernel_1 1 \
#   --sigma_t_1_kernel_2 0.3 \
#   --sigma_t_1_kernel_3 0.1 \
#   --sigma_s_1_kernel_1 14.7 \
#   --sigma_s_1_kernel_2 4.4 \
#   --sigma_s_1_kernel_3 1.5 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1








# # SegRNN - electricity - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id H_336_SegRNN_electricity_MSE_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 321 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - electricity - DILATE_08
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TUESDAY_MEETING_SegRNN_electricity_DILATE_alpha_1_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.8 \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 321 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# SegRNN - electricity - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id 3Sum_time_1_03_01_space_05sqrt_015sqrt_005sqrt_SegRNN_electricity_VARIFOLD_B_32_lr_0dot0001 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --or_kernel '3Sum_Gaussian' \
  --sigma_t_1_kernel_1 1 \
  --sigma_t_1_kernel_2 0.3 \
  --sigma_t_1_kernel_3 0.1 \
  --sigma_s_1_kernel_1 8.9 \
  --sigma_s_1_kernel_2 2.7 \
  --sigma_s_1_kernel_3 0.9 \
  --train_epochs 20 \
  --patience 5 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
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









# # exchange_rate

# # SegRNN - exchange_rate - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_SegRNN_exchange_rate_MSE_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - exchange_rate - DILATE_1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_SegRNN_exchange_rate_DILATE_alpha_1_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 1 \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - exchange_rate - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_SegRNN_exchange_rate_VARIFOLD_sigma_1_1_05sqrt_05sqrt_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 1.4 \
#   --sigma_s_2 1.4 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1








































# # traffic dataset: strong seasonality

# # SegRNN - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id SegRNN_traffic_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - traffic - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id SegRNN_traffic_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 862 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1



# # ECL: medium seasonality

# # SegRNN - electricity - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id SegRNN_electricity_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 321 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - electricity - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id SegRNN_electricity_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 321 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 2048 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1



# # exchange_rate: weak seasonality

# # SegRNN - exchange_rate - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id SegRNN_exchange_rate_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # SegRNN - exchange_rate - DILATE_05
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id SegRNN_exchange_rate_DILATE_05_96_96 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --seg_len 24 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --dropout 0 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1