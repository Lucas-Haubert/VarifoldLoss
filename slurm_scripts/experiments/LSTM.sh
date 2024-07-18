#!/bin/bash
#SBATCH --job-name=increase_H_bis_96_LSTM_VARIFOLD_1st_version_d_model_1024_B_4_lr_10e-4
#SBATCH --output=slurm_outputs/%x.job_%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast

# Choose the model
model_name=LSTM


# LSTM - increase_H - MSE - Config 1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path increase_H_bis.csv \
#   --model_id increase_H_bis_96_LSTM_MSE_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5


# # LSTM - increase_H - VARIFOLD - Config 1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/synthetic/ \
  --data_path increase_H.csv \
  --model_id increase_H_bis_96_LSTM_VARIFOLD_1st_version_d_model_1024_B_4_lr_10e-4 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --or_kernel 'Gaussian' \
  --sigma_t_1 1 \
  --sigma_t_2 1 \
  --sigma_s_1 0.5 \
  --sigma_s_2 0.5 \
  --train_epochs 20 \
  --patience 5 \
  --data custom \
  --features S \
  --target value \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 1 \
  --des 'Exp' \
  --d_model 1024 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --itr 5












# # LSTM - SNR - VARIFOLD Sum big and small - Config 2
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --structural_data_path SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id SNR_5_structural_LSTM_VARIFOLD_sum_small_big_alpha_05_sigmas_1_025_1_1_5_05_1_1_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
  # --sigma_t_1 1 \
  # --sigma_t_2 1 \
  # --sigma_s_1 0.5 \
  # --sigma_s_2 0.5 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5



































































# # SNR datasets, LSTM, Config 1

# # LSTM - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --model_id SNR_5_LSTM_MSE_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5


# # LSTM - SNR - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --model_id SNR_5_LSTM_VARIFOLD_1st_version_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 0.5 \
#   --sigma_s_2 0.5 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5




# # SNR datasets, LSTM, Config 2

# # LSTM - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --structural_data_path SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id SNR_5_structural_LSTM_MSE_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5

# # LSTM - SNR - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --structural_data_path SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id SNR_5_structural_LSTM_VARIFOLD_1st_version_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 0.5 \
#   --sigma_s_2 0.5 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5






# # synthetic_dataset_trial

# # LSTM - synthetic_dataset_noise_and_jumps - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path synthetic_dataset_noise_and_jumps.csv \
#   --model_id noise_and_jumps_July_09_LSTM_MSE_B_4_lr_10e-4_d_model_1024_num_layers_2 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1

# # LSTM - synthetic_dataset_noise_and_jumps - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path synthetic_dataset_noise_and_jumps.csv \
#   --model_id noise_and_jumps_July_09_LSTM_VARIFOLD_B_4_lr_10e-4_d_model_1024_num_layers_2 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 0.5 \
#   --sigma_s_2 0.5 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1