#!/bin/bash
#SBATCH --job-name=Synth_1_SNR_15_structural_CNN_VAR_PosOnly_1_1_d_model_512_B_4_lr_10e-4
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
model_name=CNN



# # SNR datasets, CNN, MSE, Config 2

# CNN - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_1_SNR_15.csv \
#   --structural_data_path Synth_1_SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id Noise_Robustness_Synth_1_SNR_15_structural_CNN_MSE_d_model_512_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1


# Synthetic, CNN, VARIFOLD, Config 2

# # CNN - SNR - VARIFOLD
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/synthetic/ \
  --data_path Synth_1_SNR_15.csv \
  --structural_data_path Synth_1_SNR_infty.csv \
  --evaluation_mode 'structural' \
  --model_id Synth_1_SNR_15_structural_CNN_VAR_PosOnly_1_1_d_model_512_B_4_lr_10e-4 \
  --model $model_name \
  --loss 'VARIFOLD' \
  --or_kernel 'Gaussian' \
  --sigma_t_1 1 \
  --sigma_s_1 1 \
  --train_epochs 20 \
  --patience 5 \
  --data custom \
  --features S \
  --target value \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --itr 1













# # CNN - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path increase_H_bis.csv \
#   --model_id increase_H_bis_336_CNN_MSE_d_model_512_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 336 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5

# # CNN - increase H - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path increase_H_bis.csv \
#   --model_id increase_H_bis_96_CNN_VARIFOLD_1st_version_d_model_512_B_4_lr_10e-4 \
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
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5











# # CNN - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --model_id SNR_5_CNN_MSE_d_model_512_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5



# Synthetic, CNN, VARIFOLD, Config 1

# # CNN - SNR - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --model_id SNR_5_CNN_VARIFOLD_1st_version_d_model_512_B_4_lr_10e-4 \
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
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5


# Synthetic, CNN, VARIFOLD, Config 2

# # CNN - SNR - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --structural_data_path SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id SNR_5_structural_CNN_VARIFOLD_1st_version_d_model_512_B_4_lr_10e-4 \
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
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5



















# # synthetic_dataset_trial

# # CNN - synthetic_dataset_noise_and_jumps - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path synthetic_dataset_noise_and_jumps.csv \
#   --model_id noise_and_jumps_July_09_CNN_2layers_k_33_MSE_B_4_lr_10e-4_d_model_512 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1

# # CNN - synthetic_dataset_noise_and_jumps - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path synthetic_dataset_noise_and_jumps.csv \
#   --model_id noise_and_jumps_July_09_CNN_VARIFOLD_2layers_k_33_MSE_B_4_lr_10e-4_d_model_512 \
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
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1