#!/bin/bash
#SBATCH --job-name=Tuning_DILATE_DLinear_ETTm1_MSE_horizon
#SBATCH --output=new_slurm_outputs/%x.job_%j
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
model_name=DLinear


horizon_list=(12 24 48 96)

for horizon in "${horizon_list[@]}"
do
    
    model_name_str="Tuning_DILATE_DLinear_ETTm1_MSE_horizon_${horizon}_d_model_512_B_32_lr_10e-4"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id $model_name_str \
        --model $model_name \
        --loss 'MSE' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len $horizon \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --itr 5

done

# for horizon in "${horizon_list[@]}"
# do
    
#     model_name_str="Tuning_DILATE_DLinear_Exchange_MSE_horizon_${horizon}_d_model_512_B_32_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/exchange_rate/ \
#         --data_path exchange_rate.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len $horizon \
#         --e_layers 2 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 8 \
#         --dec_in 8 \
#         --c_out 8 \
#         --des 'Exp' \
#         --batch_size 32 \
#         --learning_rate 0.0001 \
#         --itr 5

# done

# for horizon in "${horizon_list[@]}"
# do
    
#     model_name_str="Tuning_DILATE_DLinear_Electricity_MSE_horizon_${horizon}_d_model_512_B_32_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/electricity/ \
#         --data_path electricity.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len $horizon \
#         --e_layers 2 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 321 \
#         --dec_in 321 \
#         --c_out 321 \
#         --des 'Exp' \
#         --batch_size 32 \
#         --learning_rate 0.0001 \
#         --itr 5

# done

# for horizon in "${horizon_list[@]}"
# do
    
#     model_name_str="Tuning_DILATE_DLinear_Traffic_MSE_horizon_${horizon}_d_model_512_B_16_lr_10e-3"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/traffic/ \
#         --data_path traffic.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len $horizon \
#         --e_layers 2 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 862 \
#         --dec_in 862 \
#         --c_out 862 \
#         --des 'Exp' \
#         --batch_size 16 \
#         --learning_rate 0.001 \
#         --itr 5

# done

# for horizon in "${horizon_list[@]}"
# do
    
#     model_name_str="Tuning_DILATE_DLinear_Weather_MSE_horizon_${horizon}_d_model_512_B_32_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/weather/ \
#         --data_path weather.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len $horizon \
#         --e_layers 2 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 21 \
#         --dec_in 21 \
#         --c_out 21 \
#         --des 'Exp' \
#         --batch_size 32 \
#         --learning_rate 0.0001 \
#         --itr 5

# done


# alpha_dilate_list=(0.6 0.7 0.8)


# for alpha_dilate in "${alpha_dilate_list[@]}"
# do
    
#     alpha_dilate_name=$(echo $alpha_dilate | tr '.' 'dot')
    
#     model_name_str="Tuning_DILATE_DLinear_Weather_alpha_${alpha_dilate}_d_model_512_B_32_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/weather/ \
#         --data_path weather.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'DILATE' \
#         --alpha_dilate $alpha_dilate \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 144 \
#         --pred_len 72 \
#         --e_layers 2 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 21 \
#         --dec_in 21 \
#         --c_out 21 \
#         --des 'Exp' \
#         --batch_size 32 \
#         --learning_rate 0.0001 \
#         --itr 3

# done


# traffic

# # DLinear - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TUESDAY_MEETING_DLinear_traffic_MSE_B_16_lr_0dot001 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
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
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1

# # DLinear - traffic - DILATE_08
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TUESDAY_MEETING_DLinear_traffic_DILATE_alpha_08_B_16_lr_0dot001 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.8 \
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
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1

# # DLinear - traffic - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TUESDAY_MEETING_DLinear_traffic_VARIFOLD_sigma_1_1_05sqrt_05sqrt_B_16_lr_0dot001 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 14.7 \
#   --sigma_s_2 14.7 \
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
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1


# # electricity

# # DLinear - electricity - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TUESDAY_MEETING_DLinear_electricity_MSE_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1

# # DLinear - electricity - DILATE_08
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TUESDAY_MEETING_DLinear_electricity_DILATE_alpha_08_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.8 \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1

# # DLinear - electricity - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TUESDAY_MEETING_DLinear_electricity_VARIFOLD_sigma_1_1_05sqrt_05sqrt_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 8.9 \
#   --sigma_s_2 8.9 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 321 \
#   --dec_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1


# # exchange_rate

# DLinear - exchange_rate - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_DLinear_exchange_rate_MSE_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --factor 3 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1

# # DLinear - exchange_rate - DILATE_1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_DLinear_exchange_rate_DILATE_alpha_08_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 1 \
#   --train_epochs 20 \
#   --patience 10 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --factor 3 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1

# # DLinear - exchange_rate - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_DLinear_exchange_rate_VARIFOLD_sigma_1_1_05sqrt_05sqrt_B_32_lr_0dot0001 \
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
#   --factor 3 \
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1