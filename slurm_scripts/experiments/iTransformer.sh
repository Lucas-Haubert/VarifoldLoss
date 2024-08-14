#!/bin/bash
#SBATCH --job-name=Tuning_DILATE_iTransformer_Exchange_alpha
#SBATCH --output=slurm_outputs/%x.job_%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast

# Choose the model
model_name=iTransformer


alpha_dilate_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)


for alpha_dilate in "${alpha_dilate_list[@]}"
do
    
    alpha_dilate_name=$(echo $alpha_dilate | tr '.' 'dot')
    
    model_name_str="Tuning_DILATE_iTransformer_Exchange_alpha_${alpha_dilate}_d_model_512_B_32_lr_10e-4"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/exchange_rate/ \
        --data_path exchange_rate.csv \
        --model_id $model_name_str \
        --model $model_name \
        --loss 'DILATE' \
        --alpha_dilate $alpha_dilate \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 48 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --d_model 512 \
        --d_ff 512 \
        --des 'Exp' \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --itr 3

done








# alpha_dilate_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)


# for alpha_dilate in "${alpha_dilate_list[@]}"
# do
    
#     alpha_dilate_name=$(echo $alpha_dilate | tr '.' 'dot')
    
#     model_name_str="Tuning_DILATE_iTransformer_Electricity_alpha_${alpha_dilate}_d_model_512_B_16_lr_510e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/electricity/ \
#         --data_path electricity.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'DILATE' \
#         --alpha_dilate $alpha_dilate \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len 24 \
#         --e_layers 3 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 321 \
#         --dec_in 321 \
#         --c_out 321 \
#         --d_model 512 \
#         --d_ff 512 \
#         --des 'Exp' \
#         --batch_size 16 \
#         --learning_rate 0.0005 \
#         --itr 3

# done





# alpha_dilate_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)


# for alpha_dilate in "${alpha_dilate_list[@]}"
# do
    
#     alpha_dilate_name=$(echo $alpha_dilate | tr '.' 'dot')
    
#     model_name_str="Tuning_DILATE_iTransformer_Weather_alpha_${alpha_dilate}_d_model_512_B_32_lr_10e-4"
    
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
#         --e_layers 3 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 21 \
#         --dec_in 21 \
#         --c_out 21 \
#         --d_model 512 \
#         --d_ff 512 \
#         --des 'Exp' \
#         --batch_size 32 \
#         --learning_rate 0.0001 \
#         --itr 3

# done




# alpha_dilate_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)


# for alpha_dilate in "${alpha_dilate_list[@]}"
# do
    
#     alpha_dilate_name=$(echo $alpha_dilate | tr '.' 'dot')
    
#     model_name_str="Tuning_DILATE_iTransformer_Traffic_alpha_${alpha_dilate}_d_model_512_B_16_lr_10e-3"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/traffic/ \
#         --data_path traffic.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'DILATE' \
#         --alpha_dilate $alpha_dilate \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 168 \
#         --pred_len 24 \
#         --e_layers 4 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 862 \
#         --dec_in 862 \
#         --c_out 862 \
#         --d_model 512 \
#         --d_ff 512 \
#         --des 'Exp' \
#         --batch_size 16 \
#         --learning_rate 0.001 \
#         --itr 3

# done


# alpha_dilate_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)


# for alpha_dilate in "${alpha_dilate_list[@]}"
# do
    
#     alpha_dilate_name=$(echo $alpha_dilate | tr '.' 'dot')
    
#     model_name_str="Tuning_DILATE_iTransformer_ETTm1_alpha_${alpha_dilate}_d_model_512_B_32_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/ETT-small/ \
#         --data_path ETTm1.csv \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'DILATE' \
#         --alpha_dilate $alpha_dilate \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features M \
#         --seq_len 96 \
#         --pred_len 24 \
#         --e_layers 2 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 7 \
#         --dec_in 7 \
#         --c_out 7 \
#         --d_model 128 \
#         --d_ff 128 \
#         --des 'Exp' \
#         --batch_size 32 \
#         --learning_rate 0.0001 \
#         --itr 3

# done



# # traffic

# # iTransformer - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id Traffic_iTrans_MSE_B_32_lr_0dot001 \
#   --model $model_name \
#   --loss 'MSE' \
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
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1

# # iTransformer - traffic - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id Traffic_iTrans_VAR_PosOnly_1_1_14dot7_14dot7_B_32_lr_0dot001 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'PosOnly' \
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
#   --d_ff 512 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --itr 1



# # iTransformer - traffic - DILATE_08
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TUESDAY_MEETING_iTransformer_traffic_DILATE_alpha_1_B_16_lr_0dot001 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.8 \
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

# # iTransformer - traffic - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id TUESDAY_MEETING_iTransformer_traffic_VARIFOLD_sigma_1_1_05sqrt_05sqrt_B_16_lr_0dot001 \
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
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1












# # electricity

# # iTransformer - electricity - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TUESDAY_MEETING_iTransformer_electricity_MSE_B_16_lr_0dot001 \
#   --model $model_name \
#   --loss 'MSE' \
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
#   --learning_rate 0.0005 \
#   --itr 1

# # iTransformer - electricity - DILATE_08
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TUESDAY_MEETING_iTransformer_electricity_DILATE_alpha_1_B_16_lr_0dot0005 \
#   --model $model_name \
#   --loss 'DILATE' \
#   --alpha_dilate 0.8 \
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

# iTransformer - electricity - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id TUESDAY_MEETING_iTransformer_electricity_VARIFOLD_sigma_1_1_05sqrt_05sqrt_B_16_lr_0dot001 \
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







# # exchange_rate

# # iTransformer - exchange_rate - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_iTransformer_exchange_rate_MSE_B_32_lr_0dot0001 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 20 \
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
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1

# # iTransformer - exchange_rate - DILATE_1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_iTransformer_exchange_rate_DILATE_alpha_1_B_32_lr_0dot0001 \
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
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1

# # iTransformer - exchange_rate - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id TUESDAY_MEETING_iTransformer_exchange_rate_VARIFOLD_sigma_1_1_05sqrt_05sqrt_B_32_lr_0dot0001 \
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
#   --e_layers 2 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 512 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --itr 1






















# # iTransformer - traffic - VARIFOLD - Gaussian
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id B_16_lr_0dot01_sigma_1_1_05sqrt_05sqrt_iTransformer_traffic_VARIFOLD_Gauss_96_96 \
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
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.01 \
#   --itr 1


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





