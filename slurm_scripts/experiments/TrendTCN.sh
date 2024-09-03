#!/bin/bash
#SBATCH --job-name=MULTISCALE
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
# model_name=TrendTCN


# model_name=DLinear

# script_name_str="MULTISCALEDLinear"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 0.1 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 5 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --weight_little 0.02 \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --weight_big 0.98 \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 5 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --weight_little 0.02 \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --weight_big 0.98 \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 5 \
#     --orientation_kernel_big 'Current' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 5 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --weight_little 0.02 \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --weight_big 0.98 \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 5 \
#     --orientation_kernel_big 'UnorientedVarifold' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 5 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --weight_little 0.02 \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --weight_big 0.98 \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 5 \
#     --orientation_kernel_big 'OrientedVarifold' \
#     --sigma_t_or_big 1000 \
#     --sigma_s_or_big 5 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1












# # Choose the model
# model_name=TrendTCN

# script_name_str="MULTISCALETTCN"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --out_dim_first_layer 64 \
#     --e_layers 4 \
#     --fixed_kernel_size_tcn 3 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 0.1 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --out_dim_first_layer 64 \
#     --e_layers 4 \
#     --fixed_kernel_size_tcn 3 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 5 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --out_dim_first_layer 64 \
#     --e_layers 4 \
#     --fixed_kernel_size_tcn 3 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

model_name=TrendTCN
script_name_str="MULTISCALETTCN"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --out_dim_first_layer 64 \
    --e_layers 4 \
    --fixed_kernel_size_tcn 3 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'Current' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --out_dim_first_layer 64 \
    --e_layers 4 \
    --fixed_kernel_size_tcn 3 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'UnorientedVarifold' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --out_dim_first_layer 64 \
    --e_layers 4 \
    --fixed_kernel_size_tcn 3 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'OrientedVarifold' \
    --sigma_t_or_big 1000 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --out_dim_first_layer 64 \
    --e_layers 4 \
    --fixed_kernel_size_tcn 3 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1       








# Choose the model
model_name=TrendLSTM

script_name_str="MULTISCALETLSTM"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 0.1 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'Current' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'UnorientedVarifold' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Periodic_Sigmoid_With_Trend_V2.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'OrientedVarifold' \
    --sigma_t_or_big 1000 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1       



# #Choose the model
# model_name=MLP

# script_name_str="Heatmap_MSE"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_Config_2_Components_3.csv \
#     --structural_data_path Fractal_Config_2_Components_3.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 336 \
#     --pred_len 336 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --d_model 4096 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1


















# snr_values=( 10 5 )
# gamma_values=( 0.0001 0.001 0.01 )

# for gamma in "${gamma_values[@]}"
# do
#     for snr in "${snr_values[@]}"
#     do
#         script_name_str="Rob_LinTrend_TrendTCN_DILATE"
        
#         python -u run.py \
#             --is_training 1 \
#             --root_path ./dataset/synthetic/ \
#             --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
#             --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#             --evaluation_mode 'structural' \
#             --script_name $script_name_str \
#             --model $model_name \
#             --loss 'DILATE' \
#             --alpha_dilate 1 \
#             --gamma_dilate $gamma \
#             --train_epochs 20 \
#             --patience 5 \
#             --data custom \
#             --features S \
#             --target value \
#             --seq_len 96 \
#             --pred_len 96 \
#             --enc_in 1 \
#             --out_dim_first_layer 64 \
#             --e_layers 4 \
#             --fixed_kernel_size_tcn 3 \
#             --des 'Exp' \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 5
#     done
# done



# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="Rob_LinTrend_TrendTCN_DILATE"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'DILATE' \
#         --alpha_dilate 0.05 \
#         --gamma_dilate 0.1 \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --enc_in 1 \
#         --out_dim_first_layer 64 \
#         --e_layers 4 \
#         --fixed_kernel_size_tcn 3 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

#     script_name_str="Rob_LinTrend_TrendTCN_MSE"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --enc_in 1 \
#         --out_dim_first_layer 64 \
#         --e_layers 4 \
#         --fixed_kernel_size_tcn 3 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

# done