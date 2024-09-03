#!/bin/bash
#SBATCH --job-name=VARIFOLDNoiseRobLinTrend
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

# # Choose the model
# model_name=DLinear

# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do

#     script_name_str="Noise_Rob_${snr}"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 16 \
#         --orientation_kernel 'Distribution' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 16 \
#         --orientation_kernel 'Current' \
#         --sigma_t_or 1 \
#         --sigma_s_or 5 \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 16 \
#         --orientation_kernel 'UnorientedVarifold' \
#         --sigma_t_or 1 \
#         --sigma_s_or 10 \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 16 \
#         --orientation_kernel 'OrientedVarifold' \
#         --sigma_t_or 1000 \
#         --sigma_s_or 10 \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

# done






# Choose the model
model_name=TrendLSTM

snr_values=( 10 5 )

for snr in "${snr_values[@]}"
do

    script_name_str="Noise_Rob_${snr}"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'Distribution' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 1 \
        --d_model 256 \
        --e_layers 3 \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'Current' \
        --sigma_t_or 1 \
        --sigma_s_or 5 \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 1 \
        --d_model 256 \
        --e_layers 3 \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'UnorientedVarifold' \
        --sigma_t_or 1 \
        --sigma_s_or 10 \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 1 \
        --d_model 256 \
        --e_layers 3 \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'OrientedVarifold' \
        --sigma_t_or 1000 \
        --sigma_s_or 10 \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 1 \
        --d_model 256 \
        --e_layers 3 \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

done






# Choose the model
model_name=TrendTCN

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do

    script_name_str="Noise_Rob_${snr}"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'Distribution' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
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
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'Current' \
        --sigma_t_or 1 \
        --sigma_s_or 5 \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
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
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'UnorientedVarifold' \
        --sigma_t_or 1 \
        --sigma_s_or 10 \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
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
        --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 16 \
        --orientation_kernel 'OrientedVarifold' \
        --sigma_t_or 1000 \
        --sigma_s_or 10 \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 1 \
        --out_dim_first_layer 64 \
        --e_layers 4 \
        --fixed_kernel_size_tcn 3 \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

done












































# snr_values=( 20 15 10 5 )

# gamma_list=(0.1 0.01 0.001)
# for gamma in "${gamma_list[@]}"
# do

#     for snr in "${snr_values[@]}"
#     do
#         script_name_str="Rob_LinTrend_DLinear_DILATE_alpha_1_gamma_${gamma}"
        
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
#             --des 'Exp' \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 1

#     done
# done















# orientation_kernels=( "UnorientedVarifold" "OrientedVarifold" )
# sigma_s_or_big_values=( 1 0.5 2 )

# for orientation_kernel in "${orientation_kernels[@]}"; do
#     if [ "$orientation_kernel" == "OrientedVarifold" ]; then
#         sigma_t_or_big=1000
#     else
#         sigma_t_or_big=1
#     fi
    
#     for sigma_s_or_big in "${sigma_s_or_big_values[@]}"; do


#         script_name_str="KorDLin"

#         # sigma_t_or_little 0.05

#         python -u run.py \
#             --is_training 1 \
#             --root_path ./dataset/synthetic/ \
#             --data_path Periodic_Sigmoid_With_Trend_V2.csv \
#             --evaluation_mode 'raw' \
#             --script_name $script_name_str \
#             --model $model_name \
#             --loss 'VARIFOLD' \
#             --number_of_kernels 2 \
#             --position_kernel_little 'Gaussian' \
#             --weight_little 0.02 \
#             --sigma_t_pos_little 1 \
#             --sigma_s_pos_little 0.1 \
#             --position_kernel_big 'Gaussian' \
#             --weight_big 0.98 \
#             --sigma_t_pos_big 1 \
#             --sigma_s_pos_big 5 \
#             --sigma_t_or_big $sigma_t_or_big \
#             --sigma_s_or_big $sigma_s_or_big \
#             --orientation_kernel_big $orientation_kernel \
#             --train_epochs 20 \
#             --patience 5 \
#             --data custom \
#             --features S \
#             --target value \
#             --seq_len 192 \
#             --pred_len 192 \
#             --enc_in 1 \
#             --des 'Exp' \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 1
#     done
# done


































# snr_values=( 20 15 10 5 )
# gamma_values=( 0.0001 0.001 0.01 )

# for gamma in "${gamma_values[@]}"
# do
#     for snr in "${snr_values[@]}"
#     do

#         script_name_str="Rob_LinTrend_DLinear_DILATE"
        
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
#             --des 'Exp' \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 5

#     done
# done


# script_name_str="Rob_LinTrend_DLinear_MSE"
        
        # python -u run.py \
        #     --is_training 1 \
        #     --root_path ./dataset/synthetic/ \
        #     --data_path Noise_Robustness_LinTrend_SNR_${snr}.csv \
        #     --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
        #     --evaluation_mode 'structural' \
        #     --script_name $script_name_str \
        #     --model $model_name \
        #     --loss 'MSE' \
        #     --train_epochs 20 \
        #     --patience 5 \
        #     --data custom \
        #     --features S \
        #     --target value \
        #     --seq_len 96 \
        #     --pred_len 96 \
        #     --enc_in 1 \
        #     --des 'Exp' \
        #     --batch_size 4 \
        #     --learning_rate 0.0001 \
        #     --itr 1




# script_name_str="Rob_LinTrend_DLinear_DILATE_infty"
    
# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#     --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'DILATE' \
#     --alpha_dilate 0.05 \
#     --gamma_dilate 0.01 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# script_name_str="Rob_LinTrend_DLinear_MSE_infty"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#     --structural_data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1






# model_name_str="DLinear_NoiseRob_MSE"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Noise_Robustness_LinTrend_SNR_infty.csv \
#     --script_name $model_name_str \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1






