#!/bin/bash
#SBATCH --job-name=RobSimpleTCNVarifold
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
model_name=TCN

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do
    script_name_str="RobSimpleTCNVarifold_${snr}"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'VARIFOLD' \
        --position_kernel 'Gaussian' \
        --sigma_t_pos 1 \
        --sigma_s_pos 0.5 \
        --orientation_kernel 'Distribution' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 1 \
        --des 'Exp' \
        --out_dim_first_layer 64 \
        --e_layers 4 \
        --fixed_kernel_size_tcn 3 \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done




# outfirstdim_list=( 256 512 64 128 )
# elayers_list=( 4 3 2 )
# kernelsize_list=( 3 5 7 )

# for kernelsize in "${kernelsize_list[@]}"
# do
#     for elayers in "${elayers_list[@]}"
#     do
#         for outfirstdimlay in "${outfirstdim_list[@]}"
#         do
#             script_name_str="FractalTuningTCN_${outfirstdimlay}_${elayers}_${kernelsize}"
            
#             python -u run.py \
#                 --is_training 1 \
#                 --root_path ./dataset/synthetic/ \
#                 --data_path Fractal_Config_2_Components_3.csv \
#                 --structural_data_path Fractal_Config_2_Components_3.csv \
#                 --evaluation_mode 'structural' \
#                 --script_name $script_name_str \
#                 --model $model_name \
#                 --loss 'MSE' \
#                 --train_epochs 20 \
#                 --patience 5 \
#                 --data custom \
#                 --features S \
#                 --target value \
#                 --seq_len 336 \
#                 --pred_len 336 \
#                 --enc_in 1 \
#                 --des 'Exp' \
#                 --out_dim_first_layer ${outfirstdimlay} \
#                 --e_layers ${elayers} \
#                 --fixed_kernel_size_tcn ${kernelsize} \
#                 --batch_size 4 \
#                 --learning_rate 0.0001 \
#                 --itr 1

#         done
#     done
# done







# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="RobSimpleTCNVarifold"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 0.5 \
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
#         --out_dim_first_layer 64 \
#         --e_layers 4 \
#         --fixed_kernel_size_tcn 3 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1
# done

# model_name=LSTM


# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="RobSimpleLSTMVarifold"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 0.5 \
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
#         --d_model 512 \
#         --e_layers 2 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1
# done









# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="Rob_Simple_TCN_DILATE"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
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
#         --des 'Exp' \
#         --out_dim_first_layer 64 \
#         --e_layers 4 \
#         --fixed_kernel_size_tcn 3 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

#     script_name_str="Rob_Simple_TCN_MSE"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
#         --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
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
#         --des 'Exp' \
#         --out_dim_first_layer 64 \
#         --e_layers 4 \
#         --fixed_kernel_size_tcn 3 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

# done

# script_name_str="Rob_LinTrend_TCN_DILATE_infty"
    
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
#     --gamma_dilate 0.1 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --out_dim_first_layer 64 \
#     --e_layers 4 \
#     --fixed_kernel_size_tcn 3 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# script_name_str="Rob_LinTrend_TCN_MSE_infty"

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
#     --out_dim_first_layer 64 \
#     --e_layers 4 \
#     --fixed_kernel_size_tcn 3 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1






# outfirstdim_list=(64 128 32 512 16 1024)
# elayers_list=(4 3 2 5 1)
# kernelsize_list=(7 5 11 3 15 25)

# for outfirstdimlay in "${outfirstdim_list[@]}"
# do
#     for elayers in "${elayers_list[@]}"
#     do
#         for kernelsize in "${kernelsize_list[@]}"
#         do
#             script_name_str="Tuning_TCN_Noise_Red_on_MSE_${outfirstdimlay}_${elayers}_${kernelsize}"
            
#             python -u run.py \
#                 --is_training 1 \
#                 --root_path ./dataset/synthetic/ \
#                 --data_path Noise_Robustness_Simple_SNR_infty.csv \
#                 --evaluation_mode 'raw' \
#                 --script_name $script_name_str \
#                 --model $model_name \
#                 --loss 'MSE' \
#                 --train_epochs 20 \
#                 --patience 5 \
#                 --data custom \
#                 --features S \
#                 --target value \
#                 --seq_len 96 \
#                 --pred_len 96 \
#                 --enc_in 1 \
#                 --des 'Exp' \
#                 --out_dim_first_layer ${outfirstdimlay} \
#                 --e_layers ${elayers} \
#                 --fixed_kernel_size_tcn ${kernelsize} \
#                 --batch_size 4 \
#                 --learning_rate 0.0001 \
#                 --itr 1

#         done
#     done
# done