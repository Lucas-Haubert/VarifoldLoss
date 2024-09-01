#!/bin/bash
#SBATCH --job-name=MultiScaleTrendLSTMTuning
#SBATCH --output=new_slurm_outputs/%x.job_%j
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
model_name=TrendLSTM


d_model_list=( 1024 2048 4096 )
e_layers_list=( 3 2 1 )

for d_model in "${d_model_list[@]}"
do
    for e_layers in "${e_layers_list[@]}"
    do

        script_name_str="LSTMTune_${d_model}_${e_layers}"

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
            --d_model $d_model \
            --e_layers $e_layers \
            --des 'Exp' \
            --batch_size 4 \
            --learning_rate 0.0001 \
            --itr 1
    done
done











# snr_values=( 20 15 )
# gamma_values=( 0.0001 0.001 0.01 )

# for gamma in "${gamma_values[@]}"
# do
#     for snr in "${snr_values[@]}"
#     do
#         script_name_str="Rob_LinTrend_TrendLSTM_DILATE"
        
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
#             --d_model 256 \
#             --e_layers 3 \
#             --des 'Exp' \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 5
#     done
# done



