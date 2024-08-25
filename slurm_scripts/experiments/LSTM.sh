#!/bin/bash
#SBATCH --job-name=Tuning_hyperparams_noise_robustness_W_96_H_96_LSTM
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
model_name=LSTM



hyperparams_list=(
    "4096 2"
    "4096 3"
    "4096 4"
    "8192 2"
    "8192 3"
    "8192 4"
)

for pair_hyperparam in "${hyperparams_list[@]}"
do

    dmodel=$(echo $pair_hyperparam | cut -d' ' -f1)
    elayers=$(echo $pair_hyperparam | cut -d' ' -f2)
    
    dmodel_str=$(echo $dmodel | sed 's/\./dot/g')
    elayers_str=$(echo $elayers | sed 's/\./dot/g')
    
    script_name_str="Tuning_hyperparams_noise_robustness_W_96_H_96_LSTM_d_model_${dmodel_str}_e_layers_${elayers_str}"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_SNR_20.csv \
        --evaluation_mode 'raw' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'MSE' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 96 \
        --pred_len 96 \
        --enc_in 1 \
        --des 'Exp' \
        --d_model ${dmodel} \
        --e_layers ${elayers} \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

done










# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_ARatio_3dot5_FundFreq_0dot00075_FreqRatio_8_Components_4.csv \
#     --evaluation_mode 'raw' \
#     --script_name 'Loss_MSE_Model_LSTM_Dataset_Fractal_W_2000_H_2000_d_model_1024_e_layers_2_gpu_a100' \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 2000 \
#     --pred_len 2000 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --d_model 1024 \
#     --e_layers 2 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 3



# snr_values=(5 10 15 20)

# for snr in "${snr_values[@]}"
# do
    
#     model_name_str="August_18_SNR_${snr}_Synth_3_structural_LSTM_MSE_d_model_1024_B_4_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Synth_3_SNR_${snr}.csv \
#         --structural_data_path Synth_3_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --e_layers 2 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

# done


# for snr in "${snr_values[@]}"
# do
    
#     model_name_str="August_18_SNR_${snr}_Synth_3_structural_LSTM_VAR_PosOnly_1_1_1_1_d_model_1024_B_4_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Synth_3_SNR_${snr}.csv \
#         --structural_data_path Synth_3_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --or_kernel 'PosOnly' \
#         --sigma_t_1 1 \
#         --sigma_s_1 1 \
#         --sigma_t_2 1 \
#         --sigma_s_2 1 \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 96 \
#         --pred_len 96 \
#         --e_layers 2 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

# done







# # SNR datasets, LSTM, Config 1

# # LSTM - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_2_SNR_15.csv \
#   --model_id Synth_2_SNR_15_raw_LSTM_MSE_d_model_1024_B_4_lr_10e-4 \
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


# # LSTM - SNR - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_2_SNR_15.csv \
#   --model_id Synth_2_SNR_15_raw_LSTM_VAR_PosOnly_1_1_1_1_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'PosOnly' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 1 \
#   --sigma_s_2 1 \
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

# # LSTM - SNR - VARIFOLD sum kernel
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --model_id updated_metrics_SNR_5_LSTM_VARIFOLD_sum_kernel_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Sum_Kernels' \
#   --sigma_t_1_little 1 \
#   --sigma_t_2_little 1 \
#   --sigma_s_1_little 0.5 \
#   --sigma_s_2_little 0.5 \
#   --sigma_t_1_big 2 \
#   --sigma_t_2_big 2 \
#   --sigma_s_1_big 1 \
#   --sigma_s_2_big 1 \
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




# # SNR datasets, LSTM, Config 2

# # LSTM - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_3_SNR_15.csv \
#   --structural_data_path Synth_3_SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id Synth_3_SNR_15_structural_LSTM_MSE_d_model_1024_B_4_lr_10e-4 \
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

# # LSTM - SNR - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_3_SNR_15.csv \
#   --structural_data_path Synth_3_SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id Synth_3_SNR_15_structural_LSTM_VAR_Gaussian_1_1_1_1_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Gaussian' \
#   --sigma_t_1 1 \
#   --sigma_s_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_2 1 \
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

# # LSTM - SNR - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_5.csv \
#   --structural_data_path SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id updated_metrics_SNR_5_structural_LSTM_VARIFOLD_sum_kernel_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'Sum_Kernels' \
#   --sigma_t_1_little 1 \
#   --sigma_t_2_little 1 \
#   --sigma_s_1_little 0.5 \
#   --sigma_s_2_little 0.5 \
#   --sigma_t_1_big 2 \
#   --sigma_t_2_big 2 \
#   --sigma_s_1_big 1 \
#   --sigma_s_2_big 1 \
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
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path increase_H.csv \
#   --model_id increase_H_bis_96_LSTM_VARIFOLD_1st_version_d_model_1024_B_4_lr_10e-4 \
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