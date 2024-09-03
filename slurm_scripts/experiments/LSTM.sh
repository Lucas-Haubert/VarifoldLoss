#!/bin/bash
#SBATCH --job-name=DILATENoiseRobSimpleLSTM
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
model_name=LSTM

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Simple_LSTM_DILATE"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
        --evaluation_mode 'structural' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'DILATE' \
        --alpha_dilate 0.05 \
        --gamma_dilate 0.1 \
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
        --e_layers 1 \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

    script_name_str="Rob_Simple_LSTM_MSE"

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
        --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
        --evaluation_mode 'structural' \
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
        --d_model 512 \
        --e_layers 1 \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1
done




# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="RobSimpleLSTMVarifold_${snr}"
    
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
#         --itr 5
# done

# d_model_list=( 256 512 126 1024 )
# e_layers_list=( 3 2 1 )

# for d_model in "${d_model_list[@]}"
# do
#     for e_layers in "${e_layers_list[@]}"
#     do
#         script_name_str="FractalTuningLSTM_${d_model}_${e_layers}"
        
#         python -u run.py \
#             --is_training 1 \
#             --root_path ./dataset/synthetic/ \
#             --data_path Fractal_Config_2_Components_3.csv \
#             --structural_data_path Fractal_Config_2_Components_3.csv \
#             --evaluation_mode 'structural' \
#             --script_name $script_name_str \
#             --model $model_name \
#             --loss 'MSE' \
#             --train_epochs 20 \
#             --patience 5 \
#             --data custom \
#             --features S \
#             --target value \
#             --seq_len 336 \
#             --pred_len 336 \
#             --enc_in 1 \
#             --des 'Exp' \
#             --d_model $d_model \
#             --e_layers $e_layers \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 1
#     done
# done



# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="Rob_Simple_LSTM_VARIFOLD"
    
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
#         --sigma_t_or 1 \
#         --sigma_s_or 1 \
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
#         --orientation_kernel 'Current' \
#         --sigma_t_or 1 \
#         --sigma_s_or 1 \
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
#         --orientation_kernel 'UnorientedVarifold' \
#         --sigma_t_or 1 \
#         --sigma_s_or 1 \
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
#         --orientation_kernel 'OrientedVarifold' \
#         --sigma_t_or 1000 \
#         --sigma_s_or 1 \
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

            























# done

# script_name_str="Rob_LinTrend_LSTM_DILATE_infty"
    
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
#     --d_model 512 \
#     --e_layers 1 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# script_name_str="Rob_LinTrend_LSTM_MSE_infty"

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
#     --d_model 512 \
#     --e_layers 1 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1








# hyperparams_list=(
#     "1024 1"
#     "1024 2"
#     "1024 3" 
#     "512 1"
#     "512 2"
#     "512 3"
#     "2048 1"
#     "2048 2"
#     "2048 3"
#     "256 1"
#     "256 2"
#     "256 3"
#     "4096 1"
#     "4096 2"
#     "4096 3"
# )

# for pair_hyperparam in "${hyperparams_list[@]}"
# do

#     dmodel=$(echo $pair_hyperparam | cut -d' ' -f1)
#     elayers=$(echo $pair_hyperparam | cut -d' ' -f2)
    
#     script_name_str="Tuning_LSTM_Noise_Rob_d_model_${dmodel}_e_layers_${elayers}"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_Simple_SNR_infty.csv \
#         --evaluation_mode 'raw' \
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
#         --d_model ${dmodel} \
#         --e_layers ${elayers} \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

# done

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