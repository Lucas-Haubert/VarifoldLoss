#!/bin/bash
#SBATCH --job-name=RobSimpleMLPVarifold
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
model_name=MLP

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do
    script_name_str="RobSimpleMLPVarifold_${snr}"
    
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
        --d_model 1024 \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done

# sigma_t_pos_values=( 1 10 100 )
# sigma_s_pos_values_1=( 0.05 0.1 0.2 0.5 1 2 5 10 15 50 )
# sigma_s_pos_values_10=( 0.1 0.5 1 2 5 10 15 50 100 200 )
# sigma_s_pos_values_100=( 5 10 20 50 100 200 500 )

# for sigma_t_pos in "${sigma_t_pos_values[@]}"
# do
#     if [ "$sigma_t_pos" -eq 1 ]; then
#         sigma_s_pos_values=("${sigma_s_pos_values_1[@]}")
#     elif [ "$sigma_t_pos" -eq 10 ]; then
#         sigma_s_pos_values=("${sigma_s_pos_values_10[@]}")
#     elif [ "$sigma_t_pos" -eq 100 ]; then
#         sigma_s_pos_values=("${sigma_s_pos_values_100[@]}")
#     fi

#     for sigma_s_pos in "${sigma_s_pos_values[@]}"
#     do
#         script_name_str="FractalBigGrid"
        
#         python -u run.py \
#             --is_training 1 \
#             --root_path ./dataset/synthetic/ \
#             --data_path Fractal_Config_2_Components_3.csv \
#             --structural_data_path Fractal_Config_2_Components_3.csv \
#             --evaluation_mode 'structural' \
#             --script_name $script_name_str \
#             --model $model_name \
#             --loss 'VARIFOLD' \
#             --position_kernel 'Gaussian' \
#             --sigma_t_pos $sigma_t_pos \
#             --sigma_s_pos $sigma_s_pos \
#             --orientation_kernel 'Distribution' \
#             --train_epochs 20 \
#             --patience 5 \
#             --data custom \
#             --features S \
#             --target value \
#             --seq_len 336 \
#             --pred_len 336 \
#             --enc_in 1 \
#             --des 'Exp' \
#             --d_model 4096 \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 1
#     done

# done








# script_name_str="Per_Sig_Without_Trend_LittleKernel"

# # Little Only
# # 96 96

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_Without_Trend.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 0.1 \
#     --orientation_kernel 'Distribution' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --d_model 1024 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# # Little Only
# # 192 192

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_Without_Trend.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 0.1 \
#     --orientation_kernel 'Distribution' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --d_model 1024 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# script_name_str="Per_Sig_Without_Trend_BigKernel"

# # Big Only
# # 96 96

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_Without_Trend.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 1.5 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 1.5 \
#     --orientation_kernel 'Distribution' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --d_model 1024 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# # Big Only
# # 192 192

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_Without_Trend.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 1.5 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 1.5 \
#     --orientation_kernel 'Distribution' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target value \
#     --seq_len 192 \
#     --pred_len 192 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --d_model 1024 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# script_name_str="Per_Sig_Without_Trend_MSE"

# # MSE
# # 96 96

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_Without_Trend.csv \
#     --evaluation_mode 'raw' \
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
#     --d_model 1024 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1

# # MSE
# # 192 192

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Periodic_Sigmoid_Without_Trend.csv \
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
#     --d_model 1024 \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --itr 1












# snr_values=( 10 5 )

# sigma_s_or_values=( 0.25 0.5 1 2 5 )
# orientation_kernels=("Current" "UnorientedVarifold" "OrientedVarifold")

# for snr in "${snr_values[@]}"
# do

#     for orientation_kernel in "${orientation_kernels[@]}"; do
#         if [ "$orientation_kernel" == "OrientedVarifold" ]; then
#             sigma_t_or=1000
#         else
#             sigma_t_or=1
#         fi
        
#         for sigma_s_or in "${sigma_s_or_values[@]}"; do
#             script_name_str="Noise_Rob"
            
#             python -u run.py \
#                 --is_training 1 \
#                 --root_path ./dataset/synthetic/ \
#                 --data_path Noise_Robustness_Simple_SNR_${snr}.csv \
#                 --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
#                 --evaluation_mode 'structural' \
#                 --script_name $script_name_str \
#                 --model $model_name \
#                 --loss 'VARIFOLD' \
#                 --position_kernel 'Gaussian' \
#                 --sigma_t_pos 1 \
#                 --sigma_s_pos 0.5 \
#                 --orientation_kernel $orientation_kernel \
#                 --sigma_t_or $sigma_t_or \
#                 --sigma_s_or $sigma_s_or \
#                 --train_epochs 20 \
#                 --patience 5 \
#                 --data custom \
#                 --features S \
#                 --target value \
#                 --seq_len 96 \
#                 --pred_len 96 \
#                 --enc_in 1 \
#                 --des 'Exp' \
#                 --d_model 1024 \
#                 --batch_size 4 \
#                 --learning_rate 0.0001 \
#                 --itr 1
#         done
#     done

# done








# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="Rob_Simple_MLP_DILATE"
    
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
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

#     script_name_str="Rob_Simple_MLP_MSE"
    
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
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

# done





# script_name_str="Rob_LinTrend_MLP_DILATE_infty"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_infty.csv \
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
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

#     script_name_str="Rob_LinTrend_MLP_MSE_infty"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_LinTrend_SNR_infty.csv \
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
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1




# number_of_comp_list=( 2 3 )
# sigma_t_pos_values=( 1 )
# sigma_s_pos_a_values=( 0.5 0.1 1 )
# sigma_s_pos_b_values=( 2 0.4 4 )

# for components in "${number_of_comp_list[@]}"; do
#     for sigma_t_pos in "${sigma_t_pos_values[@]}"; do
#         for i in "${!sigma_s_pos_a_values[@]}"; do
#             sigma_s_pos_a=${sigma_s_pos_a_values[$i]}
#             sigma_s_pos_b=${sigma_s_pos_b_values[$i]}
            
#             script_name_str="TwoKernels"
            
#             python -u run.py \
#                 --is_training 1 \
#                 --root_path ./dataset/synthetic/ \
#                 --data_path Multi_Scale_Dataset_Components_${components}.csv \
#                 --evaluation_mode 'raw' \
#                 --script_name $script_name_str \
#                 --model $model_name \
#                 --loss 'VARIFOLD' \
#                 --number_of_kernels 2 \
#                 --position_kernel_little 'Gaussian' \
#                 --sigma_t_pos_little $sigma_t_pos \
#                 --sigma_s_pos_little $sigma_s_pos_a \
#                 --position_kernel_big 'Gaussian' \
#                 --sigma_t_pos_big $sigma_t_pos \
#                 --sigma_s_pos_big $sigma_s_pos_b \
#                 --orientation_kernel 'Distribution' \
#                 --train_epochs 20 \
#                 --patience 5 \
#                 --data custom \
#                 --features S \
#                 --target value \
#                 --seq_len 192 \
#                 --pred_len 192 \
#                 --enc_in 1 \
#                 --des 'Exp' \
#                 --d_model 1024 \
#                 --batch_size 4 \
#                 --learning_rate 0.0001 \
#                 --itr 1
#         done
#     done
# done

# number_of_comp_list=( 2 3 )

# for components in "${number_of_comp_list[@]}"; do
            
#     script_name_str="Multi_scale_components_${components}_MSE"
            
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Multi_Scale_Dataset_Components_${components}.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name_str \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 192 \
#         --pred_len 192 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1
# done


# 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

# alpha_values=( 1 0.01 0.05 )
# gamma_values=( 0.001 0.01 0.1 1 10 0.0001 )

# for gamma in "${gamma_values[@]}"
# do
#     for alpha in "${alpha_values[@]}"
#     do
#         script_name_str="Grid_search_Noise_Robust_SNR_infty_DILATE_alpha_${alpha}_gamma_${gamma}"
        
#         python -u run.py \
#             --is_training 1 \
#             --root_path ./dataset/synthetic/ \
#             --data_path Noise_Robustness_Simple_SNR_infty.csv \
#             --structural_data_path Noise_Robustness_Simple_SNR_infty.csv \
#             --evaluation_mode 'structural' \
#             --script_name $script_name_str \
#             --model $model_name \
#             --loss 'DILATE' \
#             --alpha_dilate $alpha \
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
#             --d_model 1024 \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 1

#     done
# done


# dmodel_values=( 128 256 512 1024 2048 )

# for dmodel in "${dmodel_values[@]}"
# do
    
#     script_name_str="Tuning_hyperparams_noise_robustness_Simple_struct_W_96_H_96_MLP_d_momdel_${dmodel}"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Noise_Robustness_Simple_SNR_infty.csv \
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
#         --d_model ${dmodel} \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

# done


# python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Fractal_ARatio_3dot5_FundFreq_0dot00075_FreqRatio_8_Components_4.csv \
#         --evaluation_mode 'raw' \
#         --script_name 1st_search_heat_VAR_0dot5_0dot03_MLP_Fractal_W_200_H_200_d_model_1024 \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 0.5 \
#         --sigma_s_pos 0.03 \
#         --orientation_kernel 'Distribution' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target value \
#         --seq_len 200 \
#         --pred_len 200 \
#         --enc_in 1 \
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 3



# snr_values=(5 10 15 20)

# for snr in "${snr_values[@]}"
# do
    
#     model_name_str="August_18_SNR_${snr}_Synth_3_structural_MLP_MSE_d_model_1024_B_4_lr_10e-4"
    
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
#         --enc_in 1 \
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

# done


# for snr in "${snr_values[@]}"
# do
    
#     model_name_str="August_18_SNR_${snr}_Synth_3_structural_MLP_VAR_Gaussian_1_1_1_1_d_model_1024_B_4_lr_10e-4"
    
#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/synthetic/ \
#         --data_path Synth_3_SNR_${snr}.csv \
#         --structural_data_path Synth_3_SNR_infty.csv \
#         --evaluation_mode 'structural' \
#         --model_id $model_name_str \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --or_kernel 'Gaussian' \
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
#         --enc_in 1 \
#         --des 'Exp' \
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5

# done




# # SNR datasets, MLP, Config 1

# # MLP - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_2_SNR_15.csv \
#   --model_id Synth_2_SNR_15_raw_MLP_MSE_d_model_1024_B_4_lr_10e-4 \
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
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1


# # MLP - SNR - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_2_SNR_15.csv \
#   --model_id Synth_2_SNR_15_Raw_MLP_VAR_Cauchy_Current_2_1_2_1_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD_Cauchy' \
#   --or_kernel 'Current' \
#   --sigma_t_1 2 \
#   --sigma_t_2 1 \
#   --sigma_s_1 2 \
#   --sigma_s_2 1 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1


# # MLP - SNR - VARIFOLD Sum Kernels
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_infty.csv \
#   --model_id updated_metrics_SNR_infty_MLP_VARIFOLD_sum_kernels_d_model_1024_B_4_lr_10e-4 \
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
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1




# # SNR datasets, MLP, Config 2

# # MLP - SNR - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_3_SNR_15.csv \
#   --structural_data_path Synth_3_SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id Synth_3_SNR_15_structural_MLP_MSE_d_model_1024_B_4_lr_10e-4 \
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
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1

# # MLP - SNR - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path Synth_2_SNR_15.csv \
#   --structural_data_path Synth_2_SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id Heatmap_Synth_2_SNR_15_structural_MLP_VAR_PosOnly_1dot5_1dot5_1_1_d_model_1024_B_4_lr_10e-4 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --or_kernel 'PosOnly' \
#   --sigma_t_1 1.5 \
#   --sigma_s_1 1.5 \
#   --sigma_t_2 1 \
#   --sigma_s_2 1 \
#   --train_epochs 20 \
#   --patience 5 \
#   --data custom \
#   --features S \
#   --target value \
#   --seq_len 96 \
#   --pred_len 96 \
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1


# sigma_values=(0.25 0.40 0.55 0.70 0.85 1 1.15 1.30 1.45 1.60 1.75)

# for sigma_t_1 in "${sigma_values[@]}"
# do
#     for sigma_s_1 in "${sigma_values[@]}"
#     do
        
#         sigma_t_1_name=$(echo $sigma_t_1 | tr '.' 'dot')
#         sigma_s_1_name=$(echo $sigma_s_1 | tr '.' 'dot')
        
#         model_name_str="Heatmaps_Synth_3_SNR_15_structural_MLP_VAR_PosOnly_${sigma_t_1_name}_${sigma_s_1_name}_1_1_d_model_1024_B_4_lr_10e-4"
        
#         python -u run.py \
#             --is_training 0 \
#             --root_path ./dataset/synthetic/ \
#             --data_path Synth_3_SNR_15.csv \
#             --structural_data_path Synth_3_SNR_infty.csv \
#             --evaluation_mode 'structural' \
#             --model_id $model_name_str \
#             --model $model_name \
#             --heatmaps_base_name 'Heatmaps_Synth_3_SNR_15_structural_MLP_VAR_PosOnly' \
#             --loss 'VARIFOLD' \
#             --or_kernel 'PosOnly' \
#             --sigma_t_1 $sigma_t_1 \
#             --sigma_s_1 $sigma_s_1 \
#             --sigma_t_2 1 \
#             --sigma_s_2 1 \
#             --train_epochs 20 \
#             --patience 5 \
#             --data custom \
#             --features S \
#             --target value \
#             --seq_len 96 \
#             --pred_len 96 \
#             --enc_in 1 \
#             --des 'Exp' \
#             --d_model 1024 \
#             --batch_size 4 \
#             --learning_rate 0.0001 \
#             --itr 1

#     done
# done


# # MLP - SNR - VARIFOLD sum kernels
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path SNR_15.csv \
#   --structural_data_path SNR_infty.csv \
#   --evaluation_mode 'structural' \
#   --model_id updated_metrics_SNR_15_structural_MLP_VARIFOLD_sum_kernels_d_model_1024_B_4_lr_10e-4 \
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
#   --enc_in 1 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 1










# # increase_H, MLP, Config 1

# MLP - increase_H_bis - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path increase_H_bis.csv \
#   --model_id increase_H_bis_96_MLP_MSE_d_model_1024_B_4_lr_10e-4 \
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
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5


# # MLP - increase_H_bis - VARIFOLD_1st_version
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path increase_H_bis.csv \
#   --model_id increase_H_bis_96_MLP_VARIFOLD_1st_version_d_model_1024_B_4_lr_10e-4 \
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
#   --d_model 1024 \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --itr 5












# # MLP - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id simple_model_MLP_traffic_MSE_96_96 \
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

# # MLP - traffic - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id MLP_traffic_VARIFOLD_1_1_05_05_96_96 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 14.7 \
#   --sigma_s_2 14.7 \
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



# # MLP - ETTh1 - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id MLP_ETTh1_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1

# # MLP - ETTh1 - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id MLP_ETTh1_VARIFOLD_1_1_05_05_96_96 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 1.3 \
#   --sigma_s_2 1.3 \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1



# # MLP - exchange_rate - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id MLP_exchange_rate_MSE_96_96 \
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

# # MLP - exchange_rate - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id MLP_exchange_rate_VARIFOLD_1_1_05_05_96_96 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 1.4 \
#   --sigma_s_2 1.4 \
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






# # Choose the model
# model_name=DLinear

# # DLinear - traffic - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id DLinear_traffic_MSE_96_96 \
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

# # DLinear - traffic - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id DLinear_traffic_VARIFOLD_1_1_05_05_96_96 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 14.7 \
#   --sigma_s_2 14.7 \
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



# # DLinear - ETTh1 - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id DLinear_ETTh1_MSE_96_96 \
#   --model $model_name \
#   --loss 'MSE' \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1

# # DLinear - ETTh1 - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id DLinear_ETTh1_VARIFOLD_1_1_05_05_96_96 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 1.3 \
#   --sigma_s_2 1.3 \
#   --train_epochs 10 \
#   --patience 5 \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1



# # DLinear - exchange_rate - MSE
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id DLinear_exchange_rate_MSE_96_96 \
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

# # DLinear - exchange_rate - VARIFOLD
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id DLinear_exchange_rate_VARIFOLD_1_1_05_05_96_96 \
#   --model $model_name \
#   --loss 'VARIFOLD' \
#   --sigma_t_1 1 \
#   --sigma_t_2 1 \
#   --sigma_s_1 1.4 \
#   --sigma_s_2 1.4 \
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