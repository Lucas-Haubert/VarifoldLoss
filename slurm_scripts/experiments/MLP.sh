#!/bin/bash
#SBATCH --job-name=VARIFOLDNoiseRobSimple
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

    script_name_str="Noise_Rob_${snr}"
    
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
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

done






# Choose the model
model_name=LSTM

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do

    script_name_str="Noise_Rob_${snr}"
    
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
        --d_model 512 \
        --e_layers 1 \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

done






# Choose the model
model_name=TCN

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do

    script_name_str="Noise_Rob_${snr}"
    
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
        --out_dim_first_layer 64 \
        --e_layers 4 \
        --fixed_kernel_size_tcn 3 \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

done














































































































# sigma_t_pos_values=(5 10 50 100)
# sigma_s_pos_values=(0.05 0.1 0.25 0.5 1 2 5 10 25 50 100 200 500)

# for sigma_t_pos in "${sigma_t_pos_values[@]}"
# do
#     for sigma_s_pos in "${sigma_s_pos_values[@]}"
#     do
#         script_name_str="Heatmap_${sigma_t_pos}_${sigma_s_pos}"
        
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
#         --itr 1

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
#         --itr 1

# done






# script_name_str="FractalAccurary_1_05"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_Config_2_Components_3.csv \
#     --structural_data_path Fractal_Config_2_Components_3.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 0.5 \
#     --orientation_kernel 'Distribution' \
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
#     --itr 5

# script_name_str="FractalAccurary_1_1"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_Config_2_Components_3.csv \
#     --structural_data_path Fractal_Config_2_Components_3.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 1 \
#     --orientation_kernel 'Distribution' \
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
#     --itr 5

# script_name_str="FractalCut_f2_10_5"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_Config_2_Components_3.csv \
#     --structural_data_path Fractal_Config_2_Components_2.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 10 \
#     --sigma_s_pos 5 \
#     --orientation_kernel 'Distribution' \
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
#     --itr 5

# script_name_str="FractalCut_f2_10_10"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_Config_2_Components_3.csv \
#     --structural_data_path Fractal_Config_2_Components_2.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 10 \
#     --sigma_s_pos 10 \
#     --orientation_kernel 'Distribution' \
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
#     --itr 5

# script_name_str="FractalCut_f1_100_50"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_Config_2_Components_3.csv \
#     --structural_data_path Fractal_Config_2_Components_1.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 100 \
#     --sigma_s_pos 50 \
#     --orientation_kernel 'Distribution' \
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
#     --itr 5

# script_name_str="FractalCut_f1_100_100"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/synthetic/ \
#     --data_path Fractal_Config_2_Components_3.csv \
#     --structural_data_path Fractal_Config_2_Components_1.csv \
#     --evaluation_mode 'structural' \
#     --script_name $script_name_str \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 100 \
#     --sigma_s_pos 100 \
#     --orientation_kernel 'Distribution' \
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
#     --itr 5
    



# snr_values=( 20 15 10 5 )

# for snr in "${snr_values[@]}"
# do
#     script_name_str="RobSimpleMLPVarifold_${snr}"
    
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
#         --d_model 1024 \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 5
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

