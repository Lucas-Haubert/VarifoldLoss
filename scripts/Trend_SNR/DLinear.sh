#!/bin/bash

model_name=DLinear

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Trend_DLinear_MSE"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Trend_SNR_${snr}.csv \
        --structural_data_path Trend_SNR_infty.csv \
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
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done


for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Trend_DLinear_DILATE"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Trend_SNR_${snr}.csv \
        --structural_data_path Trend_SNR_infty.csv \
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
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done


gamma_values=( 0.0001 0.001 0.01 0.1 )

for gamma in "${gamma_values[@]}"
do
    for snr in "${snr_values[@]}"
    do
        script_name_str="Rob_Trend_DLinear_Soft_DTW"
        
        python -u run.py \
            --is_training 1 \
            --root_path ./dataset/synthetic/ \
            --data_path Trend_SNR_${snr}.csv \
            --structural_data_path Trend_SNR_infty.csv \
            --evaluation_mode 'structural' \
            --script_name $script_name_str \
            --model $model_name \
            --loss 'DILATE' \
            --alpha_dilate 1 \
            --gamma_dilate $gamma \
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
            --itr 5
    done
done


for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Trend_DLinear_Distrib"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Trend_SNR_${snr}.csv \
        --structural_data_path Trend_SNR_infty.csv \
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
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done


for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Trend_DLinear_Current"

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Trend_SNR_${snr}.csv \
        --structural_data_path Trend_SNR_infty.csv \
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
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done


for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Trend_DLinear_UnorVar"

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Trend_SNR_${snr}.csv \
        --structural_data_path Trend_SNR_infty.csv \
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
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done


for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Trend_DLinear_OrVar"

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Trend_SNR_${snr}.csv \
        --structural_data_path Trend_SNR_infty.csv \
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
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done

