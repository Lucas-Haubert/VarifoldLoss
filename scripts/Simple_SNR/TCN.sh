#!/bin/bash

model_name=TCN

snr_values=( 20 15 10 5 )

for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Simple_TCN_MSE"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Simple_SNR_${snr}.csv \
        --structural_data_path Simple_SNR_infty.csv \
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
        --out_dim_first_layer 64 \
        --e_layers 4 \
        --fixed_kernel_size_tcn 3 \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done

for snr in "${snr_values[@]}"
do
    script_name_str="Rob_Simple_TCN_DILATE"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Simple_SNR_${snr}.csv \
        --structural_data_path Simple_SNR_infty.csv \
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
        --out_dim_first_layer 64 \
        --e_layers 4 \
        --fixed_kernel_size_tcn 3 \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 5
done