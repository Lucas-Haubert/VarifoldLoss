#!/bin/bash

model_name=MLP

sigma_t_pos_values=(0.05 0.5 1 5 10 50 100)
sigma_s_pos_values=(0.05 0.1 0.25 0.5 1 2 5 10 25 50 100 200 500)

for sigma_t_pos in "${sigma_t_pos_values[@]}"
do
    for sigma_s_pos in "${sigma_s_pos_values[@]}"
    do
        script_name_str="Heatmap_${sigma_t_pos}_${sigma_s_pos}"
        
        python -u run.py \
            --is_training 1 \
            --root_path ./dataset/synthetic/ \
            --data_path Fractal_2.csv \
            --structural_data_path Fractal_2.csv \
            --evaluation_mode 'structural' \
            --script_name $script_name_str \
            --model $model_name \
            --loss 'VARIFOLD' \
            --position_kernel 'Gaussian' \
            --sigma_t_pos $sigma_t_pos \
            --sigma_s_pos $sigma_s_pos \
            --orientation_kernel 'Distribution' \
            --train_epochs 20 \
            --patience 5 \
            --data custom \
            --features S \
            --target value \
            --seq_len 336 \
            --pred_len 336 \
            --enc_in 1 \
            --des 'Exp' \
            --d_model 4096 \
            --batch_size 4 \
            --learning_rate 0.0001 \
            --itr 5
    done
done