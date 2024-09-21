#!/bin/bash

model_name=TimesNet

script_name="Uni_TimesNet_Traffic"


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.5 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 2 \
    --weight_big 0.9 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.5 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 2 \
    --weight_big 0.9 \
    --orientation_kernel_big 'Current' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 1 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.5 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 2 \
    --weight_big 0.9 \
    --orientation_kernel_big 'UnorientedVarifold' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 1 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.5 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 2 \
    --weight_big 0.9 \
    --orientation_kernel_big 'OrientedVarifold' \
    --sigma_t_or_big 1000 \
    --sigma_s_or_big 2 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5