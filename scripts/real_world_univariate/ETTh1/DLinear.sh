#!/bin/bash

model_name=DLinear

script_name="Uni_DLinear_ETTh1"


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTh1.csv \
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
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 0.5 \
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
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 0.5 \
    --orientation_kernel 'Current' \
    --sigma_t_or 1 \
    --sigma_s_or 1 \
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
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 0.5 \
    --orientation_kernel 'UnorientedVarifold' \
    --sigma_t_or 1 \
    --sigma_s_or 1 \
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
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 0.5 \
    --orientation_kernel 'OrientedVarifold' \
    --sigma_t_or 1000 \
    --sigma_s_or 2 \
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
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5