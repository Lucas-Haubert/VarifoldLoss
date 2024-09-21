#!/bin/bash

model_name=SegRNN

script_name="Multi_SegRNN_Traffic"


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 14.1 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 58.6 \
    --weight_big 0.9 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 14.1 \
    --orientation_kernel_big 'Current' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 14.1 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 58.6 \
    --weight_big 0.9 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 14.1 \
    --orientation_kernel_big 'UnorientedVarifold' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 14.1 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 58.6 \
    --weight_big 0.9 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 14.1 \
    --orientation_kernel_big 'OrientedVarifold' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 29.3 \
    --weight_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 58.6 \
    --weight_big 0.9 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.001 \
    --dropout 0 \
    --itr 5