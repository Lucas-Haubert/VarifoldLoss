#!/bin/bash

model_name=SegRNN

script_name="Multi_SegRNN_ETTh1"


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
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
    --seg_len 24 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 2.64 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --seg_len 24 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 2.64 \
    --orientation_kernel 'Current' \
    --sigma_t_or 1 \
    --sigma_s_or 2.64 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 2.64 \
    --orientation_kernel 'UnorientedVarifold' \
    --sigma_t_or 1 \
    --sigma_s_or 2.64 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 2.64 \
    --orientation_kernel 'OrientedVarifold' \
    --sigma_t_or 1000 \
    --sigma_s_or 5.28 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 1024 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 5

