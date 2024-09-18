#!/bin/bash

model_name=TrendLSTM

script_name_str="Multi_Scale_TrendLSTM_MSE"


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Multi_Scale.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


script_name_str="Multi_Scale_TrendLSTM_Little"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Multi_Scale.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 0.1 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


script_name_str="Multi_Scale_TrendLSTM_Big"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Multi_Scale.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --position_kernel 'Gaussian' \
    --sigma_t_pos 1 \
    --sigma_s_pos 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


script_name_str="Multi_Scale_TrendLSTM_Sum_Distrib"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Multi_Scale.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


script_name_str="Multi_Scale_TrendLSTM_Sum_Current"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Multi_Scale.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'Current' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


script_name_str="Multi_Scale_TrendLSTM_Sum_UnorVar"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Multi_Scale.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'UnorientedVarifold' \
    --sigma_t_or_big 1 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5


script_name_str="Multi_Scale_TrendLSTM_Sum_OrVar"

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/synthetic/ \
    --data_path Multi_Scale.csv \
    --evaluation_mode 'raw' \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'VARIFOLD' \
    --number_of_kernels 2 \
    --position_kernel_little 'Gaussian' \
    --weight_little 0.02 \
    --sigma_t_pos_little 1 \
    --sigma_s_pos_little 0.1 \
    --position_kernel_big 'Gaussian' \
    --weight_big 0.98 \
    --sigma_t_pos_big 1 \
    --sigma_s_pos_big 5 \
    --orientation_kernel_big 'OrientedVarifold' \
    --sigma_t_or_big 1000 \
    --sigma_s_or_big 5 \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target value \
    --seq_len 192 \
    --pred_len 192 \
    --enc_in 1 \
    --d_model 2048 \
    --e_layers 1 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 5