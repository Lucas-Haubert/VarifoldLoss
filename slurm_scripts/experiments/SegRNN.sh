#!/bin/bash
#SBATCH --job-name=MultivariateRealWorld
#SBATCH --output=new_slurm_outputs/%x.job_%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpua100

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast


# # Choose the model
# model_name=SegRNN

# script_name="MultivariateSegRNN"

# # SegRNN Traffic Varifold

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'Current' \
#     --sigma_t_or_little 1 \
#     --sigma_s_or_little 14.1 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'Current' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1


# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'UnorientedVarifold' \
#     --sigma_t_or_little 1 \
#     --sigma_s_or_little 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'UnorientedVarifold' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'OrientedVarifold' \
#     --sigma_t_or_little 1000 \
#     --sigma_s_or_little 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'OrientedVarifold' \
#     --sigma_t_or_big 1000 \
#     --sigma_s_or_big 58.6 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1

# # SegRNN ETTh1 Varifold

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'Current' \
#     --sigma_t_or 1 \
#     --sigma_s_or 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'UnorientedVarifold' \
#     --sigma_t_or 1 \
#     --sigma_s_or 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'OrientedVarifold' \
#     --sigma_t_or 1000 \
#     --sigma_s_or 5.28 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# # SegRNN Traffic MSE

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --dropout 0 \
#     --itr 1

# # SegRNN ETTh1 MSE

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1


























# # Choose the model
# model_name=DLinear

# script_name="MultivariateDLinear"

# # DLinear Traffic Varifold

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'Current' \
#     --sigma_t_or_little 1 \
#     --sigma_s_or_little 14.1 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'Current' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'UnorientedVarifold' \
#     --sigma_t_or_little 1 \
#     --sigma_s_or_little 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'UnorientedVarifold' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'OrientedVarifold' \
#     --sigma_t_or_little 1000 \
#     --sigma_s_or_little 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'OrientedVarifold' \
#     --sigma_t_or_big 1000 \
#     --sigma_s_or_big 58.6 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# # DLinear ETTh1 Varifold

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'Current' \
#     --sigma_t_or 1 \
#     --sigma_s_or 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'UnorientedVarifold' \
#     --sigma_t_or 1 \
#     --sigma_s_or 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'OrientedVarifold' \
#     --sigma_t_or 1000 \
#     --sigma_s_or 5.28 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# # DLinear Traffic MSE

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# # DLinear ETTh1 MSE

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1
  

































# # Choose the model
# model_name=Autoformer

# script_name="MultivariateAutoformer"

# # Autoformer Traffic Varifold

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'Current' \
#     --sigma_t_or_little 1 \
#     --sigma_s_or_little 14.1 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'Current' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'UnorientedVarifold' \
#     --sigma_t_or_little 1 \
#     --sigma_s_or_little 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'UnorientedVarifold' \
#     --sigma_t_or_big 1 \
#     --sigma_s_or_big 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_little 'OrientedVarifold' \
#     --sigma_t_or_little 1000 \
#     --sigma_s_or_little 29.3 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 14.1 \
#     --orientation_kernel_big 'OrientedVarifold' \
#     --sigma_t_or_big 1000 \
#     --sigma_s_or_big 58.6 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 58.6 \
#     --weight_big 0.9 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# # Autoformer ETTh1 Varifold

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'Current' \
#     --sigma_t_or 1 \
#     --sigma_s_or 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'UnorientedVarifold' \
#     --sigma_t_or 1 \
#     --sigma_s_or 2.64 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --position_kernel 'Gaussian' \
#     --sigma_t_pos 1 \
#     --sigma_s_pos 2.64 \
#     --orientation_kernel 'OrientedVarifold' \
#     --sigma_t_or 1000 \
#     --sigma_s_or 5.28 \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# # Autoformer Traffic MSE

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# # Autoformer ETTh1 MSE

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --factor 3 \
#     --e_layers 2 \
#     --d_model 512 \
#     --d_ff 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1















# Choose the model
model_name=TimesNet

script_name="MultivariateTimesNet"

# TimesNet Traffic Varifold

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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

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
    --orientation_kernel_little 'Current' \
    --sigma_t_or_little 1 \
    --sigma_s_or_little 14.1 \
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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

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
    --orientation_kernel_little 'UnorientedVarifold' \
    --sigma_t_or_little 1 \
    --sigma_s_or_little 29.3 \
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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

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
    --orientation_kernel_little 'OrientedVarifold' \
    --sigma_t_or_little 1000 \
    --sigma_s_or_little 29.3 \
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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

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
    --sigma_t_or_big 1000 \
    --sigma_s_or_big 58.6 \
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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

# TimesNet ETTh1 Varifold

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
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

# TimesNet ETTh1 Varifold

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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

# TimesNet Traffic MSE

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
    --factor 3 \
    --e_layers 2 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0 \
    --itr 1

# TimesNet ETTh1 MSE

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
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --factor 3 \
    --e_layers 2 \
    --des 'Exp' \
    --d_model 16 \
    --d_ff 32 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1





































# script_name="Grid_Orientation"

# sigma_s_or_values=(10 5 2 1 0.5 0.25 0.1)
# for sigma_s_or in "${sigma_s_or_values[@]}"
# do

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path traffic.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --number_of_kernels 2 \
#         --position_kernel_little 'Gaussian' \
#         --sigma_t_pos_little 1 \
#         --sigma_s_pos_little 0.5 \
#         --weight_little 0.1 \
#         --position_kernel_big 'Gaussian' \
#         --sigma_t_pos_big 1 \
#         --sigma_s_pos_big 2 \
#         --weight_big 0.9 \
#         --orientation_kernel_little 'Current' \
#         --sigma_t_or_little 1 \
#         --sigma_s_or_little $sigma_s_or \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --seg_len 24 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 1024 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0 \
#         --itr 1

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path traffic.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --number_of_kernels 2 \
#         --position_kernel_little 'Gaussian' \
#         --sigma_t_pos_little 1 \
#         --sigma_s_pos_little 0.5 \
#         --weight_little 0.1 \
#         --position_kernel_big 'Gaussian' \
#         --sigma_t_pos_big 1 \
#         --sigma_s_pos_big 2 \
#         --weight_big 0.9 \
#         --orientation_kernel_little 'UnorientedVarifold' \
#         --sigma_t_or_little 1 \
#         --sigma_s_or_little $sigma_s_or \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --seg_len 24 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 1024 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0 \
#         --itr 1

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path traffic.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --number_of_kernels 2 \
#         --position_kernel_little 'Gaussian' \
#         --sigma_t_pos_little 1 \
#         --sigma_s_pos_little 0.5 \
#         --weight_little 0.1 \
#         --position_kernel_big 'Gaussian' \
#         --sigma_t_pos_big 1 \
#         --sigma_s_pos_big 2 \
#         --weight_big 0.9 \
#         --orientation_kernel_little 'OrientedVarifold' \
#         --sigma_t_or_little 1000 \
#         --sigma_s_or_little $sigma_s_or \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --seg_len 24 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 1024 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0 \
#         --itr 1

# done   









# sigma_s_or_values=(10 5 2 1 0.5 0.25 0.1)
# for sigma_s_or in "${sigma_s_or_values[@]}"
# do

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path ETTh1.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 0.5 \
#         --orientation_kernel 'Current' \
#         --sigma_t_or 1 \
#         --sigma_s_or $sigma_s_or \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --seg_len 24 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 1024 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0 \
#         --itr 1  

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path ETTh1.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 0.5 \
#         --orientation_kernel 'UnorientedVarifold' \
#         --sigma_t_or 1 \
#         --sigma_s_or $sigma_s_or \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --seg_len 24 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 1024 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0 \
#         --itr 1 

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path ETTh1.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos 0.5 \
#         --orientation_kernel 'OrientedVarifold' \
#         --sigma_t_or 1000 \
#         --sigma_s_or $sigma_s_or \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --seg_len 24 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 1024 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0 \
#         --itr 1 

# done


# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'VARIFOLD' \
#     --number_of_kernels 2 \
#     --position_kernel_little 'Gaussian' \
#     --sigma_t_pos_little 1 \
#     --sigma_s_pos_little 0.5 \
#     --weight_little 0.1 \
#     --position_kernel_big 'Gaussian' \
#     --sigma_t_pos_big 1 \
#     --sigma_s_pos_big 2 \
#     --weight_big 0.9 \
#     --orientation_kernel 'Distribution' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 1024 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1
        


















# # Choose the model
# model_name=Autoformer

# sigma_s_pos_big_list=( 2 5 10 )
# sigma_s_pos_little=( 0.1 0.5 1 )
# weight_little_list=( 0.05 0.1 0.25 0.5 0.75 0.9 0.95 )

# for sigma_s_pos_big in "${sigma_s_pos_big_list[@]}"
# do

#     for sigma_s_pos_little in "${sigma_s_pos_little[@]}"
#     do

#         for weight_little in "${weight_little_list[@]}"
#         do

#             script_name="Grid_TwoKernels"

#             python -u run.py \
#                 --is_training 1 \
#                 --root_path ./dataset/univariate/ \
#                 --data_path traffic.csv \
#                 --evaluation_mode 'raw' \
#                 --script_name $script_name \
#                 --model $model_name \
#                 --loss 'VARIFOLD' \
#                 --number_of_kernels 2 \
#                 --position_kernel_little 'Gaussian' \
#                 --sigma_t_pos_little 1 \
#                 --sigma_s_pos_little $sigma_s_pos_little \
#                 --weight_little $weight_little \
#                 --position_kernel_big 'Gaussian' \
#                 --sigma_t_pos_big 1 \
#                 --sigma_s_pos_big $sigma_s_pos_big \
#                 --weight_big $(echo "1 - $weight_little" | bc) \
#                 --orientation_kernel 'Distribution' \
#                 --train_epochs 20 \
#                 --patience 5 \
#                 --data custom \
#                 --features S \
#                 --target '0' \
#                 --seq_len 96 \
#                 --pred_len 96 \
#                 --factor 3 \
#                 --e_layers 2 \
#                 --enc_in 1 \
#                 --dec_in 1 \
#                 --c_out 1 \
#                 --d_model 512 \
#                 --d_ff 512 \
#                 --des 'Exp' \
#                 --batch_size 4 \
#                 --learning_rate 0.0001 \
#                 --dropout 0 \
#                 --itr 1

#             python -u run.py \
#                 --is_training 1 \
#                 --root_path ./dataset/univariate/ \
#                 --data_path ETTh1.csv \
#                 --evaluation_mode 'raw' \
#                 --script_name $script_name \
#                 --model $model_name \
#                 --loss 'VARIFOLD' \
#                 --number_of_kernels 2 \
#                 --position_kernel_little 'Gaussian' \
#                 --sigma_t_pos_little 1 \
#                 --sigma_s_pos_little $sigma_s_pos_little \
#                 --weight_little $weight_little \
#                 --position_kernel_big 'Gaussian' \
#                 --sigma_t_pos_big 1 \
#                 --sigma_s_pos_big $sigma_s_pos_big \
#                 --weight_big $(echo "1 - $weight_little" | bc) \
#                 --orientation_kernel 'Distribution' \
#                 --train_epochs 20 \
#                 --patience 5 \
#                 --data custom \
#                 --features S \
#                 --target '0' \
#                 --seq_len 96 \
#                 --pred_len 96 \
#                 --factor 3 \
#                 --e_layers 2 \
#                 --enc_in 1 \
#                 --dec_in 1 \
#                 --c_out 1 \
#                 --d_model 512 \
#                 --d_ff 512 \
#                 --des 'Exp' \
#                 --batch_size 4 \
#                 --learning_rate 0.0001 \
#                 --dropout 0 \
#                 --itr 1
#         done
#     done
# done



# # Choose the model
# model_name=Autoformer

# sigma_s_pos_list=( 0.1 0.25 0.5 1 2 5 10 )
# for sigma_s_pos in "${sigma_s_pos_list[@]}"
# do

#     script_name="Grid_OneKernel"

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path traffic.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos $sigma_s_pos \
#         --orientation_kernel 'Distribution' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --factor 3 \
#         --e_layers 2 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 512 \
#         --d_ff 512 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0 \
#         --itr 1

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path ETTh1.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'VARIFOLD' \
#         --position_kernel 'Gaussian' \
#         --sigma_t_pos 1 \
#         --sigma_s_pos $sigma_s_pos \
#         --orientation_kernel 'Distribution' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --factor 3 \
#         --e_layers 2 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model 512 \
#         --d_ff 512 \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --itr 1

# done



















































































# model_name=DLinear

# script_name="TuningOnMSE_DLinear"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# model_name=TrendTCN

# script_name="TuningOnMSE_TrendTCN_64_4_3"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --out_dim_first_layer 64 \
#     --e_layers 4 \
#     --fixed_kernel_size_tcn 3 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1

# model_name=TrendLSTM

# script_name="TuningOnMSE_TrendLSTM_256_3"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path traffic.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 256 \
#     --e_layers 3 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --dropout 0 \
#     --itr 1







# model_name=DLinear

# script_name="TuningOnMSE_DLinear"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# model_name=TrendTCN

# script_name="TuningOnMSE_TrendTCN_64_4_3"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --out_dim_first_layer 64 \
#     --e_layers 4 \
#     --fixed_kernel_size_tcn 3 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# model_name=TrendLSTM

# script_name="TuningOnMSE_TrendLSTM_256_3"

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path ETTh1.csv \
#     --evaluation_mode 'raw' \
#     --script_name $script_name \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 96 \
#     --seg_len 24 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 256 \
#     --e_layers 3 \
#     --des 'Exp' \
#     --batch_size 4 \
#     --learning_rate 0.0001 \
#     --dropout 0.5 \
#     --itr 1

# # Choose the model
# model_name=SegRNN

# d_model_list=(512 1024)
# for d_model in "${d_model_list[@]}"
# do

#     script_name="TuningOnMSE_SegRNN_dmodel_${d_model}"

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path ETTh1.csv \
#         --evaluation_mode 'raw' \
#         --script_name $script_name \
#         --model $model_name \
#         --loss 'MSE' \
#         --train_epochs 20 \
#         --patience 5 \
#         --data custom \
#         --features S \
#         --target '0' \
#         --seq_len 96 \
#         --pred_len 96 \
#         --seg_len 24 \
#         --enc_in 1 \
#         --dec_in 1 \
#         --c_out 1 \
#         --d_model $d_model \
#         --des 'Exp' \
#         --batch_size 4 \
#         --learning_rate 0.0001 \
#         --dropout 0.5 \
#         --itr 1

# done