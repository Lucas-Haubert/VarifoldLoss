#!/bin/bash
#SBATCH --job-name=TuningOnMSERealWorld
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

# # Choose the model
# model_name=SegRNN

# d_model_list=(512 1024)
# for d_model in "${d_model_list[@]}"
# do

#     script_name="TuningOnMSE_SegRNN_dmodel_${d_model}"

#     python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/univariate/ \
#         --data_path traffic.csv \
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
#         --dropout 0 \
#         --itr 1

# done






model_name=DLinear

script_name="TuningOnMSE_DLinear"

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
    --itr 1

model_name=TrendTCN

script_name="TuningOnMSE_TrendTCN_64_4_3"

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
    --seg_len 24 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --out_dim_first_layer 64 \
    --e_layers 4 \
    --fixed_kernel_size_tcn 3 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

model_name=TrendLSTM

script_name="TuningOnMSE_TrendLSTM_256_3"

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
    --seg_len 24 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 256 \
    --e_layers 3 \
    --des 'Exp' \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --dropout 0.5 \
    --itr 1

# Choose the model
model_name=SegRNN

d_model_list=(512 1024)
for d_model in "${d_model_list[@]}"
do

    script_name="TuningOnMSE_SegRNN_dmodel_${d_model}"

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
        --seg_len 24 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --d_model $d_model \
        --des 'Exp' \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --dropout 0.5 \
        --itr 1

done