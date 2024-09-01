#!/bin/bash
#SBATCH --job-name=Tuning_DILATE_Autoformer_Baseline_MSE
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

# Choose the model
model_name=Autoformer


python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path traffic.csv \
    --script_name Tuning_DILATE_Autoformer_traffic_MSE \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 168 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path electricity.csv \
    --script_name Tuning_DILATE_Autoformer_electricity_MSE \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 168 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTh1.csv \
    --script_name Tuning_DILATE_Autoformer_ETTh1_MSE \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 168 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTm1.csv \
    --script_name Tuning_DILATE_Autoformer_ETTm1_MSE \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 192 \
    --pred_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path exchange_rate.csv \
    --script_name Tuning_DILATE_Autoformer_exchange_rate_MSE \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path weather.csv \
    --script_name Tuning_DILATE_Autoformer_weather_MSE \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 144 \
    --pred_len 36 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 512 \
    --des 'Exp' \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dropout 0.1 \
    --itr 1
