#!/bin/bash
#SBATCH --job-name=TuningOnMSERealWorldAutoformer
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

script_name_str="TuningOnMSE_Autoformer_ETTh1_d_model_16_d_ff_32"
    
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/univariate/ \
    --data_path ETTh1.csv \
    --script_name $script_name_str \
    --model $model_name \
    --loss 'MSE' \
    --train_epochs 20 \
    --patience 5 \
    --data custom \
    --features S \
    --target '0' \
    --seq_len 96 \
    --pred_len 96 \
    --factor 3 \
    --e_layers 2 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --d_model 16 \
    --d_ff 32 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --itr 1

d_model_list=(32 256 512)
d_ff_list=(64 256 512)

for i in "${!d_model_list[@]}"
do
    d_model=${d_model_list[$i]}
    d_ff=${d_ff_list[$i]}

    script_name_str="TuningOnMSE_Autoformer_traffic_d_model_${d_model}_d_ff_${d_ff}"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/univariate/ \
        --data_path traffic.csv \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'MSE' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target '0' \
        --seq_len 96 \
        --pred_len 96 \
        --factor 3 \
        --e_layers 2 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

    script_name_str="TuningOnMSE_Autoformer_ETTh1_d_model_${d_model}_d_ff_${d_ff}"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/univariate/ \
        --data_path ETTh1.csv \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'MSE' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target '0' \
        --seq_len 96 \
        --pred_len 96 \
        --factor 3 \
        --e_layers 2 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1
done






# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path traffic.csv \
#     --script_name Tuning_DILATE_Autoformer_traffic_MSE \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 168 \
#     --pred_len 24 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path electricity.csv \
#     --script_name Tuning_DILATE_Autoformer_electricity_MSE \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 168 \
#     --pred_len 24 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path ETTh1.csv \
#     --script_name Tuning_DILATE_Autoformer_ETTh1_MSE \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 168 \
#     --pred_len 24 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path ETTm1.csv \
#     --script_name Tuning_DILATE_Autoformer_ETTm1_MSE \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 192 \
#     --pred_len 48 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path exchange_rate.csv \
#     --script_name Tuning_DILATE_Autoformer_exchange_rate_MSE \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 96 \
#     --pred_len 24 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/univariate/ \
#     --data_path weather.csv \
#     --script_name Tuning_DILATE_Autoformer_weather_MSE \
#     --model $model_name \
#     --loss 'MSE' \
#     --train_epochs 20 \
#     --patience 5 \
#     --data custom \
#     --features S \
#     --target '0' \
#     --seq_len 144 \
#     --pred_len 36 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 1 \
#     --dec_in 1 \
#     --c_out 1 \
#     --d_model 512 \
#     --des 'Exp' \
#     --batch_size 32 \
#     --learning_rate 0.0001 \
#     --dropout 0.1 \
#     --itr 1
