#!/bin/bash
#SBATCH --job-name=BigExperiment
#SBATCH --output=slurm_outputs/%x.job_%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpup100

# Module load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate flexforecast







# Transformers

#      ECL


# Transformer - electricity - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Transformer_electricity_MSE_96_96 \
  --model Transformer \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1

# Transformer - electricity - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Transformer_electricity_DILATE_96_96 \
  --model Transformer \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1

# Transformer - electricity - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Transformer_electricity_TILDEQ_96_96 \
  --model Transformer \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1


#      ETTh1


# Transformer - ETTh1 - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id Transformer_ETTh1_MSE_96_96 \
  --model Transformer \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

# Transformer - ETTh1 - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id Transformer_ETTh1_DILATE_96_96 \
  --model Transformer \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

# Transformer - ETTh1 - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id Transformer_ETTh1_TILDEQ_96_96 \
  --model Transformer \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1


#      Traffic


# Transformer - traffic - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Transformer_traffic_MSE_96_96 \
  --model Transformer \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

# Transformer - traffic - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Transformer_traffic_DILATE_96_96 \
  --model Transformer \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

# Transformer - traffic - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Transformer_traffic_TILDEQ_96_96 \
  --model Transformer \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1













# DLinear

#      ECL


# DLinear - electricity - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id DLinear_electricity_MSE_96_96 \
  --model DLinear \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1

# DLinear - electricity - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id DLinear_electricity_DILATE_96_96 \
  --model DLinear \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1

# DLinear - electricity - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id DLinear_electricity_TILDEQ_96_96 \
  --model DLinear \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1


#      ETTh1


# DLinear - ETTh1 - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id DLinear_ETTh1_MSE_96_96 \
  --model DLinear \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1

# DLinear - ETTh1 - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id DLinear_ETTh1_DILATE_96_96 \
  --model DLinear \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1

# DLinear - ETTh1 - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id DLinear_ETTh1_TILDEQ_96_96 \
  --model DLinear \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1


#      Traffic


# DLinear - traffic - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id DLinear_traffic_MSE_96_96 \
  --model DLinear \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

# DLinear - traffic - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id DLinear_traffic_DILATE_96_96 \
  --model DLinear \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

# DLinear - traffic - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id DLinear_traffic_TILDEQ_96_96 \
  --model DLinear \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1










# TimesNet

#      ECL


# TimesNet - electricity - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id TimesNet_electricity_MSE_96_96 \
  --model TimesNet \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --itr 1

# TimesNet - electricity - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id TimesNet_electricity_DILATE_96_96 \
  --model TimesNet \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --itr 1

# TimesNet - electricity - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id TimesNet_electricity_TILDEQ_96_96 \
  --model TimesNet \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --itr 1


#      ETTh1


# TimesNet - ETTh1 - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id TimesNet_ETTh1_MSE_96_96 \
  --model TimesNet \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --itr 1

# TimesNet - ETTh1 - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id TimesNet_ETTh1_DILATE_96_96 \
  --model TimesNet \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --itr 1

# TimesNet - ETTh1 - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id TimesNet_ETTh1_TILDEQ_96_96 \
  --model TimesNet \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --itr 1


#      Traffic


# TimesNet - traffic - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id TimesNet_traffic_MSE_96_96 \
  --model TimesNet \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --itr 1

# TimesNet - traffic - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id TimesNet_traffic_DILATE_96_96 \
  --model TimesNet \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --itr 1

# TimesNet - traffic - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id TimesNet_traffic_TILDEQ_96_96 \
  --model TimesNet \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --itr 1










# SegRNN

#      ECL


# SegRNN - electricity - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id SegRNN_electricity_MSE_96_96 \
  --model SegRNN \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0 \
  --learning_rate 0.001 \
  --itr 1

# SegRNN - electricity - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id SegRNN_electricity_DILATE_96_96 \
  --model SegRNN \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0 \
  --learning_rate 0.001 \
  --itr 1

# SegRNN - electricity - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id SegRNN_electricity_TILDEQ_96_96 \
  --model SegRNN \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seg_len 24 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --dropout 0 \
  --learning_rate 0.001 \
  --itr 1


#      ETTh1


# SegRNN - ETTh1 - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id SegRNN_ETTh1_MSE_96_96 \
  --model SegRNN \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --itr 1

# SegRNN - ETTh1 - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id SegRNN_ETTh1_DILATE_96_96 \
  --model SegRNN \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --itr 1

# SegRNN - ETTh1 - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id SegRNN_ETTh1_TILDEQ_96_96 \
  --model SegRNN \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --itr 1


#      Traffic


# SegRNN - traffic - MSE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id SegRNN_traffic_MSE_96_96 \
  --model SegRNN \
  --loss 'MSE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --itr 1

# SegRNN - traffic - DILATE
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id SegRNN_traffic_DILATE_96_96 \
  --model SegRNN \
  --loss 'DILATE' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --itr 1

# SegRNN - traffic - TILDEQ
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id SegRNN_traffic_TILDEQ_96_96 \
  --model SegRNN \
  --loss 'TILDEQ' \
  --train_epochs 10 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --itr 1