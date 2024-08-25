#!/bin/bash
#SBATCH --job-name=Tuning_hyperparams_Fractal_TCN_on_MSE
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
model_name=TCN


hyperparams_list=(
    "32 2 3"
    "32 3 3"
    "32 4 3"
    "32 5 3"
    "64 2 3"
    "64 3 3"
    "64 4 3"
    "64 5 3"
    "32 2 5"
    "32 3 5"
    "32 4 5"
    "32 5 5"
    "64 2 5"
    "64 3 5"
    "64 4 5"
    "64 5 5"
)

for hyperparam in "${hyperparams_list[@]}"
do

    outfirstdimlay=$(echo $hyperparam | cut -d' ' -f1)
    elayers=$(echo $hyperparam | cut -d' ' -f2)
    kernelsize=$(echo $hyperparam | cut -d' ' -f3)
    
    outfirstdimlay_str=$(echo $outfirstdimlay | sed 's/\./dot/g')
    elayers_str=$(echo $elayers | sed 's/\./dot/g')
    kernelsize_str=$(echo $kernelsize | sed 's/\./dot/g')
    
    script_name_str="Tuning_hyperparams_Fractal_TCN_on_MSE_${outfirstdimlay_str}_${elayers_str}_${kernelsize_str}"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/synthetic/ \
        --data_path Fractal_Config_1_Components_4.csv \
        --evaluation_mode 'raw' \
        --script_name $script_name_str \
        --model $model_name \
        --loss 'MSE' \
        --train_epochs 20 \
        --patience 5 \
        --data custom \
        --features S \
        --target value \
        --seq_len 2000 \
        --pred_len 2000 \
        --enc_in 1 \
        --des 'Exp' \
        --out_dim_first_layer ${outfirstdimlay} \
        --e_layers ${elayers} \
        --fixed_kernel_size_tcn ${kernelsize} \
        --batch_size 4 \
        --learning_rate 0.0001 \
        --itr 1

done