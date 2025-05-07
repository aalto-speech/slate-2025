#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-80g,gpu-h100-80g
#SBATCH --mem=160G
#SBATCH --cpus-per-task=4
#SBATCH --time=3-23:00:00
#SBATCH -J train_slate25
#SBATCH --output=logs/model_train.out

source activate ENV_DIR

# # # Define run_name for wandb
run_name="avg_15"

python model_avg.py \
    --train_tsv TRAIN_SET_DIR_TSV \
    --dev_tsv DEV_SET_DIR_TSV \
    --epochs="15" \
    --batch_size="10" \
    --learning_rate="5e-5" \
    --sample_rate="16000" \
    --output_dir = "MODEL_OUTPUT_DIR${run_name}" \
    --run_name="${run_name}" 
    
# # Define run_name
# run_name="model_tf_15"

# python model_tf.py \
#     --train_tsv TRAIN_SET_DIR_TSV \
#     --dev_tsv DEV_SET_DIR_TSV \
#     --epochs="15" \
#     --batch_size="10" \
#     --learning_rate="5e-5" \
#     --sample_rate="16000" \
#     --output_dir = "MODEL_OUTPUT_DIR${run_name}" \
#     --run_name="${run_name}"    
