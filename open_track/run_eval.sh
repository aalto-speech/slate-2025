#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=4
#SBATCH --time=3-23:00:00
#SBATCH -J eval_slate25
#SBATCH --output=logs/eval_avg_15.out

source activate ENV_DIR

python model_evaluation_avg.py \
     --model_path SAFETENSORS_DIR \
     --dataset_path TEST_SET_DIR_TSV \
     --save_name "avg_15" \
     --batch_size=1

# python model_evaluation_tf.py \
#      --model_path SAFETENSORS_DIR \
#      --dataset_path TEST_SET_DIR_TSV \
#      --save_name "tf_15" \
#      --batch_size=1     