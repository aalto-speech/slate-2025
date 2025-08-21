#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=output-files/out-Baseline_CCE-HF_trainer-4.out


echo "Start of Script at $(date)."

### sandi directory location
sandi="/m/triton/scratch/elec/t405-puhe/c/sandi2025"

### set up reference and tools directories
ref_dir=$sandi/reference-materials

### CTM input  directory
ctm_dir=$ref_dir/pre-norm


### Activate environment
module load mamba
###source activate /m/triton/scratch/elec/t405-puhe/p/slate-2025/envs/sandi25
source activate /m/triton/work/porwala1/conda_envs/sandi_env

echo "env activated $(date)"

epochs=7
learningRate=5e-5

### Define Work Dir
working_directory="/m/triton/work/porwala1/slate_models/Baseline_CCE-demo-DevSet-Submission"
preds_folder=$working_directory/preds

### create dir if doesn't exist.
mkdir -p $working_directory
mkdir -p $preds_folder

### where the models are saved
model_dir="/m/triton/scratch/elec/t405-puhe/p/porwala1/slate_baseline_CCE-demo-DevSet-Submission"
mkdir -p $model_dir


for part in 1 3 4 5; do
    echo "Running Model Training for part $part with learning rate $learningRate at $(date)"

    python3 -u /m/triton/work/porwala1/slate_models/HF_baseline_cce.py \
        --model_folder "${model_dir}" \
        --ctm_file_train $ctm_dir/train-sla-P${part}.ctm \
        --actual_score_file_train $ref_dir/sla-marks/train-sla-P${part}.tsv \
        --ctm_file_val $ctm_dir/dev-sla-P${part}.ctm \
        --actual_score_file_val $ref_dir/sla-marks/dev-sla-P${part}.tsv \
        --learning_rate $learningRate \
        --batch_size 8 \
        --epochs ${epochs} \
        --part ${part} \
        --save_preds_folder "${preds_folder}" \
        --wandb_project_name "Baseline_CCE-HF-P${part}-demo-DevSetSub"

    echo "Completed Model Training for part $part at $(date)"
done

echo "End of Script at $(date)."
