#!/bin/bash
####SBATCH --gres=gpu:1
####SBATCH --partition=gpu-v100-32g
####SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=output-files/out-InferenceEval-CornLoss-TrainData-withProbs.out


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

### Define Work Dir
preds_dir="/m/triton/work/porwala1/slate_models/CornLoss_TrainDataModel_EvalPreds_withProbs"
### create dir if doesn't exist.
mkdir -p $preds_dir

### where the models are saved
model_dir="/m/triton/scratch/elec/t405-puhe/p/porwala1/slate_baseline_CornLoss-TrainData_Day2hp-model"
####mkdir -p $model_dir
###/part-1/checkpoint-700


echo "Running Model Inferencing at $(date)"

part=1
checkpoint=700
echo "Running Model Inferencing for part $part at $(date)"
python3 -u /m/triton/work/porwala1/slate_models/Inference_Eval_Data_CornLoss_withProb.py \
    --model_path $model_dir/part-${part}/checkpoint-${checkpoint} \
    --save_preds_folder "${preds_dir}" \
    --part $part

part=3
checkpoint=450
echo "Running Model Inferencing for part $part at $(date)"
python3 -u /m/triton/work/porwala1/slate_models/Inference_Eval_Data_CornLoss_withProb.py \
    --model_path $model_dir/part-${part}/checkpoint-${checkpoint} \
    --save_preds_folder "${preds_dir}" \
    --part $part

part=4
checkpoint=1200
echo "Running Model Inferencing for part $part at $(date)"
python3 -u /m/triton/work/porwala1/slate_models/Inference_Eval_Data_CornLoss_withProb.py \
    --model_path $model_dir/part-${part}/checkpoint-${checkpoint} \
    --save_preds_folder "${preds_dir}" \
    --part $part

part=5
checkpoint=2300
echo "Running Model Inferencing for part $part at $(date)"
python3 -u /m/triton/work/porwala1/slate_models/Inference_Eval_Data_CornLoss_withProb.py \
    --model_path $model_dir/part-${part}/checkpoint-${checkpoint} \
    --save_preds_folder "${preds_dir}" \
    --part $part


echo "End of Script at $(date)."
