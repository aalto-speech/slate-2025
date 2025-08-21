# Speak & Improve Challange 2025: Spoken Language Assessment
This repository contains our model architectures and scripts for the **Speak & Improve Challenge 2025: Spoken Language Assessment (SLA)**.

---

## Open Track

All files related to the Open Track are located in the [`open_track`](open_track) folder.

| File | Purpose |
|------|---------|
| `model_avg.py` | Model architecture using average pooling for aggregation |
| `model_tf.py` | Model architecture using Transformer-based aggregation |
| `environment.yaml` | Conda environment with for Open Track paper |
| `run_train_model.sh` | Shell script to train the model |
| `run_eval.sh` | Shell script to evaluate the trained model |

To install:
```bash
cd open_track
conda env create -n slate2025 -f environment.yaml
```


---

## Close Track
| File | Purpose |
|------|---------|
| `HF_baseline_cce.py` | Replaces RMSE with CCE Loss in the provided baseline model |
| `HF_baseline_CornLoss.py` | Replaces RMSE with CORN Loss in the provided baseline model |
| `Inference_Eval_Data_CornLoss_withProb.py` | Inferences the Eval Set on the CORN Loss model and creates a csv with probabilities for each class |
| `ThresholdTuning_calibration.ipynb` | Performs Threshold Tuning/Posterior Calibration on the Dev set and saves the best thresholds in a csv |
| `EvalFinalScores-Calibration.py` | Calculates the final predictions based on new thresholds provided |
| `env.yaml` | Conda environment with for Closed Track paper |
| `run-scripts/run-HF_baseline_CCE.sh` | Shell script to train the CCE model |
| `run-scripts/run-HF_baseline_CornLoss.sh` | Shell script to train the CORN Loss model |
| `run-scripts/run-Inference_Eval_CornLoss_TrainDataModel_withProbs.sh` | Shell script to evaluate the CORN Loss model |

---

# Citation
Coming soon.

# License
Our work is shared under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)
