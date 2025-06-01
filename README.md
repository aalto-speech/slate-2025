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
Coming soon.

---

# Citation
Coming soon.

# License
Our work is shared under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)