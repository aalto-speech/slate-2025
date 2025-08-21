import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    BertConfig,
    BertModel,
    PreTrainedModel,
    BertTokenizerFast,  # or your chosen tokenizer
    EvalPrediction,
)
import numpy as np
from torch.utils.data import Dataset
from math import ceil
import wandb
import argparse
import csv
import os
from data_prep_sla_training import get_data
from pathlib import Path
from path import makeDir, checkDirExists, checkFileExists, makeCmdPath
import pandas as pd
import numpy as np
import os.path

# ===================== Custom CORN loss =====================
def corn_forward_logits(raw_logits: torch.Tensor):
    B, K1 = raw_logits.shape
    shifted_logits = torch.zeros_like(raw_logits)
    shifted_logits[:, 0] = raw_logits[:, 0]
    for k in range(1, K1):
        log_odds_prev = shifted_logits[:, k-1]
        shifted_logits[:, k] = raw_logits[:, k] + log_odds_prev
    return shifted_logits

def corn_loss_with_logits(raw_logits: torch.Tensor, labels: torch.Tensor, num_labels: int):
    """
    raw_logits: shape (B, K-1)
    labels: shape (B,) in [0..K-1]
    Return: a scalar loss

    B: Batch size
    K: number of classes
    """
    device = raw_logits.device
    K1 = num_labels - 1  # e.g. 8 classes => 7
    z = corn_forward_logits(raw_logits)  # (B, K-1)
    
    range_vec = torch.arange(K1, device=device).unsqueeze(0)  # shape (1, K-1)
    # target[i,k] = 1 if labels[i] > k else 0
    target = (labels.unsqueeze(1) > range_vec).float()
    
    bce = nn.BCEWithLogitsLoss()
    loss = bce(z, target)
    return loss

def corn_inference(raw_logits: torch.Tensor):
    """
    raw_logits: shape (B, K-1)
    Return integer predictions in [0..K].
    We'll do the same shifting, then threshold each probability at 0.5.
    """
    z = corn_forward_logits(raw_logits)  # shift
    p = torch.sigmoid(z)
    pass_mask = (p >= 0.5)
    preds = pass_mask.sum(dim=1)
    return preds



# ========================================== Model Definition ==========================================

class BERT_multiclass_CornLoss(nn.Module):
    '''
    BERT encoder, multihead attention and regression head
    '''
    def __init__(self, h1_dim=600, h2_dim=20, embedding_size=768):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

        self.attn1 = torch.nn.Linear(embedding_size, embedding_size)
        self.attn2 = torch.nn.Linear(embedding_size, embedding_size)
        self.attn3 = torch.nn.Linear(embedding_size, embedding_size)
        self.attn4 = torch.nn.Linear(embedding_size, embedding_size)

        self.layer1 = torch.nn.Linear(embedding_size*4, h1_dim)
        self.layer2 = torch.nn.Linear(h1_dim, h2_dim)
        self.layer3 = torch.nn.Linear(h2_dim, 7)


    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        '''
        input_ids: Tensor [N x L]
            Token ids

        attention_mask: Tensor [N x L]
            Utterance, token level mask

            where:
                N is batch size
                L is the maximum number of tokens in sequence (typically 512)

        '''
        output = self.encoder(input_ids, attention_mask)
        word_embeddings = output.last_hidden_state

        head1 = self._apply_attn(word_embeddings, attention_mask, self.attn1)
        head2 = self._apply_attn(word_embeddings, attention_mask, self.attn2)
        head3 = self._apply_attn(word_embeddings, attention_mask, self.attn3)
        head4 = self._apply_attn(word_embeddings, attention_mask, self.attn4)

        all_heads = torch.cat((head1, head2, head3, head4), dim=1)

        h1 = self.layer1(all_heads).clamp(min=0)
        h2 = self.layer2(h1).clamp(min=0)
        logits = self.layer3(h2).squeeze(dim=-1)

        loss = None
        if labels is not None:
            loss = corn_loss_with_logits(logits, labels, num_labels=8)

            return {
                "loss": loss,
                "logits": logits
            }

        return logits


    def _apply_attn(self, embeddings, mask, weights_transformation):
        '''
        Self-attention variant to get sentence embedding
        '''
        transformed_values = weights_transformation(embeddings)
        score = torch.einsum('ijk,ijk->ij', embeddings, transformed_values)
        T = nn.Tanh()
        score_T = T(score) * mask
        # use mask to convert padding scores to -inf (go to zero after softmax)
        mask_complement = 1 - mask
        inf_mask = mask_complement * (-10000)
        scaled_score = score_T + inf_mask
        # Normalize with softmax
        SM = nn.Softmax(dim=1)
        w = SM(scaled_score)
        repeated_w = torch.unsqueeze(w, -1).expand(-1,-1, embeddings.size(-1))
        x_attn = torch.sum(embeddings*repeated_w, dim=1)
        return x_attn


# ========================================== Metric Function For Trainer ==========================================
def compute_metrics(eval_pred: EvalPrediction):
    """
    eval_pred: NamedTuple(predictions=..., label_ids=..., ...)
    The trainer will pass in raw model outputs (logits) 
    and the ground truth labels.
    We produce a dict of metric_name -> value.
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # logits is shape (B, K-1)
    # labels is shape (B,)

    # Convert to tensor
    logits_t = torch.from_numpy(logits)
    # get predictions in [0..K-1]
    preds_t = corn_inference(logits_t)
    preds = preds_t.cpu().numpy()

    # compute accuracy
    acc = (preds == np.array(labels)).mean()
    #acc = np.sum(np.equal(preds, labels)) / len(preds)

    # compute RMSE in the original scale    
    # labels/preds are 0, 1, 2 ...something => transform back to e.g. 2..5.5 if that was your mapping
    # e.g. x/2 + 2
    org_preds = preds / 2.0 + 2
    org_labels = labels / 2.0 + 2
    rmse = np.sqrt(np.mean((org_preds - org_labels)**2))
    
    return {
        "accuracy": acc,
        "rmse": rmse
    }


# ========================================== Setting Up Trainer ==========================================
def get_tokenized_dataset(ctm_file, score_file, part):
    """
    We already have get_data returning 
     input_ids, mask, scores, submission_ids, ...
    We'll just show a minimal dataset structure for Trainer usage.
    """
    input_ids_t, masks_t, scores_t, submission_ids_l = get_data(ctm_file, score_file, part=part)
    # Convert the score to categories from regression
    # From 2, 2.5, 3,... 5.5 to 0,1,2,...7
    scores_categories_t = torch.IntTensor(scores_t.numpy()*2 - 4)

    class MyDataset(Dataset):
        def __init__(self, input_ids, masks, labels):
            self.input_ids = input_ids
            self.masks = masks
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.masks[idx],
                "labels": self.labels[idx]
            }

    ds = MyDataset(input_ids_t, masks_t, scores_categories_t)
    return ds, submission_ids_l


def main():
    #DEFINE YOUR PARSER HERE:
    parser = argparse.ArgumentParser(description="Training a baseline model with CCE Loss, using Transformer Traininer")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to save the trained model.")
    #parser.add_argument("--working_folder", type=str, required=True, help="Specify the folder where things are saved.")
    parser.add_argument("--ctm_file_train", type=str, required=True, help="CTM file with text responses to train using.")
    parser.add_argument("--actual_score_file_train", type=str, required=True, help="Path to file with the actual scores to the assessment part.")
    parser.add_argument("--ctm_file_val", type=str, required=True, help="CTM file with text responses - val set.")
    parser.add_argument("--actual_score_file_val", type=str, required=True, help="Path to file with the actual scores to the assessment part - val set.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Starting Learning Rate for training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs for training.")
    parser.add_argument("--part", type=int, default=4, help="Specify assessment part.")
    parser.add_argument("--image_folder", type=str, required=True, help="Specify the folder where all the plots will be saved.")
    parser.add_argument("--save_preds_folder", type=str, required=True, help="Specify the name with full path of the folder where the preds of the final models will be saved.")
    parser.add_argument("--wandb_project_name", type=str, required=True, help="Specify the unique name of the run so it can be logged on wandb.")
    
    args = parser.parse_args()

    wandb_run_log = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="anushaporwal-student",
            # Set the wandb project where this run will be logged.
            project=f"slate-2025",
            # Track hyperparameters and run metadata.
            config={
                "run": f"{args.wandb_project_name}",
                "architecture": "Baseline with categorial labels (not regression) using CORN Loss",
                "part": args.part,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            },
            id=f"{args.wandb_project_name}"
    )

    print("======================================================== Run Info ========================================================")
    print("Baseline with categorial labels (not regression) using CORN Loss")
    print(f"Project name    :: {args.wandb_project_name}")
    print(f"PART            :: {args.part}")
    print(f"Batch Size      :: {args.batch_size}")
    print(f"Epochs          :: {args.epochs}")
    print(f"Learning Rate   :: {args.learning_rate}")
    print(f"Model Folder    :: {args.model_folder}")
    print(f"Image Folder    :: {args.image_folder}")
    print(f"Preds Location  :: {args.save_preds_folder}")
    print("==========================================================================================================================")

    #Declare model
    model = BERT_multiclass_CornLoss()

    # Get the datasets from the ctm and score files
    train_ds, _ = get_tokenized_dataset(args.ctm_file_train, args.actual_score_file_train, args.part)
    val_ds, submission_ids_val_l = get_tokenized_dataset(args.ctm_file_val, args.actual_score_file_val, args.part)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.model_folder}/part-{args.part}/", 
        eval_strategy="steps",    # or "epochs"
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="steps",          # or "epochs"
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",   # or "eval_rmse", "accuracy", ...
        greater_is_better=False,
        report_to="wandb",
        seed=1011,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        fp16=True,
        save_total_limit=5
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=None,  # optionally pass a tokenizer if you want certain features
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # Predictions on Val set
    pred = trainer.predict(test_dataset=val_ds)
    print(pred.metrics)
    pred_logits = pred.predictions  # shape (B, K-1)
    logits_t = torch.from_numpy(pred_logits)
    preds_t = corn_inference(logits_t)
    final_preds = preds_t.cpu().numpy()

    #Convert to 2, 2.5, 3... 5.5
    final_preds_rescaled = final_preds/2 + 2


    # Save Predictions from the best model:
    actualScore = (val_ds.labels.numpy() / 2) + 2
    df = pd.DataFrame({"submissionIDs": submission_ids_val_l, "predScores": final_preds_rescaled, "actualScores": actualScore})
    df.to_csv(args.save_preds_folder + f"/val-preds-part{args.part}-epoch{args.epochs}-lr{args.learning_rate}.tsv", sep="\t", index=False)
    print("Predictions saved")
    
    #Save Model
    trainer.save_model(args.model_folder+f"/part-{args.part}/", f"baseline_CornLoss-model-P{args.part}")
    print("Model saved")

    # Save the script to W&B
    wandb.save(__file__)


if __name__ == "__main__":
    main()