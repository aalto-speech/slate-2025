import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    BertModel,
    EvalPrediction,
)
import numpy as np
from torch.utils.data import Dataset
from math import ceil
#from data_prep_sla_training import get_data, get_submission_to_utt, get_score, tokenize_text
from pathlib import Path
#from path import makeDir, checkDirExists, checkFileExists, makeCmdPath
import pandas as pd
import numpy as np
import os.path
from safetensors.torch import load_file
import argparse
from data_prep_sla_training import get_data, get_submission_to_utt, get_score, tokenize_text




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
    Return integer predictions in [0..K]. AND the probs of each class.
    We'll do the same shifting, then threshold each probability at 0.5.
    """
    z = corn_forward_logits(raw_logits)  # shift
    p = torch.sigmoid(z)
    pass_mask = (p >= 0.5)
    preds = pass_mask.sum(dim=1)
    return preds, p

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
    

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def main():
    #DEFINE YOUR PARSER HERE:
    parser = argparse.ArgumentParser(description="Inferencing a baseline model with Corn Loss, using Transformer Traininer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--save_preds_folder", type=str, required=True, help="Specify the name with full path of the folder where the preds of the final models will be saved.")
    parser.add_argument("--part", type=int, required=True, help="Task part")
    #parser.add_argument("--ckpt_list_end", type=list_of_ints, required=False, help="add the checkpoint numbers for the final model")
    args = parser.parse_args()

    print("entered code")
    print(f"Folder: {args.save_preds_folder}")

    #Inferencing middle checkpoint:
    #parts = [1,3,4,5]
    model = BERT_multiclass_CornLoss()
    model.load_state_dict(load_file(args.model_path + f"/model.safetensors"))
    model.eval()

    # Get eval data
    df = pd.read_csv(f"/m/triton/work/porwala1/slate_models/evalTranscripts/eval-data-P{args.part}.csv")
    submission_ids_list_tr = df["submissionID"].tolist()
    text_list_tr = df["text"].tolist()
    #trainData_df = pd.DataFrame({"submission_id": submission_ids_list_tr, "text": text_list_tr})

    # Get input ID tensors and mask tensors
    input_ids_t, masks_t = tokenize_text(text_list_tr)

    class MyDataset(Dataset):
        def __init__(self, input_ids, masks):
            self.input_ids = input_ids
            self.masks = masks
            
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.masks[idx]
            }

    ds = MyDataset(input_ids_t, masks_t)

    pred_args = TrainingArguments(
            output_dir="pred-output-CCE", 
            per_device_eval_batch_size=8,
            do_eval=False,       # we are not computing metrics
            do_predict=True,     # we only want predictions
        )

    trainer = Trainer(
            model=model,
            args=pred_args
        )
    

    predictions_output = trainer.predict(test_dataset=ds)
    print(f"Preds complete for part {args.part}")
    pred_logits = predictions_output.predictions
    logits_t = torch.from_numpy(pred_logits)
    preds_t, probs_t = corn_inference(logits_t)
    final_preds = preds_t.cpu().numpy()
    final_preds_rescaled = final_preds/2 + 2

    df = pd.DataFrame(probs_t.numpy(), columns=[f'bin_{j}' for j in range(probs_t.shape[1])])
    df.insert(0, "score", final_preds_rescaled)
    df.insert(0, "submissionID", submission_ids_list_tr)
    #df = pd.DataFrame({"submissionID": submission_ids_list_tr, "score": final_preds_rescaled})
    df.to_csv(f"{args.save_preds_folder}/eval-preds-P{args.part}-withProb.csv", index=False)
    print(f"Preds saved for part {args.part}")


if __name__ == "__main__":
    main()