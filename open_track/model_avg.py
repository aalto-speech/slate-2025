import argparse
import os
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

from transformers.trainer_utils import get_last_checkpoint

# Using W&B (https://wandb.ai/) for logging
# This is optional and can be removed if not needed
import wandb

PROJECT_NAME = "slate-2025-submission"
os.environ["WANDB_PROJECT"]= PROJECT_NAME

class MultiAudioDataset(Dataset):
    def __init__(self, csv_path):
        # Load metadata (expects columns: part1_path, part3_path, part4_path, part5_path, and score_norm)
        self.data = pd.read_csv(csv_path, sep='\t')
        self.sample_rate = 16000  # Whisper expects 16kHz audio
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Collect the file paths for the four parts
        audio_paths = [
            row["part1_merged_path"],
            row["part3_path"],
            row["part4_path"],
            row["part5_merged_path"]
        ]
        # Load each audio file and convert to float32
        audio_parts = []
        for audio_path in audio_paths:
            # Load audio using torchaudio
            wave, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                wave = torchaudio.functional.resample(wave, sr, self.sample_rate)

            # Convert stereo/multi-channel to mono if necessary
            if wave.shape[0] > 1:  # Check if it's multi-channel
                wave = wave.mean(dim=0, keepdim=True)  # Convert to mono
            
            # Convert to float32 and remove batch dimension
            wave = wave.squeeze(0).float()  # shape (samples,)
            
            audio_parts.append(wave)
        # Get the target score/label for this sample
        label = row["score_norm"]  # adjust if your CSV uses a different column name for the score
        return {
            "part1": audio_parts[0],
            "part3": audio_parts[1],
            "part4": audio_parts[2],
            "part5": audio_parts[3],
            "label": label
        }

class MultiAudioDataCollator:
    def __call__(self, batch):
        # batch is a list of dataset examples (dicts)
        part1_list = [item["part1"] for item in batch]
        part3_list = [item["part3"] for item in batch]
        part4_list = [item["part4"] for item in batch]
        part5_list = [item["part5"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        return {
            "part1": part1_list,
            "part3": part3_list,
            "part4": part4_list,
            "part5": part5_list,
            "labels": labels
        }


class MultiAudioWhisperModel(nn.Module):
    def __init__(self, whisper_name="openai/whisper-small", chunk_duration=30):
        super().__init__()
        # Load pre-trained Whisper (encoder-decoder model)
        # Use the encoder part of the model for feature extraction
        # and the decoder part is not needed for this task (see paper)
        self.whisper = WhisperModel.from_pretrained(whisper_name)

        # Set up feature extractor for log-Mel spectrograms
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_name)
        self.sample_rate = 16000
        self.chunk_samples = int(chunk_duration * self.sample_rate)  # number of audio samples in one chunk

        # Define a prediction head that outputs a single score
        hidden_size = self.whisper.config.d_model  # Whisper encoder hidden size
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, part1, part3, part4, part5, labels=None):
        device = next(self.whisper.parameters()).device  # current device (CPU/GPU) for model tensors
        batch_size = len(part1)  # number of samples in the batch
        predictions = []  # collect predictions for each sample
        
        # Loop over each sample in the batch
        for i in range(batch_size):
            # List of the four audio parts for this sample
            audio_parts = [part1[i], part3[i], part4[i], part5[i]]
            part_embeddings = []
            
            # Process each audio part independently
            # Extract chunk embeddings
            for audio in audio_parts:
                # Ensure the audio is a NumPy array of type float32                
                # Move tensor to CPU before converting to NumPy
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.cpu().numpy()
                else:
                    audio_np = np.array(audio, dtype=np.float32)
                                
                # Split the audio into chunks of length <= chunk_duration (in seconds)
                total_len = len(audio_np)
                num_chunks = int(np.ceil(total_len / self.chunk_samples)) if total_len > 0 else 0
                chunk_embeds = []
                for j in range(num_chunks):
                    start = j * self.chunk_samples
                    end = min((j + 1) * self.chunk_samples, total_len)
                    audio_chunk = audio_np[start:end]
                    
                    # Convert the audio chunk to log-Mel spectrogram features
                    inputs = self.feature_extractor(audio_chunk, 
                                                    sampling_rate=self.sample_rate, 
                                                    return_tensors="pt")
                    input_features = inputs.input_features.to(device)  # shape: (1, T, time_frames)
                    
                    # Forward pass through the Whisper encoder (no decoder needed for embedding extraction)
                    encoder_outputs = self.whisper.encoder(input_features)
                    hidden_states = encoder_outputs.last_hidden_state  # shape: (1, T, hidden_size)

                    # Pool the encoder output over the time dimension to get a fixed-length chunk embedding
                    chunk_embed = hidden_states.mean(dim=1)  # shape: (1, hidden_size)
                    chunk_embeds.append(chunk_embed)
                
                # Merge chunk embeddings for this part (average pooling across chunks)
                if len(chunk_embeds) == 0:
                    # If for some reason the audio part is empty, use a zero vector
                    part_embed = torch.zeros((1, self.whisper.config.d_model), device=device)
                elif len(chunk_embeds) == 1:
                    part_embed = chunk_embeds[0]  # single chunk, use it directly
                else:
                    # Average multiple chunk embeddings to get one embedding for the part
                    part_embed = torch.mean(torch.stack(chunk_embeds, dim=0), dim=0)  # shape: (1, hidden_size)
                part_embeddings.append(part_embed)
            
            # Aggregate embeddings from all four parts (average pooling across parts)
            combined_embed = torch.mean(torch.stack(part_embeddings, dim=0), dim=0)  # shape: (1, hidden_size)
            # Predict the overall score from the combined embedding
            pred_score = self.fc(combined_embed)  # shape: (1, 1)
            predictions.append(pred_score.squeeze(0))  # remove the batch dimension for this sample
        
        # Stack predictions for all samples in the batch
        preds = torch.cat(predictions, dim=0)           # shape: (batch_size, 1)
        preds = preds.view(batch_size)                  # shape: (batch_size,) as a 1D tensor of scores

        # If labels are provided, compute the loss (using RMSE for regression)
        loss = None
        if labels is not None:
            labels = labels.to(device, dtype=torch.float32)

            # RMSE loss instead of MSE            
            # Scale the loss by 100 to avoid mixed precision underflow (see paper)
            loss = 100 * torch.sqrt(nn.functional.mse_loss(preds, labels) + 1e-6)        
        
        # Return output for Trainer compatibility
        if loss is not None:
            return {"loss": loss, "logits": preds}
        else:
            return {"logits": preds}

def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute accuracy and RMSE for a regression output in [2..5.5].
    """
    # eval_pred.predictions is shape (batch_size, 1)
    # eval_pred.label_ids is shape (batch_size,)
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Squeeze the extra dim => shape (batch_size,)
    # Ensure logits is a 1D array
    preds = logits.flatten()  # This works for both (batch_size, 1) and (batch_size,)
    
    # normalise
    # preds and labels are expected to be in the range [2, 5.5]
    # during preprocessing, we normalised the scores by subtracting 4.0
    # So we need to reverse this operation to get the original scores
    preds = preds + 4.0
    labels = labels + 4.0
    preds = np.clip(preds, 2.0, 5.5)

    # Round to nearest integer and clamp to [2, 6] if desired
    preds_rounded = np.rint(preds).clip(2, 6)

    # Compute "exact match" accuracy (only if you want integer exact matches)
    accuracy = (preds_rounded == labels).mean()

    rmse = np.sqrt(np.mean((preds - labels) ** 2))

    return {
        "accuracy": accuracy,
        "rmse": rmse,
    }

def save_predictions(trainer, dataset, output_file):
    predictions = trainer.predict(dataset)
    preds = predictions.predictions.flatten() # Ensure it's a 1D array
    labels = predictions.label_ids

    # normalise
    # preds and labels are expected to be in the range [2, 5.5]
    # during preprocessing, we normalised the scores by subtracting 4.0
    # So we need to reverse this operation to get the original scores
    preds = preds + 4.0
    labels = labels + 4.0
    preds = np.clip(preds, 2.0, 5.5)

    df = pd.DataFrame({
        "predicted_score": preds.flatten(),
        "true_score": labels.flatten()
    })

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Define bin edges from 2 to 5.5 with 0.25 increments
    bin_edges = np.arange(2.0, 5.75, 0.25)  # 5.75 ensures it includes 5.5
    bin_labels = [round(b, 2) for b in bin_edges[:-1]]  # Labels for confusion matrix

    # Discretize predictions and labels into bins
    preds_binned = np.digitize(preds, bins=bin_edges, right=False) - 1
    labels_binned = np.digitize(labels, bins=bin_edges, right=False) - 1

    # Generate confusion matrix for wandb
    # Remove if not using wandb
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                        y_true=labels_binned, preds=preds_binned,
                        class_names=[str(b) for b in bin_edges])})
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--dev_tsv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunk_sec", type=float, default=30.0)
    parser.add_argument("--output_dir", type=str, default="whisper_slate2025")
    parser.add_argument("--run_name", type=str, default="whisper_slate2025")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--focal_gamma", type=float, default=2.0) # Focal loss gamma, not used in this code and the paper
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the updated dataset
    train_dataset = MultiAudioDataset(args.train_tsv)
    dev_dataset = MultiAudioDataset(args.dev_tsv)

    # Create the new data collator
    data_collator = MultiAudioDataCollator()
    
    # Initialize the new model
    model = MultiAudioWhisperModel(
        whisper_name="openai/whisper-small",
        chunk_duration=args.chunk_sec
    ).to(device)
    
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_total_limit=5,
        load_best_model_at_end=True,
        report_to="wandb", # Log to W&B, remove if not using
        run_name=args.run_name,
        fp16=True,
        warmup_ratio=0.05,
        metric_for_best_model="rmse",
        greater_is_better=False,
        logging_dir="./logs",
        logging_steps=50,
    )

    # Build Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Check if there's a last checkpoint in the output directory
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir) and 
        len(os.listdir(args.output_dir)) > 0
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint {last_checkpoint}")
            
    # Train
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Evaluate
    eval_results = trainer.evaluate()
    print("Final evaluation:", eval_results)

    # Save final model
    trainer.save_model(args.output_dir)

    # Save predictions
    save_predictions(trainer, dev_dataset, f"{args.output_dir}/predictions.csv")
        
    # Print all arguments after running
    print("\nTraining Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("\n")

    # Log the final evaluation results to W&B
    # remove if not using W&B
    wandb.init(project=PROJECT_NAME, name=args.run_name)
    # Save the script to W&B
    wandb.save(__file__)
    wandb.save("*.sh")
    # Finish the W&B run
    wandb.finish()
    
if __name__ == "__main__":
    main()