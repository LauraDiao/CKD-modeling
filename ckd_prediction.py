#!/usr/bin/env python

import os
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple longitudinal models for CKD classification.")
    parser.add_argument("--embedding-root", type=str, default="./ckd_embeddings_100", help="Path to the embeddings directory.")
    parser.add_argument("--window-size", type=int, default=5, help="Sequence window size.")
    parser.add_argument("--embed-dim", type=int, default=768, help="Dimensionality of the embeddings.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train each model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--metadata-file", type=str, default="patient_embedding_metadata.csv", help="Name of the CSV file containing metadata.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for RNN and ODE models.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of GRU layers or Transformer encoder layers.")
    parser.add_argument("--transformer-nhead", type=int, default=4, help="Number of attention heads in the Transformer encoder.")
    parser.add_argument("--transformer-dim-feedforward", type=int, default=256, help="Feedforward dimension in the Transformer encoder.")
    parser.add_argument("--output-model-prefix", type=str, default="best_model", help="Filename prefix for saving models.")
    args = parser.parse_args()
    return args

def clean_ckd_stage(value):
    try:
        return int(value)
    except ValueError:
        if isinstance(value, str) and value[0].isdigit():
            return int(value[0])
        else:
            return np.nan

def embedding_exists(row, root):
    embed_path = os.path.join(root, row["embedding_file"])
    return os.path.exists(embed_path)

def load_embedding(path, cache):
    if path not in cache:
        with np.load(path) as data:
            keys = list(data.keys())
            cache[path] = data[keys[0]]
    return cache[path]

def monotonic_labels(labels):
    # Once label is 1, keep it 1 thereafter to enforce no reversal to 0
    monotonic = []
    has_progressed = False
    for lab in labels:
        if lab == 1:
            has_progressed = True
        monotonic.append(1 if has_progressed else 0)
    return monotonic

def build_sequences(meta, window_size):
    sequence_data = []
    for pid, group in meta.groupby("PatientID"):
        group = group.sort_values(by="EventDate")
        embeddings = list(group["embedding"])
        # Enforce monotonic label progression
        original_labels = list(group["next_label"])
        labels = monotonic_labels(original_labels)
        for i in range(1, len(embeddings)):
            start_idx = max(0, i - window_size)
            context = embeddings[start_idx:i]
            target = labels[i - 1]
            sequence_data.append((context, target))
    return sequence_data

def pad_sequence(seq, length, dim):
    if len(seq) < length:
        padding = [np.zeros(dim)] * (length - len(seq))
        seq = padding + seq
    return np.stack(seq[-length:], axis=0)

class CKDSequenceDataset(Dataset):
    def __init__(self, data, window_size, embed_dim):
        self.data = data
        self.window_size = window_size
        self.embed_dim = embed_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, label = self.data[idx]
        context_padded = pad_sequence(context, self.window_size, self.embed_dim)
        return torch.tensor(context_padded, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# RNN model
class LongitudinalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.classifier(h_n[-1])

# Transformer-based model
class LongitudinalTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # Transformer expects (batch, seq_len, d_model)
        out = self.transformer_encoder(x)
        # We just take the last token representation
        last_token = out[:, -1, :]
        return self.classifier(last_token)

# Rudimentary neural ODE-like model (manual Euler integration for demonstration)
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, y):
        # ODE derivative estimate
        return self.net(y)

class NeuralODEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.initial_map = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        # We'll treat each time step as a small integration step
        h = self.initial_map(x[:, 0, :])  # initialize hidden state from first time step
        for t in range(1, x.size(1)):
            dt = 1.0  # pretend each step is dt=1
            h = h + dt * self.ode_func(h)  # Euler step
            # incorporate new observation into hidden if wanted:
            obs = self.initial_map(x[:, t, :])
            h = (h + obs) / 2
        return self.classifier(h)

def train_and_evaluate(model, device, train_loader, val_loader, test_loader, args, model_name):
    logger.info(f"Starting {model_name} training.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        logger.info(f"{model_name} Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"{model_name}: Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"{model_name}: Early stopping triggered.")
                break

    logger.info(f"{model_name}: Loading best model for final evaluation.")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = nn.Softmax(dim=1)(logits)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    logger.info(f"{model_name} Test Accuracy: {accuracy:.4f}")
    logger.info(f"{model_name} Test F1 Score: {f1:.4f}")
    logger.info(f"{model_name} Test Precision: {precision:.4f}")
    logger.info(f"{model_name} Test Recall: {recall:.4f}")
    logger.info(f"{model_name} Test AUROC: {auroc:.4f}")
    logger.info(f"{model_name} Test AUPRC: {auprc:.4f}")
    logger.info(f"{model_name} training complete.")
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "auprc": auprc
    }

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    logger.info("Loading metadata.")
    metadata_path = os.path.join(args.embedding_root, args.metadata_file)
    metadata = pd.read_csv(metadata_path)
    metadata['CKD_stage_clean'] = metadata['CKD_stage'].apply(clean_ckd_stage)
    metadata = metadata.sort_values(by=['PatientID', 'EventDate'])
    metadata['CKD_stage_clean'] = metadata.groupby('PatientID')['CKD_stage_clean'].bfill()
    metadata = metadata.dropna(subset=['CKD_stage_clean'])
    metadata['CKD_stage_clean'] = metadata['CKD_stage_clean'].astype(int)

    # Binarize: 0 if stage <4, else 1
    metadata['label'] = metadata['CKD_stage_clean'].apply(lambda x: 0 if x < 4 else 1)

    # Next label is the target
    metadata['next_label'] = metadata.groupby('PatientID')['label'].shift(-1)
    metadata = metadata.dropna(subset=['next_label'])
    metadata['next_label'] = metadata['next_label'].astype(int)

    logger.info("Filtering valid embedding rows.")
    metadata = metadata[metadata.apply(lambda row: embedding_exists(row, args.embedding_root), axis=1)]

    logger.info("Loading embeddings.")
    embedding_cache = {}
    def load_embedding_for_row(row):
        full_path = os.path.join(args.embedding_root, row['embedding_file'])
        return load_embedding(full_path, embedding_cache)

    metadata['embedding'] = metadata.apply(load_embedding_for_row, axis=1)

    logger.info("Creating train/val/test splits.")
    unique_patients = metadata['PatientID'].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=args.random_seed)
    train_patients, val_patients = train_test_split(train_patients, test_size=0.1, random_state=args.random_seed)

    train_metadata = metadata[metadata['PatientID'].isin(train_patients)]
    val_metadata = metadata[metadata['PatientID'].isin(val_patients)]
    test_metadata = metadata[metadata['PatientID'].isin(test_patients)]

    logger.info("Building sequence datasets with monotonic progression constraints.")
    train_sequences = build_sequences(train_metadata, args.window_size)
    val_sequences = build_sequences(val_metadata, args.window_size)
    test_sequences = build_sequences(test_metadata, args.window_size)

    train_dataset = CKDSequenceDataset(train_sequences, args.window_size, args.embed_dim)
    val_dataset = CKDSequenceDataset(val_sequences, args.window_size, args.embed_dim)
    test_dataset = CKDSequenceDataset(test_sequences, args.window_size, args.embed_dim)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Defining our models.")
    rnn_model = LongitudinalRNN(args.embed_dim, args.hidden_dim, args.num_layers)
    transformer_model = LongitudinalTransformer(
        input_dim=args.embed_dim,
        num_layers=args.num_layers,
        nhead=args.transformer_nhead,
        dim_feedforward=args.transformer_dim_feedforward
    )
    ode_model = NeuralODEModel(args.embed_dim, args.hidden_dim)

    logger.info("Training RNN, Transformer, and Neural ODE models in succession.")
    rnn_results = train_and_evaluate(rnn_model, device, train_loader, val_loader, test_loader, args, model_name="RNN")
    transformer_results = train_and_evaluate(transformer_model, device, train_loader, val_loader, test_loader, args, model_name="Transformer")
    ode_results = train_and_evaluate(ode_model, device, train_loader, val_loader, test_loader, args, model_name="NeuralODE")

    logger.info("All trainings complete. Summary of final test metrics:")
    for result in [rnn_results, transformer_results, ode_results]:
        logger.info(
            f"Model={result['model_name']} "
            f"Accuracy={result['accuracy']:.4f} "
            f"F1={result['f1']:.4f} "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"AUROC={result['auroc']:.4f} "
            f"AUPRC={result['auprc']:.4f}"
        )

if __name__ == "__main__":
    main()
