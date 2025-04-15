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
    precision_recall_fscore_support,
    confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("full_tte_e2e.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Example CKD classification and time-to-event (multi-task) training.")
    parser.add_argument("--embedding-root", type=str, default="./ckd_embeddings_full", help="Path to embeddings.")
    parser.add_argument("--window-size", type=int, default=10, help="Sequence window size.")
    parser.add_argument("--embed-dim", type=int, default=768, help="Dimensionality of embeddings.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--scheduler-patience", type=int, default=2, help="Patience for scheduler LR reduction.")
    parser.add_argument("--metadata-file", type=str, default="patient_embedding_metadata.csv", help="CSV with metadata.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for RNN/LSTM/TCN.")
    parser.add_argument("--num-layers", type=int, default=2, help="Layers for RNN/LSTM/Transformer/TCN.")
    parser.add_argument("--rnn-dropout", type=float, default=0.2, help="Dropout in RNN/LSTM.")
    parser.add_argument("--rnn-bidir", action="store_true", help="Use bidirectional RNN/LSTM if set.")
    parser.add_argument("--transformer-nhead", type=int, default=4, help="Heads in Transformer encoder.")
    parser.add_argument("--transformer-dim-feedforward", type=int, default=256, help="Transformer feedforward dim.")
    parser.add_argument("--transformer-dropout", type=float, default=0.2, help="Dropout in Transformer layers.")
    parser.add_argument("--max-patients", type=int, default=None, help="If set, only load embeddings for up to this many patients.")
    parser.add_argument("--output-model-prefix", type=str, default="best_model", help="Filename prefix for saved models.")
    return parser.parse_args()

def clean_ckd_stage(value):
    try:
        return int(value)
    except ValueError:
        if isinstance(value, str) and value[0].isdigit():
            return int(value[0])
        else:
            return np.nan

def embedding_exists(row, root):
    return os.path.exists(os.path.join(root, row["embedding_file"]))

def load_embedding(full_path, cache):
    if full_path not in cache:
        with np.load(full_path) as data:
            keys = list(data.keys())
            cache[full_path] = data[keys[0]]
    return cache[full_path]

def pad_sequence(seq, length, dim):
    if len(seq) < length:
        padding = [np.zeros(dim)] * (length - len(seq))
        seq = padding + seq
    return np.stack(seq[-length:], axis=0)

class CKDSequenceDataset(Dataset):
    def __init__(self, data, window_size, embed_dim, use_survival=False):
        self.data = data
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.use_survival = use_survival

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_survival:
            context, label, pid, local_idx, tte = self.data[idx]
            context_padded = pad_sequence(context, self.window_size, self.embed_dim)
            return (
                torch.tensor(context_padded, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                pid,
                local_idx,
                torch.tensor(tte, dtype=torch.float32)
            )
        else:
            context, label, pid, local_idx = self.data[idx]
            context_padded = pad_sequence(context, self.window_size, self.embed_dim)
            return (
                torch.tensor(context_padded, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                pid,
                local_idx
            )

class LongitudinalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional
        )
        self.classifier = nn.Linear(hidden_dim * self.num_directions, 2)

    def forward(self, x):
        _, h_n = self.rnn(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        top_layer = h_n[-1]
        top_layer = top_layer.transpose(0, 1).contiguous().view(x.size(0), -1)
        return self.classifier(top_layer)

class LongitudinalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional
        )
        self.classifier = nn.Linear(hidden_dim * self.num_directions, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        top_layer = h_n[-1]
        top_layer = top_layer.transpose(0, 1).contiguous().view(x.size(0), -1)
        return self.classifier(top_layer)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class LongitudinalTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x):
        out = self.pos_encoder(x)
        out = self.transformer_encoder(out)
        last_token = out[:, -1, :]
        return self.classifier(last_token)

class MLPSimple(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.flatten_dim = input_dim * window_size
        layers = []
        current_dim = self.flatten_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out[:, :, :x.size(2)]
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu2(out)
        out = self.dropout2(out)
        return out

class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            out_channels = hidden_dim
            dilation_size = 2**i
            padding = (kernel_size - 1) * dilation_size
            block = TemporalBlock(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=1, dilation=dilation_size, padding=padding, dropout=dropout
            )
            layers.append(block)
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        last_time = out[:, :, -1]
        return self.classifier(last_time)

class MultiTaskTimeToEventRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional
        )
        self.classifier_head = nn.Linear(hidden_dim * self.num_directions, 2)
        self.time_head = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        top_layer = h_n[-1]
        top_layer = top_layer.transpose(0, 1).contiguous().view(x.size(0), -1)
        logits = self.classifier_head(top_layer)
        time_pred = self.time_head(top_layer).squeeze(-1)
        return logits, time_pred

# Additional multitask variants for LSTM, Transformer, MLP, and TCN follow the same principle,
# where a shared representation is mapped to two output heads.

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional
        )
        self.classifier_head = nn.Linear(hidden_dim * self.num_directions, 2)
        self.time_head = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        top_layer = h_n[-1]
        top_layer = top_layer.transpose(0, 1).contiguous().view(x.size(0), -1)
        logits = self.classifier_head(top_layer)
        time_pred = self.time_head(top_layer).squeeze(-1)
        return logits, time_pred

class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward, dropout=0.1, max_len=5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier_head = nn.Linear(input_dim, 2)
        self.time_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last_token = out[:, -1, :]
        logits = self.classifier_head(last_token)
        time_pred = self.time_head(last_token).squeeze(-1)
        return logits, time_pred

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.flatten_dim = input_dim * window_size
        layers = []
        current_dim = self.flatten_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)
        self.classifier_head = nn.Linear(current_dim, 2)
        self.time_head = nn.Linear(current_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        shared_repr = self.shared_net(x)
        logits = self.classifier_head(shared_repr)
        time_pred = self.time_head(shared_repr).squeeze(-1)
        return logits, time_pred

class MultiTaskTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            out_channels = hidden_dim
            dilation_size = 2**i
            padding = (kernel_size - 1) * dilation_size
            block = TemporalBlock(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=1, dilation=dilation_size, padding=padding, dropout=dropout
            )
            layers.append(block)
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.classifier_head = nn.Linear(hidden_dim, 2)
        self.time_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        last_time = out[:, :, -1]
        logits = self.classifier_head(last_time)
        time_pred = self.time_head(last_time).squeeze(-1)
        return logits, time_pred

def compute_metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    ppv = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    npv = 0.0 if (tn + fn) == 0 else tn / (tn + fn)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "ppv": ppv,
        "npv": npv,
        "auroc": auroc,
        "auprc": auprc
    }

def bootstrap_metrics(labels, probs, threshold, n_boot=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(labels)
    all_keys = ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc"]
    metric_samples = {k: [] for k in all_keys}
    for _ in range(n_boot):
        indices = rng.randint(0, n, size=n)
        sample_labels = labels[indices]
        sample_probs = probs[indices]
        sample_result = compute_metrics_at_threshold(sample_labels, sample_probs, threshold)
        for k in all_keys:
            metric_samples[k].append(sample_result[k])
    ci_results = {}
    for k in all_keys:
        arr = np.array(metric_samples[k])
        lower = np.percentile(arr, 2.5)
        upper = np.percentile(arr, 97.5)
        meanv = np.mean(arr)
        ci_results[k] = (meanv, lower, upper)
    return ci_results

def time_to_event_preprocessing(meta):
    meta = meta.sort_values(by=["PatientID", "EventDate"]).reset_index(drop=True)
    meta["EventDate"] = pd.to_datetime(meta["EventDate"])
    meta["time_until_progression"] = np.nan
    grouped = meta.groupby("PatientID")
    for pid, group in grouped:
        indices_with_label1 = group.index[group["label"] == 1].tolist()
        if len(indices_with_label1) == 0:
            continue
        first_prog_index = indices_with_label1[0]
        progression_date = group.loc[first_prog_index, "EventDate"]
        for i in group.index:
            current_date = group.loc[i, "EventDate"]
            if i <= first_prog_index:
                delta = (progression_date - current_date).days
                meta.loc[i, "time_until_progression"] = float(delta)
    return meta

def build_sequences_for_multitask(meta, window_size):
    data_records = []
    for pid, group in meta.groupby("PatientID"):
        group = group.sort_values(by="EventDate")
        embeddings = list(group["embedding"])
        labels = list(group["next_label"])
        tte_vals = list(group["time_until_progression"])
        for i in range(1, len(embeddings)):
            start_idx = max(0, i - window_size)
            context = embeddings[start_idx:i]
            target = labels[i - 1]
            time_for_aux = tte_vals[i - 1]
            data_records.append((context, target, pid, i - 1, time_for_aux))
    return data_records

def build_sequences(meta, window_size):
    # For classification-only training with monotonic constraints.
    def monotonic_labels(labels):
        monotonic = []
        has_progressed = False
        for lab in labels:
            if lab == 1:
                has_progressed = True
            monotonic.append(1 if has_progressed else 0)
        return monotonic

    sequence_data = []
    for pid, group in meta.groupby("PatientID"):
        group = group.sort_values(by="EventDate")
        embeddings = list(group["embedding"])
        original_labels = list(group["next_label"])
        labels = monotonic_labels(original_labels)
        for i in range(1, len(embeddings)):
            start_idx = max(0, i - window_size)
            context = embeddings[start_idx:i]
            target = labels[i - 1]
            sequence_data.append((context, target, pid, i - 1))
    return sequence_data

def train_and_evaluate(model, device, train_loader, val_loader, test_loader, args, model_name):
    logger.info(f"Starting {model_name} training.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Use classification + MSE (for TTE) if multitask; classification only otherwise.
    if hasattr(model, 'time_head'):
        classification_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()
    else:
        classification_criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.scheduler_patience)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            if hasattr(model, 'time_head'):
                x_batch, y_batch, _, _, tte_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                tte_batch = tte_batch.to(device)
                optimizer.zero_grad()
                logits, time_pred = model(x_batch)
                class_loss = classification_criterion(logits, y_batch)
                mask = ~torch.isnan(tte_batch)
                if mask.sum() > 0:
                    time_loss = mse_criterion(time_pred[mask], tte_batch[mask])
                else:
                    time_loss = 0.0
                loss = class_loss + 0.1 * time_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(loss.item())
            else:
                x_batch, y_batch, _, _ = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = classification_criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if hasattr(model, 'time_head'):
                    x_batch, y_batch, _, _, tte_batch = batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    tte_batch = tte_batch.to(device)
                    logits, time_pred = model(x_batch)
                    class_loss = classification_criterion(logits, y_batch)
                    mask = ~torch.isnan(tte_batch)
                    if mask.sum() > 0:
                        time_loss = mse_criterion(time_pred[mask], tte_batch[mask])
                    else:
                        time_loss = 0.0
                    val_loss = class_loss + 0.1 * time_loss
                    val_losses.append(val_loss.item())
                else:
                    x_batch, y_batch, _, _ = batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits = model(x_batch)
                    loss = classification_criterion(logits, y_batch)
                    val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        logger.info(f"{model_name} Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

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
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if hasattr(model, 'time_head'):
                x_batch, y_batch, _, _, _ = batch
                x_batch = x_batch.to(device)
                logits, _ = model(x_batch)
            else:
                x_batch, y_batch, _, _ = batch
                x_batch = x_batch.to(device)
                logits = model(x_batch)
            probs = nn.Softmax(dim=1)(logits)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    prevalence = np.mean(all_labels)
    logger.info(f"The Prevalance is {prevalence:.4f}")
    threshold = prevalence
    raw_metrics = compute_metrics_at_threshold(all_labels, all_probs, threshold)
    ci_dict = bootstrap_metrics(all_labels, all_probs, threshold)

    logger.info(f"{model_name} Threshold set to prevalence={threshold:.4f}.")
    for k in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc"]:
        point_est = raw_metrics[k]
        meanv, lower, upper = ci_dict[k]
        logger.info(f"{model_name} {k}: {point_est:.4f} (95% CI: {lower:.4f}-{upper:.4f})")

    return {
        "model_name": model_name,
        **raw_metrics
    }

def predict_label_switches(model, loader, device):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            if hasattr(model, 'time_head'):
                x_batch, y_batch, pid_batch, idx_batch, _ = batch
                x_batch = x_batch.to(device)
                logits, _ = model(x_batch)
            else:
                x_batch, y_batch, pid_batch, idx_batch = batch
                x_batch = x_batch.to(device)
                logits = model(x_batch)
            probs = nn.Softmax(dim=1)(logits)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            for pid, true_label, pred_label, local_idx in zip(pid_batch, y_batch.numpy(), preds, idx_batch.numpy()):
                records.append((pid, local_idx, true_label, pred_label))
    df = pd.DataFrame(records, columns=["PatientID", "LocalIndex", "TrueLabel", "PredLabel"])
    return df

def analyze_switches(df):
    analysis = []
    for pid, group in df.groupby("PatientID"):
        group = group.sort_values("LocalIndex").reset_index(drop=True)
        true_switch = group[group["TrueLabel"] == 1]["LocalIndex"].min()
        pred_switch = group[group["PredLabel"] == 1]["LocalIndex"].min()
        analysis.append({
            "PatientID": pid,
            "TrueSwitchIdx": true_switch if pd.notnull(true_switch) else None,
            "PredSwitchIdx": pred_switch if pd.notnull(pred_switch) else None
        })
    analysis_df = pd.DataFrame(analysis)
    analysis_df["SwitchDifference"] = analysis_df["PredSwitchIdx"] - analysis_df["TrueSwitchIdx"]
    return analysis_df

def main():
    args = parse_args()
    logger.info("Running with the following configuration:")
    for key, val in vars(args).items():
        logger.info(f"{key}: {val}")

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
    metadata['label'] = metadata['CKD_stage_clean'].apply(lambda x: 0 if x < 4 else 1)
    metadata['next_label'] = metadata.groupby('PatientID')['label'].shift(-1)
    metadata = metadata.dropna(subset=['next_label'])
    metadata['next_label'] = metadata['next_label'].astype(int)

    logger.info("Filtering valid embedding rows.")
    metadata = metadata[metadata.apply(lambda row: embedding_exists(row, args.embedding_root), axis=1)]
    unique_patients = sorted(metadata['PatientID'].unique())
    if args.max_patients is not None and args.max_patients < len(unique_patients):
        subset_patients = unique_patients[: args.max_patients]
        metadata = metadata[metadata['PatientID'].isin(subset_patients)]
        logger.info(f"Using only {args.max_patients} patients for debugging.")

    embedding_cache = {}
    def load_embedding_for_row(row):
        path = os.path.join(args.embedding_root, row['embedding_file'])
        return load_embedding(path, embedding_cache)

    logger.info("Loading embeddings.")
    metadata['embedding'] = metadata.apply(load_embedding_for_row, axis=1)

    logger.info("Creating train/val/test splits.")
    unique_patients = metadata['PatientID'].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=args.random_seed)
    train_patients, val_patients = train_test_split(train_patients, test_size=0.1, random_state=args.random_seed)

    train_metadata = metadata[metadata['PatientID'].isin(train_patients)]
    val_metadata = metadata[metadata['PatientID'].isin(val_patients)]
    test_metadata = metadata[metadata['PatientID'].isin(test_patients)]

    logger.info("Adding time-to-event columns for multi-task learning.")
    train_metadata = time_to_event_preprocessing(train_metadata)
    val_metadata = time_to_event_preprocessing(val_metadata)
    test_metadata = time_to_event_preprocessing(test_metadata)

    logger.info("Building sequence datasets.")
    train_sequences_class = build_sequences(train_metadata, args.window_size)
    val_sequences_class = build_sequences(val_metadata, args.window_size)
    test_sequences_class = build_sequences(test_metadata, args.window_size)
    train_dataset_class = CKDSequenceDataset(train_sequences_class, args.window_size, args.embed_dim, use_survival=False)
    val_dataset_class = CKDSequenceDataset(val_sequences_class, args.window_size, args.embed_dim, use_survival=False)
    test_dataset_class = CKDSequenceDataset(test_sequences_class, args.window_size, args.embed_dim, use_survival=False)
    train_loader_class = DataLoader(train_dataset_class, batch_size=args.batch_size, shuffle=True)
    val_loader_class = DataLoader(val_dataset_class, batch_size=args.batch_size, shuffle=False)
    test_loader_class = DataLoader(test_dataset_class, batch_size=args.batch_size, shuffle=False)

    logger.info("Building sequences for multi-task (classification + TTE).")
    train_sequences_tte = build_sequences_for_multitask(train_metadata, args.window_size)
    val_sequences_tte = build_sequences_for_multitask(val_metadata, args.window_size)
    test_sequences_tte = build_sequences_for_multitask(test_metadata, args.window_size)
    train_dataset_tte = CKDSequenceDataset(train_sequences_tte, args.window_size, args.embed_dim, use_survival=True)
    val_dataset_tte = CKDSequenceDataset(val_sequences_tte, args.window_size, args.embed_dim, use_survival=True)
    test_dataset_tte = CKDSequenceDataset(test_sequences_tte, args.window_size, args.embed_dim, use_survival=True)
    train_loader_tte = DataLoader(train_dataset_tte, batch_size=args.batch_size, shuffle=True)
    val_loader_tte = DataLoader(val_dataset_tte, batch_size=args.batch_size, shuffle=False)
    test_loader_tte = DataLoader(test_dataset_tte, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Defining classification-only models.")
    improved_rnn = LongitudinalRNN(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        bidirectional=args.rnn_bidir
    )
    improved_lstm = LongitudinalLSTM(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        bidirectional=args.rnn_bidir
    )
    improved_transformer = LongitudinalTransformer(
        input_dim=args.embed_dim,
        num_layers=args.num_layers,
        nhead=args.transformer_nhead,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout
    )
    mlp_model = MLPSimple(
        input_dim=args.embed_dim,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout
    )
    tcn_model = TCN(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=3,
        dropout=args.rnn_dropout
    )

    logger.info("Defining multi-task models.")
    multitask_rnn = MultiTaskTimeToEventRNN(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        bidirectional=args.rnn_bidir
    )
    multitask_lstm = MultiTaskLSTM(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        bidirectional=args.rnn_bidir
    )
    multitask_transformer = MultiTaskTransformer(
        input_dim=args.embed_dim,
        num_layers=args.num_layers,
        nhead=args.transformer_nhead,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout
    )
    multitask_mlp = MultiTaskMLP(
        input_dim=args.embed_dim,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout
    )
    multitask_tcn = MultiTaskTCN(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=3,
        dropout=args.rnn_dropout
    )

    logger.info("Training classification-only models (RNN, LSTM, Transformer, MLP, TCN).")
    rnn_results = train_and_evaluate(improved_rnn, device, train_loader_class, val_loader_class, test_loader_class, args, "ImprovedRNN")
    lstm_results = train_and_evaluate(improved_lstm, device, train_loader_class, val_loader_class, test_loader_class, args, "ImprovedLSTM")
    transformer_results = train_and_evaluate(improved_transformer, device, train_loader_class, val_loader_class, test_loader_class, args, "ImprovedTransformer")
    mlp_results = train_and_evaluate(mlp_model, device, train_loader_class, val_loader_class, test_loader_class, args, "MLP")
    tcn_results = train_and_evaluate(tcn_model, device, train_loader_class, val_loader_class, test_loader_class, args, "TCN")

    logger.info("Training multi-task models (Classification + Time-to-Event) using TTE datasets.")
    multitask_rnn_results = train_and_evaluate(multitask_rnn, device, train_loader_tte, val_loader_tte, test_loader_tte, args, "MultiTaskRNN")
    multitask_lstm_results = train_and_evaluate(multitask_lstm, device, train_loader_tte, val_loader_tte, test_loader_tte, args, "MultiTaskLSTM")
    multitask_transformer_results = train_and_evaluate(multitask_transformer, device, train_loader_tte, val_loader_tte, test_loader_tte, args, "MultiTaskTransformer")
    multitask_mlp_results = train_and_evaluate(multitask_mlp, device, train_loader_tte, val_loader_tte, test_loader_tte, args, "MultiTaskMLP")
    multitask_tcn_results = train_and_evaluate(multitask_tcn, device, train_loader_tte, val_loader_tte, test_loader_tte, args, "MultiTaskTCN")

    logger.info("All trainings complete. Summary of final test metrics follows.")
    all_results = [
        rnn_results,
        lstm_results,
        transformer_results,
        mlp_results,
        tcn_results,
        multitask_rnn_results,
        multitask_lstm_results,
        multitask_transformer_results,
        multitask_mlp_results,
        multitask_tcn_results
    ]
    for result in all_results:
        if result is None:
            continue
        logger.info(
            f"Model={result['model_name']} "
            f"Accuracy={result['accuracy']:.4f} "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1']:.4f} "
            f"PPV={result['ppv']:.4f} "
            f"NPV={result['npv']:.4f} "
            f"AUROC={result['auroc']:.4f} "
            f"AUPRC={result['auprc']:.4f}"
        )

    logger.info("Analyzing label switches on test set for all models.")
    models_for_analysis = [
        ("RNN", improved_rnn),
        ("LSTM", improved_lstm),
        ("Transformer", improved_transformer),
        ("MLP", mlp_model),
        ("TCN", tcn_model),
        ("TTE_RNN", multitask_rnn),
        ("TTE_LSTM", multitask_lstm),
        ("TTE_Transformer", multitask_transformer),
        ("TTE_MLP", multitask_mlp),
        ("TTE_TCN", multitask_tcn)
    ]
    for name, model_obj in models_for_analysis:
        if "MultiTask" in name:
            df_preds = predict_label_switches(model_obj, test_loader_tte, device)
        else:
            df_preds = predict_label_switches(model_obj, test_loader_class, device)
        df_analysis = analyze_switches(df_preds)
        logger.info(f"Label-switch analysis for {name}:\n{df_analysis.head(10)}\n(Showing up to 10 rows)")

if __name__ == "__main__":
    main()
