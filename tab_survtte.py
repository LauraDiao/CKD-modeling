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
        logging.FileHandler("full_deepserv_tte.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CKD classification and time-to-event training with DeepSurv models replacing the TTE branch.")
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
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for models (RNN, LSTM, Transformer, MLP, TCN).")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers for models.")
    parser.add_argument("--rnn-dropout", type=float, default=0.2, help="Dropout in RNN/LSTM.")
    parser.add_argument("--rnn-bidir", action="store_true", help="Use bidirectional RNN/LSTM if set.")
    parser.add_argument("--transformer-nhead", type=int, default=4, help="Number of heads in Transformer encoder.")
    parser.add_argument("--transformer-dim-feedforward", type=int, default=256, help="Feedforward dim in Transformer layers.")
    parser.add_argument("--transformer-dropout", type=float, default=0.2, help="Dropout in Transformer layers.")
    parser.add_argument("--max-patients", type=int, default=None, help="If set, only load embeddings for up to this many patients.")
    parser.add_argument("--output-model-prefix", type=str, default="best_model", help="Filename prefix for saved models.")
    parser.add_argument("--log-tte", action="store_true", help="Apply log transformation to time-to-event targets")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
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
    # Convert the stacked array to remove any nans.
    padded = np.stack(seq[-length:], axis=0)
    return np.nan_to_num(padded, nan=0.0)

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
                torch.tensor(label, dtype=torch.float32),
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
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)


class LongitudinalTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        # Map input to d_model which is divisible by nhead.
        if input_dim % nhead != 0:
            d_model = nhead * ((input_dim + nhead - 1) // nhead)
        else:
            d_model = input_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.input_projection(x)
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

class DeepSurvRNN(nn.Module):
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
        self.risk = nn.Linear(hidden_dim * self.num_directions, 1)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, 2)
    
    def forward(self, x):
        _, h_n = self.rnn(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        top_layer = h_n[-1]
        top_layer = top_layer.transpose(0, 1).contiguous().view(x.size(0), -1)
        risk = self.risk(top_layer).squeeze(-1)
        logits = self.classifier(top_layer)
        return logits, risk

class DeepSurvLSTM(nn.Module):
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
        self.risk = nn.Linear(hidden_dim * self.num_directions, 1)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, 2)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        top_layer = h_n[-1]
        top_layer = top_layer.transpose(0, 1).contiguous().view(x.size(0), -1)
        risk = self.risk(top_layer).squeeze(-1)
        logits = self.classifier(top_layer)
        return logits, risk

class DeepSurvTransformer(nn.Module):
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
        self.risk = nn.Linear(input_dim, 1)
        self.classifier = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last_token = out[:, -1, :]
        risk = self.risk(last_token).squeeze(-1)
        logits = self.classifier(last_token)
        return logits, risk

class DeepSurvMLP(nn.Module):
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
        self.net = nn.Sequential(*layers)
        self.risk = nn.Linear(current_dim, 1)
        self.classifier = nn.Linear(current_dim, 2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.net(x)
        risk = self.risk(features).squeeze(-1)
        logits = self.classifier(features)
        return logits, risk

class DeepSurvTCN(nn.Module):
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
        self.risk = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        last_time = out[:, :, -1]
        risk = self.risk(last_time).squeeze(-1)
        logits = self.classifier(last_time)
        return logits, risk

def cox_ph_loss(risk, time, event):
    """
    Compute the negative log partial likelihood for Cox proportional hazards.
    Assumes risk, time and event are 1D tensors.
    """
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    time = time[order]
    event = event[order]
    exp_risk = torch.exp(risk)
    cum_sum = torch.cumsum(exp_risk, dim=0)
    eps = 1e-7
    loss = -(risk[event == 1] - torch.log(cum_sum[event == 1] + eps)).sum()
    n_events = event.sum()
    if n_events > 0:
        loss = loss / n_events
    return loss

def concordance_index(event, time, risk):
    """
    Compute the concordance index (C-index) for survival data.
    This implementation is quadratic in the number of samples (for illustration).
    """
    event = event.astype(int)
    n = len(time)
    num = 0.0
    num_valid = 0.0
    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i] == 1:
                num_valid += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return num / num_valid if num_valid > 0 else 0.0

def train_and_evaluate_deepsurv(model, device, train_loader, val_loader, test_loader, args, model_name):
    logger.info(f"Starting {model_name} training (DeepSurv multi-task, classification evaluation).")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.scheduler_patience)
    classification_criterion = nn.CrossEntropyLoss()
    aux_weight = 1.0

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            x_batch, event_batch, _, _, tte_batch = batch
            x_batch = x_batch.to(device)
            event_batch = event_batch.to(device)
            tte_batch = tte_batch.to(device)
            logits, risk = model(x_batch)
            mask = ~torch.isnan(tte_batch)
            if mask.sum() == 0:
                surv_loss = torch.tensor(0.0, device=device)
            else:
                surv_loss = cox_ph_loss(risk[mask], tte_batch[mask], event_batch[mask])
            class_loss = classification_criterion(logits, event_batch.long())
            loss = surv_loss + aux_weight * class_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, event_batch, _, _, tte_batch = batch
                x_batch = x_batch.to(device)
                event_batch = event_batch.to(device)
                tte_batch = tte_batch.to(device)
                logits, risk = model(x_batch)
                mask = ~torch.isnan(tte_batch)
                if mask.sum() == 0:
                    surv_loss = torch.tensor(0.0, device=device)
                else:
                    surv_loss = cox_ph_loss(risk[mask], tte_batch[mask], event_batch[mask])
                class_loss = classification_criterion(logits, event_batch.long())
                loss = surv_loss + aux_weight * class_loss
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
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

    if os.path.exists(best_model_path):
        logger.info(f"{model_name}: Loading best model for final evaluation.")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    else:
        logger.info(f"{model_name}: Best model not saved due to NaN loss. Using current model state.")

    model.eval()

    all_logits = []
    all_risk = []
    all_events = []
    all_times = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, event_batch, _, _, tte_batch = batch
            x_batch = x_batch.to(device)
            logits, risk = model(x_batch)
            all_logits.extend(logits.cpu().numpy())
            all_risk.extend(risk.cpu().numpy())
            all_events.extend(event_batch.numpy())
            all_times.extend(tte_batch.numpy())
    all_events = np.array(all_events)
    all_times = np.array(all_times)
    all_risk = np.array(all_risk)
    c_index = concordance_index(all_events, all_times, all_risk)

    all_logits = np.array(all_logits)
    all_probs = nn.Softmax(dim=1)(torch.tensor(all_logits)).numpy()
    prevalence = np.mean(all_events)
    logger.info(f"{model_name} Test Prevalence: {prevalence:.4f}")
    threshold = prevalence
    raw_metrics = compute_metrics_at_threshold(all_events, all_probs[:, 1], threshold)
    ci_dict = bootstrap_metrics(all_events, all_probs[:, 1], threshold)
    logger.info(f"{model_name} Threshold set to prevalence={threshold:.4f}.")
    for k in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc"]:
        point_est = raw_metrics[k]
        meanv, lower, upper = ci_dict[k]
        logger.info(f"{model_name} {k}: {point_est:.4f} (95% CI: {lower:.4f}-{upper:.4f})")
    result = {"model_name": model_name, "concordance_index": c_index}
    result.update(raw_metrics)
    return result

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
    if hasattr(model, 'risk'):
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
            optimizer.zero_grad()
            if hasattr(model, 'risk'):
                x_batch, y_batch, _, _, tte_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                tte_batch = tte_batch.to(device)
                logits, _ = model(x_batch), None
                loss = classification_criterion(logits, y_batch.long())
            else:
                x_batch, y_batch, _, _ = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = classification_criterion(logits, y_batch.long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if hasattr(model, 'risk'):
                    x_batch, y_batch, _, _, tte_batch = batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    tte_batch = tte_batch.to(device)
                    logits, _ = model(x_batch), None
                    loss = classification_criterion(logits, y_batch.long())
                else:
                    x_batch, y_batch, _, _ = batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits = model(x_batch)
                    loss = classification_criterion(logits, y_batch.long())
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
    if os.path.exists(best_model_path):
        logger.info(f"{model_name}: Loading best model for final evaluation.")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    else:
        logger.info(f"{model_name}: Best model not saved due to NaN loss. Using current model state.")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if hasattr(model, 'risk'):
                x_batch, y_batch, _, _, _ = batch
                x_batch = x_batch.to(device)
                logits = model(x_batch)
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
    logger.info(f"The Prevalence is {prevalence:.4f}")
    threshold = prevalence
    raw_metrics = compute_metrics_at_threshold(all_labels, all_probs, threshold)
    ci_dict = bootstrap_metrics(all_labels, all_probs, threshold)
    logger.info(f"{model_name} Threshold set to prevalence={threshold:.4f}.")
    for k, v in raw_metrics.items():
        if k != 'model_name':
            meanv, lower, upper = ci_dict[k]
            logger.info(f"{model_name} {k}: {v:.4f} (95% CI: {lower:.4f}-{upper:.4f})")
    return {"model_name": model_name, **raw_metrics}

def predict_label_switches(model, loader, device):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            if hasattr(model, 'risk'):
                x_batch, y_batch, pid_batch, idx_batch, _ = batch
                x_batch = x_batch.to(device)
                logits = model(x_batch)
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

    logger.info("Loading tabular CKD patient-day file.")
    metadata = pd.read_csv("./ckd_processed_tab_data_100.csv", parse_dates=["EventDate"])
    metadata = metadata.sort_values(by=['PatientID', 'EventDate'])

    if 'CKD_stage' in metadata.columns:
        metadata['CKD_stage'] = metadata['CKD_stage'].replace({'3a': 3.1, '3b': 3.2})

    if 'CKD_stage' in metadata.columns:
        metadata['CKD_stage'] = metadata.groupby('PatientID')['CKD_stage'].bfill()
        metadata = metadata.dropna(subset=['CKD_stage'])

    metadata['label'] = metadata['CKD_stage'].apply(lambda x: 0 if float(x) < 4 else 1)
    metadata['next_label'] = metadata.groupby('PatientID')['label'].shift(-1)
    metadata = metadata.dropna(subset=['next_label'])
    metadata['next_label'] = metadata['next_label'].astype(int)

    feature_cols = metadata.columns.difference(['PatientID', "GFR_combined",'EventDate', 'CKD_stage', 'label', 'next_label'])
    metadata['embedding'] = metadata[feature_cols].values.tolist()
    # Replace any NaNs in the embedding vectors with 0.
    metadata['embedding'] = metadata['embedding'].apply(lambda x: np.nan_to_num(np.array(x), nan=0.0).tolist())

    logger.info("Creating train/val/test splits.")
    unique_patients = metadata['PatientID'].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=args.random_seed)
    train_patients, val_patients = train_test_split(train_patients, test_size=0.1, random_state=args.random_seed)

    train_metadata = metadata[metadata['PatientID'].isin(train_patients)]
    val_metadata = metadata[metadata['PatientID'].isin(val_patients)]
    test_metadata = metadata[metadata['PatientID'].isin(test_patients)]

    logger.info("Adding time-to-event columns.")
    train_metadata = time_to_event_preprocessing(train_metadata)
    val_metadata = time_to_event_preprocessing(val_metadata)
    test_metadata = time_to_event_preprocessing(test_metadata)

    # Fill any remaining NaNs in the metadata with 0.
    logger.info("Filling NaNs with 0 in metadata.")
    train_metadata.fillna(0, inplace=True)
    val_metadata.fillna(0, inplace=True)
    test_metadata.fillna(0, inplace=True)

    logger.info("Building sequence datasets.")
    train_sequences_class = build_sequences(train_metadata, args.window_size)
    val_sequences_class = build_sequences(val_metadata, args.window_size)
    test_sequences_class = build_sequences(test_metadata, args.window_size)

    train_dataset_class = CKDSequenceDataset(train_sequences_class, args.window_size, len(feature_cols), use_survival=False)
    val_dataset_class = CKDSequenceDataset(val_sequences_class, args.window_size, len(feature_cols), use_survival=False)
    test_dataset_class = CKDSequenceDataset(test_sequences_class, args.window_size, len(feature_cols), use_survival=False)

    train_loader_class = DataLoader(train_dataset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader_class = DataLoader(val_dataset_class, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader_class = DataLoader(test_dataset_class, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logger.info("Building sequences for survival analysis (DeepSurv).")
    train_sequences_surv = build_sequences_for_multitask(train_metadata, args.window_size)
    val_sequences_surv = build_sequences_for_multitask(val_metadata, args.window_size)
    test_sequences_surv = build_sequences_for_multitask(test_metadata, args.window_size)

    train_dataset_surv = CKDSequenceDataset(train_sequences_surv, args.window_size, len(feature_cols), use_survival=True)
    val_dataset_surv = CKDSequenceDataset(val_sequences_surv, args.window_size, len(feature_cols), use_survival=True)
    test_dataset_surv = CKDSequenceDataset(test_sequences_surv, args.window_size, len(feature_cols), use_survival=True)

    train_loader_surv = DataLoader(train_dataset_surv, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader_surv = DataLoader(val_dataset_surv, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader_surv = DataLoader(test_dataset_surv, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Defining classification-only models.")
    models_classification = [
        LongitudinalRNN(len(feature_cols), args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        LongitudinalLSTM(len(feature_cols), args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        LongitudinalTransformer(len(feature_cols), args.num_layers, args.transformer_nhead, args.transformer_dim_feedforward, args.transformer_dropout),
        MLPSimple(len(feature_cols), args.window_size, args.hidden_dim, args.num_layers, args.rnn_dropout),
        TCN(len(feature_cols), args.hidden_dim, args.num_layers, 3, args.rnn_dropout)
    ]
    model_names_class = ["RNN", "LSTM", "Transformer", "MLP", "TCN"]

    for model, name in zip(models_classification, model_names_class):
        results = train_and_evaluate(model, device, train_loader_class, val_loader_class, test_loader_class, args, name)
        for k, v in results.items():
            if k != 'model_name':
                logger.info(f"Classification Model {name} - {k}: {v:.4f}")

    logger.info("Defining DeepSurv models for survival analysis.")
    models_survival = [
        DeepSurvRNN(len(feature_cols), args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        DeepSurvLSTM(len(feature_cols), args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        DeepSurvTransformer(len(feature_cols), args.num_layers, args.transformer_nhead, args.transformer_dim_feedforward, args.transformer_dropout),
        DeepSurvMLP(len(feature_cols), args.window_size, args.hidden_dim, args.num_layers, args.rnn_dropout),
        DeepSurvTCN(len(feature_cols), args.hidden_dim, args.num_layers, 3, args.rnn_dropout)
    ]
    model_names_surv = ["DeepSurv_RNN", "DeepSurv_LSTM", "DeepSurv_Transformer", "DeepSurv_MLP", "DeepSurv_TCN"]

    for model, name in zip(models_survival, model_names_surv):
        results = train_and_evaluate_deepsurv(model, device, train_loader_surv, val_loader_surv, test_loader_surv, args, name)
        for k, v in results.items():
            if k != 'model_name':
                logger.info(f"Survival Model {name} - {k}: {v:.4f}")

if __name__ == "__main__":
    main()
