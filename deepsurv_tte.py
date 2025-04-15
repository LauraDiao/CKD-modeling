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
        logging.FileHandler("deepserv_tte.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CKD classification and time-to-event training with DeepSurv models replacing the TTE branch.")
    parser.add_argument("--embedding-root", type=str, default="./ckd_embeddings_100", help="Path to embeddings.")
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
    # Log transformation flag for TTE (still present if you want to transform survival time in preprocessing)
    parser.add_argument("--log-tte", action="store_true", help="Apply log transformation to time-to-event targets")
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
                # For survival models the label serves as the event indicator (0 for censored, 1 for event)
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

# The following are your original classification models (unchanged)
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

##############################################
# Revised DeepSurv Model Definitions with Auxiliary Classification
##############################################

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

##############################################
# Revised DeepSurv Training and Evaluation Function
##############################################

##############################################
# DeepSurv Loss and Evaluation Functions
##############################################

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
    event = event.astype(int)  # Use the built-in int instead of np.int
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
    # We use cross-entropy for the auxiliary classification branch.
    classification_criterion = nn.CrossEntropyLoss()
    aux_weight = 1.0  # Adjust the weight of the classification loss if desired.

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"

    # Training loop (multi-task loss: survival + classification)
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            x_batch, event_batch, _, _, tte_batch = batch
            x_batch = x_batch.to(device)
            event_batch = event_batch.to(device)
            tte_batch = tte_batch.to(device)
            # Forward pass returns both auxiliary classification logits and survival risk.
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

    logger.info(f"{model_name}: Loading best model for final evaluation.")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # Evaluation: obtain both auxiliary (classification) predictions and survival risk scores.
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
    # Compute c-index on survival risk predictions.
    c_index = concordance_index(all_events, all_times, all_risk)

    # Use the auxiliary classifier's outputs for classification evaluation.
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
    # Return both classification metrics and the c-index.
    result = {"model_name": model_name, "concordance_index": c_index}
    result.update(raw_metrics)
    return result





##############################################
# End DeepSurv Loss and Evaluation Functions
##############################################

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
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"  # Fixed: using output_model_prefix

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
                logits, _ = model(x_batch), None  # This branch is not used for DeepSurv
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
    logger.info(f"{model_name}: Loading best model for final evaluation.")
    model.load_state_dict(torch.load(best_model_path))
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
    for k in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc"]:
        point_est = raw_metrics[k]
        meanv, lower, upper = ci_dict[k]
        logger.info(f"{model_name} {k}: {point_est:.4f} (95% CI: {lower:.4f}-{upper:.4f})")
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

    logger.info("Adding time-to-event columns for survival analysis.")
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

    logger.info("Building sequences for survival analysis (DeepSurv).")
    train_sequences_surv = build_sequences_for_multitask(train_metadata, args.window_size)
    val_sequences_surv = build_sequences_for_multitask(val_metadata, args.window_size)
    test_sequences_surv = build_sequences_for_multitask(test_metadata, args.window_size)
    train_dataset_surv = CKDSequenceDataset(train_sequences_surv, args.window_size, args.embed_dim, use_survival=True)
    val_dataset_surv = CKDSequenceDataset(val_sequences_surv, args.window_size, args.embed_dim, use_survival=True)
    test_dataset_surv = CKDSequenceDataset(test_sequences_surv, args.window_size, args.embed_dim, use_survival=True)
    train_loader_surv = DataLoader(train_dataset_surv, batch_size=args.batch_size, shuffle=True)
    val_loader_surv = DataLoader(val_dataset_surv, batch_size=args.batch_size, shuffle=False)
    test_loader_surv = DataLoader(test_dataset_surv, batch_size=args.batch_size, shuffle=False)

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

    logger.info("Training classification-only models (RNN, LSTM, Transformer, MLP, TCN).")
    rnn_results = train_and_evaluate(improved_rnn, device, train_loader_class, val_loader_class, test_loader_class, args, "ImprovedRNN")
    lstm_results = train_and_evaluate(improved_lstm, device, train_loader_class, val_loader_class, test_loader_class, args, "ImprovedLSTM")
    transformer_results = train_and_evaluate(improved_transformer, device, train_loader_class, val_loader_class, test_loader_class, args, "ImprovedTransformer")
    mlp_results = train_and_evaluate(mlp_model, device, train_loader_class, val_loader_class, test_loader_class, args, "MLP")
    tcn_results = train_and_evaluate(tcn_model, device, train_loader_class, val_loader_class, test_loader_class, args, "TCN")

    logger.info("Defining DeepSurv models for survival analysis.")
    deep_surv_rnn = DeepSurvRNN(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        bidirectional=args.rnn_bidir
    )
    deep_surv_lstm = DeepSurvLSTM(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout,
        bidirectional=args.rnn_bidir
    )
    deep_surv_transformer = DeepSurvTransformer(
        input_dim=args.embed_dim,
        num_layers=args.num_layers,
        nhead=args.transformer_nhead,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout
    )
    deep_surv_mlp = DeepSurvMLP(
        input_dim=args.embed_dim,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.rnn_dropout
    )
    deep_surv_tcn = DeepSurvTCN(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=3,
        dropout=args.rnn_dropout
    )

    logger.info("Training DeepSurv models (RNN, LSTM, Transformer, MLP, TCN) for time-to-event prediction.")
    deep_surv_rnn_results = train_and_evaluate_deepsurv(deep_surv_rnn, device, train_loader_surv, val_loader_surv, test_loader_surv, args, "DeepSurv_RNN")
    deep_surv_lstm_results = train_and_evaluate_deepsurv(deep_surv_lstm, device, train_loader_surv, val_loader_surv, test_loader_surv, args, "DeepSurv_LSTM")
    deep_surv_transformer_results = train_and_evaluate_deepsurv(deep_surv_transformer, device, train_loader_surv, val_loader_surv, test_loader_surv, args, "DeepSurv_Transformer")
    deep_surv_mlp_results = train_and_evaluate_deepsurv(deep_surv_mlp, device, train_loader_surv, val_loader_surv, test_loader_surv, args, "DeepSurv_MLP")
    deep_surv_tcn_results = train_and_evaluate_deepsurv(deep_surv_tcn, device, train_loader_surv, val_loader_surv, test_loader_surv, args, "DeepSurv_TCN")

    logger.info("All trainings complete. Summary of final test metrics for classification models follows.")
    all_results = [
        rnn_results,
        lstm_results,
        transformer_results,
        mlp_results,
        tcn_results
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

    logger.info("DeepSurv model results for time-to-event prediction:")
    deep_surv_results = [
        deep_surv_rnn_results,
        deep_surv_lstm_results,
        deep_surv_transformer_results,
        deep_surv_mlp_results,
        deep_surv_tcn_results
    ]
    for result in deep_surv_results:
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
    for result in deep_surv_results:
        if result is None:
            continue
        logger.info(f"Model={result['model_name']} Concordance Index={result['concordance_index']:.4f}")

    logger.info("Analyzing label switches on test set for classification models.")
    models_for_analysis = [
        ("RNN", improved_rnn),
        ("LSTM", improved_lstm),
        ("Transformer", improved_transformer),
        ("MLP", mlp_model),
        ("Deep_surv_TCN", tcn_model),
        ("Deep_surv_RNN", deep_surv_rnn),
        ("Deep_surv_LSTM", deep_surv_lstm),
        ("Deep_surv_Transformer", deep_surv_transformer),
        ("Deep_surv_MLP", deep_surv_mlp),
        ("Deep_surv_TCN", deep_surv_tcn)
    ]
    for name, model_obj in models_for_analysis:
        df_preds = predict_label_switches(model_obj, test_loader_class, device)
        df_analysis = analyze_switches(df_preds)
        logger.info(f"Label-switch analysis for {name}:\n{df_analysis.head(10)}\n(Showing up to 10 rows)")

if __name__ == "__main__":
    main()
