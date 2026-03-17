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
from datetime import timedelta # Added for future label generation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("full_tab_deepserv_1year_future.log", mode='w'), # Updated log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CKD classification (1-year future window) and time-to-event training with DeepSurv models on tabular data.")
    # Removed --embedding-root as we use tabular CSV directly
    parser.add_argument("--tabular-data-file", type=str, default="./ckd_processed_tab_full.csv", help="Path to the processed tabular CKD data CSV file.")
    parser.add_argument("--window-size", type=int, default=10, help="Sequence window size.")
    parser.add_argument("--embed-dim", type=int, default=None, help="Dimensionality of features (will be auto-detected if None).") # Modified
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--scheduler-patience", type=int, default=2, help="Patience for scheduler LR reduction.")
    # Removed --metadata-file as it's combined into --tabular-data-file
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for models (RNN, LSTM, Transformer, MLP, TCN).")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers for models.")
    parser.add_argument("--rnn-dropout", type=float, default=0.2, help="Dropout in RNN/LSTM.")
    parser.add_argument("--rnn-bidir", action="store_true", help="Use bidirectional RNN/LSTM if set.")
    parser.add_argument("--transformer-nhead", type=int, default=4, help="Number of heads in Transformer encoder.")
    parser.add_argument("--transformer-dim-feedforward", type=int, default=256, help="Feedforward dim in Transformer layers.")
    parser.add_argument("--transformer-dropout", type=float, default=0.2, help="Dropout in Transformer layers.")
    parser.add_argument("--max-patients", type=int, default=None, help="If set, only use data for up to this many patients.")
    parser.add_argument("--output-model-prefix", type=str, default="best_tab_model_1yr_future", help="Filename prefix for saved models.") # Updated prefix
    parser.add_argument("--log-tte", action="store_true", help="Apply log transformation to time-to-event targets") # Kept for consistency if TTE values are used
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (set to 0 for Windows or debugging).") # Default to 0 for broader compatibility
    parser.add_argument("--prediction-horizon-days", type=int, default=365, help="Number of days into the future to check for an event for label generation.")
    return parser.parse_args()

def clean_ckd_stage(value):
    try:
        # Handle cases like '3.1' or '3.2' if they are strings from CSV
        val_float = float(value)
        return int(val_float) # Truncate to integer stage
    except ValueError:
        if isinstance(value, str):
            if value.lower() == '3a': return 3
            if value.lower() == '3b': return 3 # Often grouped as stage 3
            if value[0].isdigit():
                return int(value[0])
        return np.nan
    except TypeError: # Handles if value is already NaN or None
        return np.nan


# This function is from the previous script to generate the 1-year future label
def add_future_event_label_column(df, source_label_col, new_label_col, date_col='EventDate', patient_id_col='PatientID', horizon_days=365):
    logger.info(f"Generating '{new_label_col}' based on '{source_label_col}' over a {horizon_days}-day future window.")
    df[new_label_col] = 0
    df[date_col] = pd.to_datetime(df[date_col])
    prediction_timedelta = timedelta(days=horizon_days)
    
    df_copy = df.sort_values(by=[patient_id_col, date_col]).copy()
    temp_new_labels = pd.Series(index=df_copy.index, dtype=int)

    for pid, group in tqdm(df_copy.groupby(patient_id_col), desc=f"Generating {new_label_col}", leave=False, disable=True):
        group_indices = group.index 
        event_dates_list = list(group[date_col])
        source_labels_list = list(group[source_label_col])
        
        for i in range(len(group)):
            current_visit_date = event_dates_list[i]
            horizon_end_date = current_visit_date + prediction_timedelta
            current_original_index = group_indices[i]
            
            future_event_found = 0
            for j in range(i + 1, len(group)): 
                future_visit_date = event_dates_list[j]
                if future_visit_date <= horizon_end_date: 
                    if source_labels_list[j] == 1: 
                        future_event_found = 1
                        break 
                else: 
                    break
            temp_new_labels.loc[current_original_index] = future_event_found
    
    df[new_label_col] = temp_new_labels
    return df

# embedding_exists and load_embedding are not needed for this tabular script

def pad_sequence(seq, length, dim): # seq is a list of feature lists/arrays
    # Ensure all elements in seq are numpy arrays of the correct dim
    processed_seq_elements = []
    for item_list in seq:
        item_array = np.array(item_list, dtype=np.float32)
        if item_array.shape == (dim,):
            processed_seq_elements.append(item_array)
        elif item_array.ndim == 0 and dim == 1: # Handle scalar features if dim is 1
             processed_seq_elements.append(np.array([item_array],dtype=np.float32))
        else:
            logger.warning(f"Unexpected item shape in sequence for padding. Expected ({dim},), got {item_array.shape}. Using zeros.")
            processed_seq_elements.append(np.zeros(dim, dtype=np.float32))
    
    current_len = len(processed_seq_elements)
    if current_len < length:
        padding = [np.zeros(dim, dtype=np.float32)] * (length - current_len)
        processed_seq_elements = padding + processed_seq_elements # Pre-pend padding
    
    padded_arr = np.stack(processed_seq_elements[-length:], axis=0)
    return np.nan_to_num(padded_arr, nan=0.0)


class CKDSequenceDataset(Dataset):
    def __init__(self, data, window_size, embed_dim, use_survival_multitask=False): # Renamed for clarity
        self.data = data
        self.window_size = window_size
        self.embed_dim = embed_dim # This is num_features for tabular data
        self.use_survival_multitask = use_survival_multitask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_survival_multitask:
            # context, classification_target, pid, local_idx, tte_value
            context, classification_target, pid, local_idx, tte = self.data[idx] # This tte is T_cox
            # The 'event' for cox loss in train_and_evaluate_deepsurv is taken from 'classification_target'
            context_padded = pad_sequence(context, self.window_size, self.embed_dim)
            return (
                torch.tensor(context_padded, dtype=torch.float32),
                torch.tensor(classification_target, dtype=torch.long), # Classification target (e.g. label_ckd_1_year_future)
                pid,
                local_idx,
                torch.tensor(tte if pd.notna(tte) else float('nan'), dtype=torch.float32) # TTE (e.g. time_until_progression)
            )
        else: # Standard classification
            context, classification_target, pid, local_idx = self.data[idx]
            context_padded = pad_sequence(context, self.window_size, self.embed_dim)
            return (
                torch.tensor(context_padded, dtype=torch.float32),
                torch.tensor(classification_target, dtype=torch.long), # Classification target
                pid,
                local_idx
            )

# --- Model Architectures (RNN, LSTM, Transformer, MLP, TCN, and their DeepSurv variants) ---
# These are assumed to be the same as in your provided script. For brevity, I'll C&P them.
# Ensure input_dim matches the number of features from your tabular data.

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
        # Get the hidden state of the last layer
        if self.bidirectional:
            top_layer = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        else:
            top_layer = h_n[-1, 0, :, :]
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
        if self.bidirectional:
            top_layer = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        else:
            top_layer = h_n[-1, 0, :, :]
        return self.classifier(top_layer)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model by ensuring cosine part doesn't exceed dimensions
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        pe_to_add = self.pe[:, :seq_len, :].to(x.device)
        return x + pe_to_add


class LongitudinalTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        # Transformer's d_model must be divisible by nhead.
        # If input_dim is not, we might need a projection or ensure input_dim is appropriate.
        # For simplicity, assume input_dim (d_model) is suitable or add a projection if needed.
        d_model = input_dim 
        if d_model % nhead != 0:
            # Adjust d_model or raise error. For now, let's assume it's handled or project.
            # This simple projection ensures d_model is usable.
            logger.warning(f"Transformer input_dim {input_dim} not divisible by nhead {nhead}. Projecting to {nhead * ((input_dim + nhead -1)//nhead)}")
            d_model = nhead * ((input_dim + nhead -1)//nhead) # Make d_model divisible by nhead
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            self.input_projection = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model) # Use d_model for PE
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2) # Classify based on d_model

    def forward(self, x):
        x = self.input_projection(x) # Apply projection
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last_token = out[:, -1, :]
        return self.classifier(last_token)

class MLPSimple(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.flatten_dim = input_dim * window_size
        layers = []
        current_dim = self.flatten_dim
        if num_layers < 1: raise ValueError("MLP must have at least one layer")
        if num_layers == 1:
            layers.append(nn.Linear(current_dim, 2))
        else:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1) # Flatten
        return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU() # To be applied after adding residual

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # Ensure residual connection matches output length if padding makes it different
        # This can happen with causal TCNs where padding is only on one side.
        # A common fix is to slice 'out' or 'res' if lengths mismatch.
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        return self.final_relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3, dropout=0.1):
        super(TCN, self).__init__()
        layers = []
        num_channels_list = [input_dim] + [hidden_dim] * num_layers

        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = num_channels_list[i]
            out_channels = num_channels_list[i+1]
            # Causal padding for TCNs: (kernel_size - 1) * dilation_size
            # This padding is typically applied only to the left side.
            # PyTorch's Conv1d `padding` applies to both sides if it's an int.
            # For true causal, one might use asymmetric padding (e.g. F.pad before conv)
            # or ensure the TemporalBlock handles it or adjust slicing.
            # Here, we assume symmetric padding for simplicity as in original code, or that block handles it.
            padding = (kernel_size - 1) * dilation_size 
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x): # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len) for Conv1d
        out = self.network(x)   # (batch, hidden_dim, seq_len)
        # Use the output of the last time step
        last_time_features = out[:, :, -1] # (batch, hidden_dim)
        return self.classifier(last_time_features)

# DeepSurv versions of models
class DeepSurvRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0), bidirectional=bidirectional
        )
        feature_dim = hidden_dim * self.num_directions
        self.risk = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, 2)
    
    def forward(self, x):
        _, h_n = self.rnn(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        if self.bidirectional:
            top_layer_features = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        else:
            top_layer_features = h_n[-1, 0, :, :]
        risk_pred = self.risk(top_layer_features).squeeze(-1)
        classification_logits = self.classifier(top_layer_features)
        return classification_logits, risk_pred

class DeepSurvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0), bidirectional=bidirectional
        )
        feature_dim = hidden_dim * self.num_directions
        self.risk = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, 2)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        if self.bidirectional:
            top_layer_features = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        else:
            top_layer_features = h_n[-1, 0, :, :]
        risk_pred = self.risk(top_layer_features).squeeze(-1)
        classification_logits = self.classifier(top_layer_features)
        return classification_logits, risk_pred

class DeepSurvTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        d_model = input_dim
        if d_model % nhead != 0:
            d_model = nhead * ((input_dim + nhead -1)//nhead)
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            self.input_projection = nn.Identity()
            
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.risk = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        last_token_features = out[:, -1, :]
        risk_pred = self.risk(last_token_features).squeeze(-1)
        classification_logits = self.classifier(last_token_features)
        return classification_logits, risk_pred

class DeepSurvMLP(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.flatten_dim = input_dim * window_size
        
        # Shared MLP part
        shared_layers = []
        current_dim = self.flatten_dim
        if num_layers < 1 : # No shared hidden layers if num_layers is 0 for shared part
            self.shared_net = nn.Identity()
            feature_dim_for_heads = current_dim
        else:
            for _ in range(num_layers): # num_layers here refers to number of hidden layers in shared part
                shared_layers.append(nn.Linear(current_dim, hidden_dim))
                shared_layers.append(nn.ReLU())
                shared_layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            self.shared_net = nn.Sequential(*shared_layers)
            feature_dim_for_heads = hidden_dim
            
        self.risk = nn.Linear(feature_dim_for_heads, 1)
        self.classifier = nn.Linear(feature_dim_for_heads, 2)
        
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1) # Flatten
        features = self.shared_net(x)
        risk_pred = self.risk(features).squeeze(-1)
        classification_logits = self.classifier(features)
        return classification_logits, risk_pred

class DeepSurvTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        tcn_layers = []
        num_channels_list = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = num_channels_list[i]
            out_ch = num_channels_list[i+1]
            padding = (kernel_size - 1) * dilation
            tcn_layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1, dilation=dilation,
                padding=padding, dropout=dropout
            ))
        self.tcn_network = nn.Sequential(*tcn_layers)
        self.risk = nn.Linear(hidden_dim, 1) # Output from TCN is hidden_dim
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        tcn_out = self.tcn_network(x)
        last_time_features = tcn_out[:, :, -1]
        risk_pred = self.risk(last_time_features).squeeze(-1)
        classification_logits = self.classifier(last_time_features)
        return classification_logits, risk_pred


# Cox PH Loss (as in the provided tabular script)
def cox_ph_loss(risk_scores, event_times, event_observed):
    """
    Simplified Cox PH loss. Assumes event_observed is 0 for censored, 1 for event.
    """
    device = risk_scores.device
    # Ensure event_observed is boolean for indexing where needed, but use original for sum
    event_observed_bool = event_observed.bool()

    if not torch.any(event_observed_bool):
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Sort by event_times in descending order
    desc_indices = torch.argsort(event_times, descending=True)
    risk_scores_sorted = risk_scores[desc_indices]
    # event_times_sorted = event_times[desc_indices] # Not directly used but good for debugging
    event_observed_sorted = event_observed[desc_indices]
    event_observed_bool_sorted = event_observed_bool[desc_indices] # Sorted boolean event indicators

    # Calculate log of sum of exp(risk_scores) for risk sets
    hazard_ratios = torch.exp(risk_scores_sorted)
    # log_sum_exp_terms = torch.logcumsumexp(risk_scores_sorted, dim=0) # Alternative if available and stable
    log_hazard_sum_terms = torch.log(torch.cumsum(hazard_ratios, dim=0) + 1e-9) # Denominators for each observation

    # Select only terms for actual events
    numerator_terms = risk_scores_sorted[event_observed_bool_sorted]
    denominator_terms = log_hazard_sum_terms[event_observed_bool_sorted]
    
    loss = -(numerator_terms - denominator_terms).sum()
    
    num_events = event_observed_sorted.sum() # Sum of 0s and 1s
    if num_events > 0:
        loss = loss / num_events
    else: # Should be caught by initial check
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    return loss


def concordance_index(event_times_np, predicted_scores_np, event_observed_np):
    """ Calculates Harrell's C-index """
    # Ensure inputs are numpy arrays
    event_times = np.asarray(event_times_np)
    predicted_scores = np.asarray(predicted_scores_np) # Higher score = higher risk
    event_observed = np.asarray(event_observed_np).astype(int)

    # Filter out NaNs
    nan_mask = np.isnan(event_times) | np.isnan(predicted_scores) | np.isnan(event_observed)
    if np.any(nan_mask):
        event_times = event_times[~nan_mask]
        predicted_scores = predicted_scores[~nan_mask]
        event_observed = event_observed[~nan_mask]

    if len(event_times) < 2: return 0.5

    concordant_pairs = 0
    permissible_pairs = 0 # Total comparable pairs

    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            # Check for permissible pairs
            if event_observed[i] == 1 and event_observed[j] == 1: # Both events
                if event_times[i] != event_times[j]:
                    permissible_pairs += 1
                    if event_times[i] < event_times[j]: # i failed before j
                        if predicted_scores[i] > predicted_scores[j]: concordant_pairs += 1
                        elif predicted_scores[i] == predicted_scores[j]: concordant_pairs += 0.5
                    else: # j failed before i
                        if predicted_scores[j] > predicted_scores[i]: concordant_pairs += 1
                        elif predicted_scores[j] == predicted_scores[i]: concordant_pairs += 0.5
            elif event_observed[i] == 1 and event_observed[j] == 0: # i event, j censored
                if event_times[i] < event_times[j]: # i failed before j was censored
                    permissible_pairs += 1
                    if predicted_scores[i] > predicted_scores[j]: concordant_pairs += 1
                    elif predicted_scores[i] == predicted_scores[j]: concordant_pairs += 0.5
            elif event_observed[i] == 0 and event_observed[j] == 1: # i censored, j event
                if event_times[j] < event_times[i]: # j failed before i was censored
                    permissible_pairs += 1
                    if predicted_scores[j] > predicted_scores[i]: concordant_pairs += 1
                    elif predicted_scores[j] == predicted_scores[i]: concordant_pairs += 0.5
            # If both censored, not a permissible pair for risk comparison

    return concordant_pairs / permissible_pairs if permissible_pairs > 0 else 0.5


def compute_metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    # Use zero_division=0 for precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    
    # Ensure confusion matrix can handle cases with only one class in labels or preds
    try:
        cm = confusion_matrix(labels, preds)
        if cm.size == 1: # Only one class present and predicted
             if labels[0] == 0 and preds[0] == 0: tn, fp, fn, tp = len(labels), 0, 0, 0
             elif labels[0] == 1 and preds[0] == 1: tn, fp, fn, tp = 0, 0, 0, len(labels)
             else: # Should not happen if cm.size == 1 based on above
                 tn, fp, fn, tp = 0,0,0,0 
        elif cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
        else: # Fallback for non-2x2 cm (e.g. only one class predicted over all samples)
            tp = np.sum((labels == 1) & (preds == 1))
            tn = np.sum((labels == 0) & (preds == 0))
            fp = np.sum((labels == 0) & (preds == 1))
            fn = np.sum((labels == 1) & (preds == 0))
    except ValueError: # If confusion_matrix itself fails (e.g. empty labels/preds)
        tp, tn, fp, fn = 0,0,0,0


    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Handle cases where labels might have only one class for ROC/PRC
    auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    auprc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else (np.mean(labels) if len(labels)>0 else 0.0)
    
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "ppv": ppv, "npv": npv, "auroc": auroc, "auprc": auprc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def bootstrap_metrics(labels, probs, threshold, n_boot=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = len(labels)
    if n_samples == 0:
        all_keys = ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]
        return {k: (np.nan, np.nan, np.nan) for k in all_keys}

    all_keys = ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]
    metric_samples = {k: [] for k in all_keys}

    for _ in range(n_boot):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_labels = labels[indices]
        sample_probs = probs[indices]
        
        if len(sample_labels) == 0: continue # Should not happen if n_samples > 0

        sample_result = compute_metrics_at_threshold(sample_labels, sample_probs, threshold)
        for k_single in all_keys:
            metric_samples[k_single].append(sample_result.get(k_single, np.nan))

    ci_results = {}
    for k_ci in all_keys:
        arr = np.array(metric_samples[k_ci])
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) < 2:
            lower, upper, meanv = np.nan, np.nan, np.nan
            if len(arr_clean) == 1: meanv = arr_clean[0]
        else:
            lower = np.percentile(arr_clean, 2.5)
            upper = np.percentile(arr_clean, 97.5)
            meanv = np.mean(arr_clean)
        ci_results[k_ci] = (meanv, lower, upper)
    return ci_results

# Adapted time_to_event_preprocessing for the tabular script
def time_to_event_preprocessing_tabular(meta_df_input, source_label_col='label_ckd_stage_4_plus',
                                   time_col_name='time_until_progression', 
                                   event_indicator_col_name='event_for_cox_indicator'):
    meta = meta_df_input.copy()
    meta = meta.sort_values(by=["PatientID", "EventDate"]).reset_index(drop=True)
    meta["EventDate"] = pd.to_datetime(meta["EventDate"])
    
    meta[time_col_name] = np.nan
    meta[event_indicator_col_name] = 0 # Initialize event indicator to 0 (censored)

    if source_label_col not in meta.columns:
        raise ValueError(f"Source label column '{source_label_col}' not found in metadata for TTE preprocessing.")

    for pid, group in meta.groupby("PatientID"):
        progression_event_indices = group[group[source_label_col] == 1].index
        
        if not progression_event_indices.empty:
            first_progression_actual_idx = progression_event_indices.min() # Actual index in `meta`
            progression_date = meta.loc[first_progression_actual_idx, "EventDate"]

            for current_visit_actual_idx in group.index:
                current_date = meta.loc[current_visit_actual_idx, "EventDate"]
                if current_date <= progression_date:
                    delta_days = (progression_date - current_date).days
                    meta.loc[current_visit_actual_idx, time_col_name] = float(delta_days)
                    if current_visit_actual_idx == first_progression_actual_idx:
                        meta.loc[current_visit_actual_idx, event_indicator_col_name] = 1 # Event occurred at this visit
                    # else: event_indicator remains 0 (censored or pre-event)
                # Visits after first progression: time_col remains NaN, event_indicator remains 0
        else:
            # No progression event for this patient, all visits are censored w.r.t this event type
            # Set time to last observation for this patient from each visit
            if not group.empty:
                last_observed_date_for_patient = group["EventDate"].max()
                for current_visit_actual_idx in group.index:
                    current_date = meta.loc[current_visit_actual_idx, "EventDate"]
                    delta_to_last_obs = (last_observed_date_for_patient - current_date).days
                    meta.loc[current_visit_actual_idx, time_col_name] = float(delta_to_last_obs)
                    # event_indicator_col_name remains 0
    return meta


# Adapted build_sequences for the new 1-year future label
def build_sequences(meta, window_size, target_label_col='label_ckd_1_year_future', feature_col='embedding'):
    sequence_data = []
    if target_label_col not in meta.columns:
        raise ValueError(f"Target label column '{target_label_col}' not found in metadata for build_sequences.")
    if feature_col not in meta.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in metadata for build_sequences.")

    for pid, group in meta.groupby("PatientID"):
        group = group.sort_values(by="EventDate").reset_index(drop=True) # Ensure consistent indexing within group
        
        # 'embedding' column now holds the list of features for each row
        feature_sequences = list(group[feature_col]) 
        classification_labels = list(group[target_label_col]) 

        for k in range(len(group)): # For each visit k
            context_end_idx = k + 1
            context_start_idx = max(0, k - window_size + 1)
            
            # context is a list of feature lists/arrays from previous time steps
            context = feature_sequences[context_start_idx : context_end_idx]
            target = classification_labels[k] # label_ckd_1_year_future for visit k
            
            if not context: 
                # This case should ideally not be hit if group is non-empty and window_size >= 1
                # For window_size=0 (current features only), context would be features_sequences[k:k+1]
                # Let's ensure context is never empty, if k=0 and window_size=1, context is features_sequences[0:1]
                logger.warning(f"Empty context for PatientID {pid}, visit index {k}. Skipping.")
                continue
            
            sequence_data.append((context, target, pid, k)) 
    return sequence_data

# Adapted build_sequences_for_multitask
def build_sequences_for_multitask(meta, window_size, 
                                  classification_target_col='label_ckd_1_year_future',
                                  tte_time_col='time_until_progression', # Was 'time_until_progression'
                                  # Removed tte_event_col as it's taken from classification_target_col by current DeepSurv train loop
                                  feature_col='embedding'):
    data_records = []
    required_cols = [classification_target_col, tte_time_col, feature_col]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Required column '{col}' not found for build_sequences_for_multitask.")

    for pid, group in meta.groupby("PatientID"):
        group = group.sort_values(by="EventDate").reset_index(drop=True)
        
        feature_sequences = list(group[feature_col])
        classification_labels = list(group[classification_target_col])
        tte_values = list(group[tte_time_col])

        for k in range(len(group)):
            context_end_idx = k + 1
            context_start_idx = max(0, k - window_size + 1)
            context = feature_sequences[context_start_idx : context_end_idx]
            
            classification_target = classification_labels[k]
            time_for_tte = tte_values[k] 
            # The 'event' for Cox loss in the existing train_and_evaluate_deepsurv is `event_batch`,
            # which will be `classification_target`.
            
            if not context: continue
            data_records.append((context, classification_target, pid, k, time_for_tte)) # 5 items
    return data_records


def train_and_evaluate_deepsurv(model, device, train_loader, val_loader, test_loader, args, model_name):
    logger.info(f"Starting {model_name} training (DeepSurv multi-task). Classification target: 'label_ckd_1_year_future'.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, verbose=False)
    classification_criterion = nn.CrossEntropyLoss()
    # aux_weight from original tabular script = 1.0, let's call it classification_loss_weight
    classification_loss_weight = 1.0 

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"
    os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)


    for epoch in range(args.epochs):
        model.train()
        train_total_losses, train_class_losses, train_surv_losses = [], [], []
        for batch in train_loader:
            optimizer.zero_grad()
            # context, classification_target, pid, local_idx, tte_value
            x_batch, classification_target_batch, _, _, tte_batch = batch # Unpack 5 items
            
            x_batch = x_batch.to(device)
            classification_target_batch = classification_target_batch.to(device) # This is used as 'event' for Cox
            tte_batch = tte_batch.to(device)
            
            logits, risk = model(x_batch) # Model returns (classification_logits, risk_pred_for_tte)
            
            # Survival loss for TTE (time_until_progression)
            # Mask for valid TTE data (non-NaN time and non-NaN risk)
            # The 'event' for this Cox loss is `classification_target_batch` (label_ckd_1_year_future)
            mask = ~torch.isnan(tte_batch) & ~torch.isnan(risk) & ~torch.isnan(classification_target_batch.float())

            surv_loss = torch.tensor(0.0, device=device)
            if mask.sum() > 0:
                # Ensure classification_target_batch is float for cox_ph_loss's event_observed
                surv_loss = cox_ph_loss(risk[mask], tte_batch[mask], classification_target_batch[mask].float()) 
            
            # Classification loss for the 1-year future label
            class_loss = classification_criterion(logits, classification_target_batch.long())
            
            total_loss = surv_loss + classification_loss_weight * class_loss # Combine losses
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN/Inf loss in {model_name} train epoch {epoch+1}. CL:{class_loss.item():.4f} SL:{surv_loss.item():.4f}. Skipping.")
                continue

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_total_losses.append(total_loss.item())
            train_class_losses.append(class_loss.item())
            train_surv_losses.append(surv_loss.item())

        # Validation phase
        model.eval()
        val_total_losses, val_class_losses, val_surv_losses = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, classification_target_batch, _, _, tte_batch = batch
                x_batch = x_batch.to(device)
                classification_target_batch = classification_target_batch.to(device)
                tte_batch = tte_batch.to(device)
                
                logits, risk = model(x_batch)
                
                mask = ~torch.isnan(tte_batch) & ~torch.isnan(risk) & ~torch.isnan(classification_target_batch.float())
                surv_loss_val = torch.tensor(0.0, device=device)
                if mask.sum() > 0:
                    surv_loss_val = cox_ph_loss(risk[mask], tte_batch[mask], classification_target_batch[mask].float())
                
                class_loss_val = classification_criterion(logits, classification_target_batch.long())
                total_loss_val = surv_loss_val + classification_loss_weight * class_loss_val
                
                if not (torch.isnan(total_loss_val) or torch.isinf(total_loss_val)):
                    val_total_losses.append(total_loss_val.item())
                    val_class_losses.append(class_loss_val.item())
                    val_surv_losses.append(surv_loss_val.item())

        avg_train_loss = np.nanmean(train_total_losses) if train_total_losses else float('inf')
        avg_train_cl = np.nanmean(train_class_losses) if train_class_losses else float('inf')
        avg_train_sl = np.nanmean(train_surv_losses) if train_surv_losses else float('inf')
        
        avg_val_loss = np.nanmean(val_total_losses) if val_total_losses else float('inf')
        avg_val_cl = np.nanmean(val_class_losses) if val_class_losses else float('inf')
        avg_val_sl = np.nanmean(val_surv_losses) if val_surv_losses else float('inf')

        logger.info(f"{model_name} Ep {epoch+1}/{args.epochs} -> Tr L: {avg_train_loss:.4f}(C:{avg_train_cl:.4f} S:{avg_train_sl:.4f}), "
                    f"Val L: {avg_val_loss:.4f}(C:{avg_val_cl:.4f} S:{avg_val_sl:.4f})")
        scheduler.step(avg_val_loss) # Step scheduler based on total validation loss

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"{model_name}: Val loss improved to {best_val_loss:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"{model_name}: Early stopping. No improvement for {args.patience} epochs.")
                break
            if np.isnan(avg_val_loss) and epoch > args.scheduler_patience:
                 logger.warning(f"{model_name}: Val loss is NaN. Early stopping.")
                 break


    if os.path.exists(best_model_path):
        logger.info(f"{model_name}: Loading best model for final evaluation: {best_model_path}")
        # Ensure map_location for loading if CUDA status differs from saving
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning(f"{model_name}: Best model not found at {best_model_path}. Using current model state.")

    model.eval()
    all_cl_logits_test, all_risk_scores_test = [], []
    all_classification_targets_test, all_tte_times_test = [], [] # TTE times from tte_batch

    with torch.no_grad():
        for batch in test_loader:
            x_batch, classification_target_b, _, _, tte_b = batch
            x_batch = x_batch.to(device)
            
            logits, risk = model(x_batch) # (classification_logits, risk_for_tte)
            
            all_cl_logits_test.extend(logits.cpu().numpy())
            all_risk_scores_test.extend(risk.cpu().numpy())
            all_classification_targets_test.extend(classification_target_b.numpy()) # This is label_ckd_1_year_future
            all_tte_times_test.extend(tte_b.numpy()) # This is time_until_progression

    all_cl_logits_test = np.array(all_cl_logits_test)
    all_risk_scores_test = np.array(all_risk_scores_test)
    all_classification_targets_test = np.array(all_classification_targets_test) # For classification metrics
    all_tte_times_test = np.array(all_tte_times_test) # For C-index with risk_scores

    # Classification metrics (based on label_ckd_1_year_future)
    all_cl_probs_test = nn.Softmax(dim=1)(torch.tensor(all_cl_logits_test)).numpy()[:, 1] # Prob of class 1
    
    final_results_dict = {"model_name": model_name}

    if len(all_classification_targets_test) > 0:
        prevalence = np.mean(all_classification_targets_test)
        logger.info(f"{model_name} Classification Test Prevalence ('label_ckd_1_year_future'): {prevalence:.4f}")
        threshold = prevalence if 0 < prevalence < 1 else 0.5
        
        cl_metrics_raw = compute_metrics_at_threshold(all_classification_targets_test, all_cl_probs_test, threshold)
        cl_metrics_ci = bootstrap_metrics(all_classification_targets_test, all_cl_probs_test, threshold, random_state=args.random_seed)
        
        logger.info(f"{model_name} Classification Threshold set to {threshold:.4f}.")
        for k_metric in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]:
            raw_val = cl_metrics_raw.get(k_metric, np.nan)
            ci_mean, ci_low, ci_high = cl_metrics_ci.get(k_metric, (np.nan, np.nan, np.nan))
            logger.info(f"{model_name} Class-{k_metric.upper()}: {raw_val:.4f} (CI Mean: {ci_mean:.4f} [{ci_low:.4f}-{ci_high:.4f}])")
            final_results_dict[f"class_{k_metric}"] = raw_val
    else:
        logger.warning(f"{model_name}: No classification targets in test set.")

    # Survival metric (C-index)
    # Event for C-index is all_classification_targets_test (label_ckd_1_year_future)
    # Time for C-index is all_tte_times_test (time_until_progression to actual CKD stage 4+)
    # Risk for C-index is all_risk_scores_test (from model's risk head)
    
    # Filter NaNs for C-index calculation (especially for time)
    valid_cindex_mask = ~np.isnan(all_tte_times_test) & \
                        ~np.isnan(all_risk_scores_test) & \
                        ~np.isnan(all_classification_targets_test) # Ensure target is not NaN
    
    if np.sum(valid_cindex_mask) > 1 and np.sum(all_classification_targets_test[valid_cindex_mask]) > 0 : # Need at least one event
        c_index_val = concordance_index(
            event_times_np=all_tte_times_test[valid_cindex_mask],
            predicted_scores_np=all_risk_scores_test[valid_cindex_mask],
            event_observed_np=all_classification_targets_test[valid_cindex_mask] # label_ckd_1_year_future as event
        )
        logger.info(f"{model_name} Concordance Index (Time: time_until_progression, Event: label_ckd_1_year_future, Risk: model_risk_head): {c_index_val:.4f}")
        final_results_dict["concordance_index_multitask"] = c_index_val
    else:
        logger.warning(f"{model_name}: Not enough valid data or events for C-index calculation.")
        final_results_dict["concordance_index_multitask"] = np.nan


    # Save detailed outputs
    output_dir = os.path.dirname(args.output_model_prefix) or "."
    os.makedirs(os.path.join(output_dir, "results_tabular_1yr"), exist_ok=True) # Specific folder
    
    df_output_details = pd.DataFrame({
        "cl_logit_0": all_cl_logits_test[:, 0] if all_cl_logits_test.ndim == 2 else np.nan,
        "cl_logit_1": all_cl_logits_test[:, 1] if all_cl_logits_test.ndim == 2 else np.nan,
        "cl_prob_1": all_cl_probs_test,
        "cl_true_label_1yr_future": all_classification_targets_test,
        "risk_score_tte": all_risk_scores_test,
        "time_for_tte": all_tte_times_test
    })
    df_output_details.to_csv(os.path.join(output_dir, "results_tabular_1yr", f"tab_{model_name}_deepsurv_outputs.csv"), index=False)
    logger.info(f"{model_name}: Detailed outputs saved to results_tabular_1yr/tab_{model_name}_deepsurv_outputs.csv")
    return final_results_dict


# train_and_evaluate for classification-only models
def train_and_evaluate(model, device, train_loader, val_loader, test_loader, args, model_name):
    logger.info(f"Starting {model_name} training (Classification Target: 'label_ckd_1_year_future').")
    model.to(device)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, verbose=False)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"
    os.makedirs(os.path.dirname(best_model_path) or ".", exist_ok=True)


    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader: # Standard classification loader
            optimizer.zero_grad()
            x_batch, y_batch, _, _ = batch # y_batch is label_ckd_1_year_future
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(x_batch) # Classification models return only logits
            loss = classification_criterion(logits, y_batch.long())
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss in {model_name} train epoch {epoch+1}. Skipping.")
                continue
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch, _, _ = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss_val = classification_criterion(logits, y_batch.long())
                if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                    val_losses.append(loss_val.item())
        
        avg_train_loss = np.nanmean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.nanmean(val_losses) if val_losses else float('inf')
        logger.info(f"{model_name} Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss) :
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"{model_name}: Val loss improved to {best_val_loss:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"{model_name}: Early stopping.")
                break
            if np.isnan(avg_val_loss) and epoch > args.scheduler_patience:
                 logger.warning(f"{model_name}: Val loss is NaN. Early stopping.")
                 break

    if os.path.exists(best_model_path):
        logger.info(f"{model_name}: Loading best model: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning(f"{model_name}: Best model not found. Using current model state.")

    model.eval()
    all_cl_probs_test, all_cl_targets_test = [], []
    all_cl_logits_test = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch, _, _ = batch
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = nn.Softmax(dim=1)(logits)
            all_cl_probs_test.extend(probs[:, 1].cpu().numpy()) # Prob of class 1
            all_cl_targets_test.extend(y_batch.numpy())
            all_cl_logits_test.extend(logits.cpu().numpy())
            
    all_cl_targets_test = np.array(all_cl_targets_test)
    all_cl_probs_test = np.array(all_cl_probs_test)
    all_cl_logits_test = np.array(all_cl_logits_test)
    
    final_results_dict = {"model_name": model_name}
    if len(all_cl_targets_test) > 0:
        prevalence = np.mean(all_cl_targets_test)
        logger.info(f"{model_name} Classification Test Prevalence ('label_ckd_1_year_future'): {prevalence:.4f}")
        threshold = prevalence if 0 < prevalence < 1 else 0.5
        
        raw_metrics = compute_metrics_at_threshold(all_cl_targets_test, all_cl_probs_test, threshold)
        ci_dict = bootstrap_metrics(all_cl_targets_test, all_cl_probs_test, threshold, random_state=args.random_seed)
        
        logger.info(f"{model_name} Classification Threshold set to {threshold:.4f}.")
        for k_metric in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]:
            raw_val = raw_metrics.get(k_metric, np.nan)
            ci_mean, ci_low, ci_high = ci_dict.get(k_metric, (np.nan, np.nan, np.nan))
            logger.info(f"{model_name} {k_metric.upper()}: {raw_val:.4f} (CI Mean: {ci_mean:.4f} [{ci_low:.4f}-{ci_high:.4f}])")
            final_results_dict[k_metric] = raw_val
    else:
        logger.warning(f"{model_name}: No classification targets in test set for evaluation.")

    output_dir = os.path.dirname(args.output_model_prefix) or "."
    os.makedirs(os.path.join(output_dir, "results_tabular_1yr"), exist_ok=True)
    df_output_details = pd.DataFrame({
        "logit_0": all_cl_logits_test[:, 0] if all_cl_logits_test.ndim == 2 else np.nan,
        "logit_1": all_cl_logits_test[:, 1] if all_cl_logits_test.ndim == 2 else np.nan,
        "prob_positive_1yr_future": all_cl_probs_test,
        "true_label_1yr_future": all_cl_targets_test
    })
    df_output_details.to_csv(os.path.join(output_dir, "results_tabular_1yr", f"tab_{model_name}_classification_outputs.csv"), index=False)
    logger.info(f"{model_name}: Detailed classification outputs saved to results_tabular_1yr/tab_{model_name}_classification_outputs.csv")
    
    return final_results_dict


# predict_label_switches and analyze_switches from the original tabular script
# They will use the new `label_ckd_1_year_future` as the "TrueLabel" implicitly via the dataloaders
def predict_label_switches(model, loader, device, is_multitask_model=False): # Added is_multitask_model flag
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            if is_multitask_model: # DeepSurv type model
                x_batch, y_batch, pid_batch, idx_batch, _ = batch # 5 items
                logits, _ = model(x_batch.to(device)) # Get classification logits
            else: # Classification only model
                x_batch, y_batch, pid_batch, idx_batch = batch # 4 items
                logits = model(x_batch.to(device))
                
            probs = nn.Softmax(dim=1)(logits)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Ensure pid_batch and idx_batch are correctly iterated (they might be tensors or lists)
            pid_list = pid_batch if isinstance(pid_batch, list) else pid_batch.tolist()
            idx_list = idx_batch if isinstance(idx_batch, list) else idx_batch.tolist()

            for pid, true_label, pred_label, local_idx in zip(pid_list, y_batch.numpy(), preds, idx_list):
                records.append((pid, local_idx, true_label, pred_label))
                
    # The label columns will reflect the 1-year future target now
    df = pd.DataFrame(records, columns=["PatientID", "LocalIndex", f"TrueLabel_{args.prediction_horizon_days}DayFuture", f"PredLabel_{args.prediction_horizon_days}DayFuture"])
    return df

def analyze_switches(df_preds_Nday, N_days_horizon): # Renamed to match previous script
    analysis_records = []
    true_label_col = f"TrueLabel_{N_days_horizon}DayFuture"
    pred_label_col = f"PredLabel_{N_days_horizon}DayFuture"

    if not {true_label_col, pred_label_col}.issubset(df_preds_Nday.columns):
        logger.error(f"Required columns for switch analysis missing: Need {true_label_col}, {pred_label_col}. Got {df_preds_Nday.columns}")
        return pd.DataFrame()

    for pid, group_df in df_preds_Nday.groupby("PatientID"):
        group_df = group_df.sort_values("LocalIndex").reset_index(drop=True)
        
        true_first_Nday_event_idx = group_df[group_df[true_label_col] == 1]["LocalIndex"].min()
        pred_first_Nday_event_idx = group_df[group_df[pred_label_col] == 1]["LocalIndex"].min()
        
        analysis_records.append({
            "PatientID": pid,
            f"TrueFirst{N_days_horizon}DayEventIdx": true_first_Nday_event_idx if pd.notna(true_first_Nday_event_idx) else None,
            f"PredFirst{N_days_horizon}DayEventIdx": pred_first_Nday_event_idx if pd.notna(pred_first_Nday_event_idx) else None
        })
    analysis_df = pd.DataFrame(analysis_records)
    
    true_col_name = f"TrueFirst{N_days_horizon}DayEventIdx"
    pred_col_name = f"PredFirst{N_days_horizon}DayEventIdx"
    analysis_df[f"SwitchDifference_{N_days_horizon}DayFuture"] = analysis_df[pred_col_name] - analysis_df[true_col_name]
    return analysis_df
    
def main():
    global args # Make args global for predict_label_switches to access prediction_horizon_days
    args = parse_args()
    logger.info("Running tabular CKD script with 1-year future labels and all features:")
    # for key, val in vars(args).items():
    #     logger.info(f"{key}: {val}")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    logger.info(f"Loading tabular CKD data from: {args.tabular_data_file}")
    try:
        metadata = pd.read_csv(args.tabular_data_file, parse_dates=["EventDate"])
    except FileNotFoundError:
        logger.error(f"Tabular data file not found: {args.tabular_data_file}. Exiting.")
        return
        
    metadata = metadata.sort_values(by=['PatientID', 'EventDate']).reset_index(drop=True)

    # CKD Stage Cleaning (as in original tabular script, adapted)
    if 'CKD_stage' in metadata.columns:
        metadata['CKD_stage_cleaned'] = metadata['CKD_stage'].apply(clean_ckd_stage)
        # Fill missing stages within a patient's record
        metadata['CKD_stage_cleaned'] = metadata.groupby('PatientID')['CKD_stage_cleaned'].bfill().ffill()
        metadata = metadata.dropna(subset=['CKD_stage_cleaned']) # Remove patients with no stage info
        metadata['CKD_stage_cleaned'] = metadata['CKD_stage_cleaned'].astype(int)
    else:
        logger.error("'CKD_stage' column not found in tabular data. Cannot proceed with label generation.")
        return

    # Label 1: Current CKD stage >= 4
    metadata['label_ckd_stage_4_plus'] = metadata['CKD_stage_cleaned'].apply(lambda x: 1 if x >= 4 else 0)
    logger.info(f"Value counts for 'label_ckd_stage_4_plus':\n{metadata['label_ckd_stage_4_plus'].value_counts(dropna=False).to_string()}")


    # Time-to-event preprocessing (using the adapted tabular version)
    logger.info("Preprocessing time-to-event data (time_until_progression to first CKD stage 4+).")
    metadata = time_to_event_preprocessing_tabular(metadata, source_label_col='label_ckd_stage_4_plus',
                                             time_col_name='time_until_progression',
                                             event_indicator_col_name='event_for_cox_indicator') # This event_indicator is not directly used by current deepsurv loop

    # Label 2: CKD stage 4+ within 1 year (args.prediction_horizon_days)
    metadata = add_future_event_label_column(
        metadata,
        source_label_col='label_ckd_stage_4_plus',
        new_label_col='label_ckd_1_year_future',
        horizon_days=args.prediction_horizon_days
    )
    logger.info(f"Value counts for 'label_ckd_1_year_future':\n{metadata['label_ckd_1_year_future'].value_counts(dropna=False).to_string()}")
    logger.info(f"Metadata with new labels (first 3 rows):\n{metadata.head(3).to_string()}")

    # Feature Selection: Use all columns except identifiers, raw date/stage, and created labels/TTE info
    exclude_cols = ['PatientID', 'EventDate', 'CKD_stage', # Raw stage column
                    'CKD_stage_cleaned', # Intermediate cleaned stage
                    'label_ckd_stage_4_plus', 'label_ckd_1_year_future', # Generated labels
                    'time_until_progression', 'event_for_cox_indicator'] # Generated TTE info
    
    # Also consider other known non-feature columns from the original script if any (e.g. 'GFR_combined')
    if 'GFR_combined' in metadata.columns: # Example if it was a target or identifier
         exclude_cols.append('GFR_combined')

    potential_feature_cols = metadata.columns.difference(exclude_cols)
    feature_cols = [col for col in potential_feature_cols if metadata[col].dtype in [np.number, 'bool']] # Keep only numeric/boolean
    
    # Convert boolean columns to int (0 or 1)
    for col in feature_cols:
        if metadata[col].dtype == 'bool':
            metadata[col] = metadata[col].astype(int)
            
    if not feature_cols:
        logger.error("No feature columns selected. Check data and exclude_cols list. Exiting.")
        return

    logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")
    
    # Dynamically set embed_dim based on selected features
    args.embed_dim = len(feature_cols)
    logger.info(f"Using args.embed_dim = {args.embed_dim} (number of selected features).")

    # Create the 'embedding' column (list of features)
    metadata['embedding'] = metadata[feature_cols].values.tolist()
    # Ensure all elements within the 'embedding' lists are numeric, handle NaNs
    metadata['embedding'] = metadata['embedding'].apply(
        lambda x: [float(val) if pd.notna(val) else 0.0 for val in x]
    )


    if args.max_patients is not None:
        unique_pids_initial = sorted(metadata['PatientID'].unique())
        if args.max_patients < len(unique_pids_initial):
            subset_pids = unique_pids_initial[:args.max_patients]
            metadata = metadata[metadata['PatientID'].isin(subset_pids)].reset_index(drop=True)
            logger.info(f"Filtered to {args.max_patients} patients. Rows: {len(metadata)}")


    logger.info("Creating train/val/test splits by PatientID.")
    unique_patients = metadata['PatientID'].unique()
    if len(unique_patients) < 3: # Need at least 3 patients for train/val/test
        logger.error(f"Not enough unique patients ({len(unique_patients)}) to create train/val/test splits. Exiting.")
        return

    train_patients, temp_patients = train_test_split(unique_patients, test_size=0.3, random_state=args.random_seed) # 70% train
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=args.random_seed) # 15% val, 15% test


    train_metadata = metadata[metadata['PatientID'].isin(train_patients)].copy()
    val_metadata = metadata[metadata['PatientID'].isin(val_patients)].copy()
    test_metadata = metadata[metadata['PatientID'].isin(test_patients)].copy()
    logger.info(f"Data split: Train PIDs={len(train_patients)} ({len(train_metadata)} recs), Val PIDs={len(val_patients)} ({len(val_metadata)} recs), Test PIDs={len(test_patients)} ({len(test_metadata)} recs)")


    # Fill any remaining NaNs in TTE columns after splits (e.g., for patients with no progression)
    # TTE values (time_until_progression) that are NaN because a patient never progressed (or progressed after last obs)
    # might need specific handling (e.g. fill with a large number if CoxPH requires non-NaN, or ensure masking handles it)
    # For now, the dataloader's getitem handles NaN tte_val.
    # The `time_to_event_preprocessing_tabular` already handles this for censored patients by giving time to last obs.
    # Nan-filling for feature columns was done when creating 'embedding' list.

    logger.info("Building sequence datasets for classification (Target: 'label_ckd_1_year_future').")
    train_sequences_class = build_sequences(train_metadata, args.window_size, target_label_col='label_ckd_1_year_future', feature_col='embedding')
    val_sequences_class = build_sequences(val_metadata, args.window_size, target_label_col='label_ckd_1_year_future', feature_col='embedding')
    test_sequences_class = build_sequences(test_metadata, args.window_size, target_label_col='label_ckd_1_year_future', feature_col='embedding')

    train_dataset_class = CKDSequenceDataset(train_sequences_class, args.window_size, args.embed_dim, use_survival_multitask=False)
    val_dataset_class = CKDSequenceDataset(val_sequences_class, args.window_size, args.embed_dim, use_survival_multitask=False)
    test_dataset_class = CKDSequenceDataset(test_sequences_class, args.window_size, args.embed_dim, use_survival_multitask=False)

    train_loader_class = DataLoader(train_dataset_class, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False)
    val_loader_class = DataLoader(val_dataset_class, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False)
    test_loader_class = DataLoader(test_dataset_class, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False)
    logger.info(f"Classification Datasets: Train={len(train_dataset_class)}, Val={len(val_dataset_class)}, Test={len(test_dataset_class)} sequences.")


    logger.info("Building sequence datasets for DeepSurv multi-task.")
    train_sequences_surv = build_sequences_for_multitask(train_metadata, args.window_size, 
                                                          classification_target_col='label_ckd_1_year_future',
                                                          tte_time_col='time_until_progression',
                                                          feature_col='embedding')
    val_sequences_surv = build_sequences_for_multitask(val_metadata, args.window_size,
                                                        classification_target_col='label_ckd_1_year_future',
                                                        tte_time_col='time_until_progression',
                                                        feature_col='embedding')
    test_sequences_surv = build_sequences_for_multitask(test_metadata, args.window_size,
                                                         classification_target_col='label_ckd_1_year_future',
                                                         tte_time_col='time_until_progression',
                                                         feature_col='embedding')

    train_dataset_surv = CKDSequenceDataset(train_sequences_surv, args.window_size, args.embed_dim, use_survival_multitask=True)
    val_dataset_surv = CKDSequenceDataset(val_sequences_surv, args.window_size, args.embed_dim, use_survival_multitask=True)
    test_dataset_surv = CKDSequenceDataset(test_sequences_surv, args.window_size, args.embed_dim, use_survival_multitask=True)

    train_loader_surv = DataLoader(train_dataset_surv, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False)
    val_loader_surv = DataLoader(val_dataset_surv, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False)
    test_loader_surv = DataLoader(test_dataset_surv, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False)
    logger.info(f"Multi-task (DeepSurv) Datasets: Train={len(train_dataset_surv)}, Val={len(val_dataset_surv)}, Test={len(test_dataset_surv)} sequences.")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    # Model Definitions (ensure input_dim matches args.embed_dim)
    cl_models_defs = {
        "RNN": lambda: LongitudinalRNN(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "LSTM": lambda: LongitudinalLSTM(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "Transformer": lambda: LongitudinalTransformer(args.embed_dim, args.num_layers, args.transformer_nhead, args.transformer_dim_feedforward, args.transformer_dropout),
        "MLP": lambda: MLPSimple(args.embed_dim, args.window_size, args.hidden_dim, args.num_layers, args.rnn_dropout),
        "TCN": lambda: TCN(args.embed_dim, args.hidden_dim, args.num_layers, kernel_size=3, dropout=args.rnn_dropout)
    }
    ds_models_defs = {
        "DeepSurv_RNN": lambda: DeepSurvRNN(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "DeepSurv_LSTM": lambda: DeepSurvLSTM(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "DeepSurv_Transformer": lambda: DeepSurvTransformer(args.embed_dim, args.num_layers, args.transformer_nhead, args.transformer_dim_feedforward, args.transformer_dropout),
        "DeepSurv_MLP": lambda: DeepSurvMLP(args.embed_dim, args.window_size, args.hidden_dim, args.num_layers, args.rnn_dropout),
        "DeepSurv_TCN": lambda: DeepSurvTCN(args.embed_dim, args.hidden_dim, args.num_layers, kernel_size=3, dropout=args.rnn_dropout)
    }
    
    all_cl_results, all_ds_results = [], []
    trained_cl_models, trained_ds_models = {}, {} # To store trained models for switch analysis

    model_name_suffix = f"Tabular_{args.prediction_horizon_days}DayFutureTarget" # For filenames

    if args.epochs > 0:
        logger.info("--- Training Classification-Only Models (Tabular Data) ---")
        if len(train_loader_class) > 0:
            for name, model_fn in cl_models_defs.items():
                model_instance = model_fn()
                results = train_and_evaluate(model_instance, device, train_loader_class, val_loader_class, test_loader_class, args, f"{name}_{model_name_suffix}")
                all_cl_results.append(results)
                trained_cl_models[name] = model_instance
        else: logger.warning("Skipping classification model training due to empty train_loader_class.")

        logger.info("--- Training DeepSurv Multi-Task Models (Tabular Data) ---")
        if len(train_loader_surv) > 0:
            for name, model_fn in ds_models_defs.items():
                model_instance = model_fn()
                results = train_and_evaluate_deepsurv(model_instance, device, train_loader_surv, val_loader_surv, test_loader_surv, args, f"{name}_{model_name_suffix}")
                all_ds_results.append(results)
                trained_ds_models[name] = model_instance
        else: logger.warning("Skipping DeepSurv model training due to empty train_loader_surv.")

    else:
        logger.info("Skipping all model training as epochs is 0.")

    # Summarize Results
    logger.info(f"\n--- Summary: Final Test Metrics (Classification Models - Tabular, Target: 'label_ckd_1_year_future') ---")
    for res in all_cl_results:
        if res and isinstance(res, dict):
            log_line = f"Model={res.get('model_name', 'N/A')} "
            for met in ["auroc", "auprc", "f1", "accuracy", "precision", "recall", "ppv", "npv"]:
                log_line += f"{met.upper()}={res.get(met, float('nan')):.4f} "
            logger.info(log_line)

    logger.info(f"\n--- Summary: Final Test Metrics (DeepSurv Multi-Task Models - Tabular, Target: 'label_ckd_1_year_future') ---")
    for res_ds in all_ds_results:
        if res_ds and isinstance(res_ds, dict):
            log_line_ds = f"Model={res_ds.get('model_name', 'N/A')} "
            for met_ds in ["class_auroc", "class_auprc", "class_f1", "class_accuracy"]: 
                log_line_ds += f"{met_ds.upper()}={res_ds.get(f'class_{met_ds}', res_ds.get(met_ds, float('nan'))):.4f} " # Check for 'class_' prefix
            log_line_ds += f"C_INDEX_MULTITASK={res_ds.get('concordance_index_multitask', float('nan')):.4f}"
            logger.info(log_line_ds)
    
    # Analyze Switches
    logger.info(f"\n--- Analyzing First Prediction of {args.prediction_horizon_days}-Day Future Event on Test Set (Tabular Models) ---")
    all_switch_dfs_tabular = []
    output_dir_results = os.path.join(os.path.dirname(args.output_model_prefix) or ".", "results_tabular_1yr")


    if args.epochs > 0: # Only if models were trained
        if len(test_loader_class) > 0 :
            for name, model_obj in trained_cl_models.items():
                df_preds = predict_label_switches(model_obj, test_loader_class, device, is_multitask_model=False)
                if not df_preds.empty:
                    df_sw = analyze_switches(df_preds, args.prediction_horizon_days) # Use updated analyze_switches
                    df_sw["ModelType"] = f"{name}_{model_name_suffix}" # Use full model name
                    all_switch_dfs_tabular.append(df_sw)
                    logger.info(f"{args.prediction_horizon_days}-Day Future Event Pred Switch Analysis for {name}:\n{df_sw.head(3).to_string()}")
                    valid_diffs = df_sw[f"SwitchDifference_{args.prediction_horizon_days}DayFuture"].dropna()
                    if not valid_diffs.empty: logger.info(f"{name} - Mean SwitchDiff: {valid_diffs.mean():.2f}, Median: {valid_diffs.median():.2f} visits")
        
        if len(test_loader_surv) > 0:
            for name, model_obj in trained_ds_models.items():
                df_preds_ds = predict_label_switches(model_obj, test_loader_surv, device, is_multitask_model=True)
                if not df_preds_ds.empty:
                    df_sw_ds = analyze_switches(df_preds_ds, args.prediction_horizon_days)
                    df_sw_ds["ModelType"] = f"{name}_{model_name_suffix}" # Use full model name
                    all_switch_dfs_tabular.append(df_sw_ds)
                    logger.info(f"{args.prediction_horizon_days}-Day Future Event Pred Switch Analysis for {name}:\n{df_sw_ds.head(3).to_string()}")
                    valid_diffs_ds = df_sw_ds[f"SwitchDifference_{args.prediction_horizon_days}DayFuture"].dropna()
                    if not valid_diffs_ds.empty: logger.info(f"{name} - Mean SwitchDiff: {valid_diffs_ds.mean():.2f}, Median: {valid_diffs_ds.median():.2f} visits")

    if all_switch_dfs_tabular:
        combined_sw_df_tabular = pd.concat(all_switch_dfs_tabular, ignore_index=True)
        sw_out_path_tabular = os.path.join(output_dir_results, f"all_tabular_models_{args.prediction_horizon_days}day_future_switch_analysis.csv")
        combined_sw_df_tabular.to_csv(sw_out_path_tabular, index=False)
        logger.info(f"Combined TABULAR {args.prediction_horizon_days}-day future switch analysis saved to: {sw_out_path_tabular}")


    logger.info("Tabular script finished.")

if __name__ == "__main__":
    main()
