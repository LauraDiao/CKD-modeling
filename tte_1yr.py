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
from datetime import timedelta # Import timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("full_deepserv_tte_1year_future.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CKD classification (1-year future window) and time-to-event training with DeepSurv models.")
    parser.add_argument("--embedding-root", type=str, default="./ckd_embeddings_10", help="Path to embeddings.")
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
    parser.add_argument("--output-model-prefix", type=str, default="best_model_1yr_future", help="Filename prefix for saved models.")
    parser.add_argument("--log-tte", action="store_true", help="Apply log transformation to time-to-event targets for Cox loss.")
    parser.add_argument("--prediction-horizon-days", type=int, default=365, help="Number of days into the future to check for an event for label generation.")
    parser.add_argument("--cox-loss-weight", type=float, default=1.0, help="Weight for the Cox PH loss in DeepSurv models.")
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
    if not seq: # Handle empty sequence
        return np.zeros((length, dim), dtype=np.float32)
    if len(seq) < length:
        padding = [np.zeros(dim, dtype=np.float32)] * (length - len(seq)) # ensure float32 for padding
        seq = padding + seq # Pre-pend padding
    # Ensure all elements in seq are numpy arrays of the correct dim before stacking
    processed_seq = []
    for item in seq[-length:]:
        if isinstance(item, np.ndarray) and item.shape == (dim,):
            processed_seq.append(item)
        else: # Fallback if an item is not as expected (e.g. corrupted data)
            logger.warning(f"Unexpected item type or shape in sequence for padding. Expected ({dim},), got {type(item)} {getattr(item, 'shape', 'N/A')}. Using zeros.")
            processed_seq.append(np.zeros(dim, dtype=np.float32))

    return np.stack(processed_seq, axis=0)


class CKDSequenceDataset(Dataset):
    def __init__(self, data, window_size, embed_dim, is_multitask_data=False):
        self.data = data
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.is_multitask_data = is_multitask_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item[0]
        # This 'label' is always the primary classification target (e.g., 1-year future event)
        classification_target_label = item[1]
        pid = item[2]
        local_idx = item[3]

        context_padded = pad_sequence(list(context), self.window_size, self.embed_dim)

        if self.is_multitask_data:
            # item: (context, classification_target_label, pid, local_idx, tte_for_cox, event_for_cox)
            tte_for_cox = item[4]
            event_for_cox = item[5] # This is the E for Cox loss
            return (
                torch.tensor(context_padded, dtype=torch.float32),
                torch.tensor(classification_target_label, dtype=torch.long), # For classification loss
                pid, # Keep as is, not converted to tensor
                local_idx, # Keep as is
                torch.tensor(tte_for_cox if pd.notna(tte_for_cox) else float('nan'), dtype=torch.float32), # T_cox
                torch.tensor(event_for_cox, dtype=torch.float32) # E_cox (should be float for loss)
            )
        else:
            # item: (context, classification_target_label, pid, local_idx)
            return (
                torch.tensor(context_padded, dtype=torch.float32),
                torch.tensor(classification_target_label, dtype=torch.long), # For classification loss
                pid, # Keep as is
                local_idx # Keep as is
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
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        # Reshape to (num_layers, num_directions, batch, hidden_dim)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        # Get the hidden state of the last layer: h_n[-1] -> (num_directions, batch, hidden_dim)
        # Concatenate the hidden states from both directions if bidirectional
        if self.bidirectional:
            # h_n[-1, 0] is forward, h_n[-1, 1] is backward
            top_layer = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        else:
            top_layer = h_n[-1, 0, :, :] # (batch, hidden_dim)
        # The original view operation was also fine, this is just an alternative way
        # top_layer_orig = h_n[-1].transpose(0, 1).contiguous().view(x.size(0), -1)
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
        _, (h_n, _) = self.lstm(x) # h_n shape: (num_layers * num_directions, batch, hidden_dim)
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
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # shape (1, max_len, d_model)

    def forward(self, x): # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # Adjust self.pe if it's on a different device than x
        pe_to_add = self.pe[:, :seq_len, :].to(x.device)
        return x + pe_to_add

class LongitudinalTransformer(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Important: input format is (batch, seq_len, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 2) # Classify based on the representation of the last token

    def forward(self, x): # x shape: (batch, seq_len, input_dim)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x) # out shape: (batch, seq_len, input_dim)
        # Use the output of the last token in the sequence for classification
        # This is a common approach, akin to using the [CLS] token's output in BERT.
        last_token_features = out[:, -1, :] # shape: (batch, input_dim)
        return self.classifier(last_token_features)

class MLPSimple(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.flatten_dim = input_dim * window_size
        layers = []
        current_dim = self.flatten_dim
        # Ensure num_layers corresponds to number of hidden layers + output layer
        # Or, if num_layers is total layers including input, adjust loop
        # Let's assume num_layers is the number of Linear layers.
        if num_layers < 1:
            raise ValueError("MLPSimple must have at least one layer.")
        if num_layers == 1:
            layers.append(nn.Linear(current_dim, 2)) # Direct to output
        else:
            for _ in range(num_layers - 1): # Hidden layers
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, 2)) # Output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x): # x shape: (batch, window_size, input_dim)
        x = x.view(x.size(0), -1) # Flatten the sequence: (batch, window_size * input_dim)
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels) # Changed from WeightNorm for potential stability
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels) # Changed from WeightNorm
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        # Downsample layer for residual connection if dimensions mismatch
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU() # Final ReLU for the block

    def forward(self, x): # x shape: (batch, in_channels, seq_len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # Ensure `out` and `res` have the same sequence length before adding.
        # This is crucial if padding in conv layers doesn't perfectly preserve length.
        # For causal convolutions, output length might be same if padding is (kernel_size-1)*dilation.
        # If not, slicing `out` might be needed: `out = out[:, :, :res.size(2)]`
        # Or adjust padding in conv1d. For simplicity, assume padding handles this.
        if out.size(2) != res.size(2): # Dynamic adjustment if lengths differ
            # This can happen if padding is not perfectly 'causal' or 'same' for all layers/dilations
            # A common strategy is to pad `res` or truncate `out`. Truncating `out` is often simpler.
            # logger.debug(f"TCN Residual size mismatch: out {out.shape}, res {res.shape}. Truncating out.")
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]

        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_channels_list = [input_dim] + [hidden_dim] * num_layers # [input_dim, hidden, hidden, ...]

        for i in range(num_layers):
            dilation_size = 2**i
            in_channels = num_channels_list[i]
            out_channels = num_channels_list[i+1]
            # Causal padding: (kernel_size - 1) * dilation
            # This padding is applied to the left side for causality.
            # PyTorch Conv1D padding parameter means padding on both sides if single int.
            # For causal, often use asymmetric padding (manual or nn.ConstantPad1d before conv)
            # Or, ensure (kernel_size - 1) * dilation is the padding value for 'same' like behavior if non-causal.
            # Let's assume the padding parameter in TemporalBlock is handled correctly for causality if desired.
            # Standard 'padding' in nn.Conv1d adds to both sides.
            # For causal, it should be ( (kernel_size-1)*dilation, 0 ).
            # Or, set padding=(kernel_size-1)*dilation // 2 for 'same' if stride=1, non-causal.
            # Given the original TemporalBlock code, it seems like `padding` argument refers to symmetric padding.
            # A common TCN padding is causal: pad `(kernel_size-1)*dilation` on the left.
            # If using standard Conv1d with an integer `padding`, it's symmetric.
            # For now, we use the padding calculation as in original code, assuming it's intended for the block.
            padding_val = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding_val, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, 2) # Output from last layer's hidden_dim

    def forward(self, x): # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1) # Convert to (batch, input_dim, seq_len) for Conv1d
        out = self.network(x)  # Output shape: (batch, hidden_dim, seq_len)
        # Use features from the last time step of the output sequence
        last_time_features = out[:, :, -1] # shape: (batch, hidden_dim)
        return self.classifier(last_time_features)


##############################################
# DeepSurv Model Definitions
# All these models will have a 'risk' head for TTE (Cox loss)
# and a 'classifier' head for the 1-year future event classification.
##############################################

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
        # Shared representation layer output_dim
        feature_dim = hidden_dim * self.num_directions
        self.risk = nn.Linear(feature_dim, 1) # For Cox loss (time-to-first-progression)
        self.classifier = nn.Linear(feature_dim, 2) # For 1-year future classification

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
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Features from transformer output (last token)
        feature_dim = input_dim
        self.risk = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x):
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
        
        shared_layers = []
        current_dim = self.flatten_dim
        if num_layers < 1: raise ValueError("DeepSurvMLP must have at least one effective layer in shared part if num_layers > 0 for shared.")
        # If num_layers = 1, shared part goes from current_dim to hidden_dim
        # If num_layers = 0, current_dim is used directly for heads (no shared hidden layers)

        if num_layers > 0: # Create shared hidden layers
            for i in range(num_layers):
                # Last shared layer outputs hidden_dim, or current_dim if only one shared layer to hidden_dim
                out_feat_shared = hidden_dim
                shared_layers.append(nn.Linear(current_dim, out_feat_shared))
                shared_layers.append(nn.ReLU())
                shared_layers.append(nn.Dropout(dropout))
                current_dim = out_feat_shared
            self.shared_net = nn.Sequential(*shared_layers)
            feature_dim_for_heads = hidden_dim
        else: # No shared hidden layers, use flattened input directly
            self.shared_net = nn.Identity() # Pass input through
            feature_dim_for_heads = self.flatten_dim


        self.risk = nn.Linear(feature_dim_for_heads, 1)
        self.classifier = nn.Linear(feature_dim_for_heads, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        features = self.shared_net(x) # Pass through shared layers (or identity)

        risk_pred = self.risk(features).squeeze(-1)
        classification_logits = self.classifier(features)
        return classification_logits, risk_pred

class DeepSurvTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3, dropout=0.2):
        super().__init__()
        tcn_layers_list = []
        num_channels_tcn = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            dilation = 2**i
            in_ch = num_channels_tcn[i]
            out_ch = num_channels_tcn[i+1]
            padding = (kernel_size - 1) * dilation # For causal conv
            tcn_layers_list.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1, dilation=dilation,
                padding=padding, dropout=dropout
            ))
        self.tcn_network = nn.Sequential(*tcn_layers_list)
        
        # Features from TCN output (last time step)
        feature_dim = hidden_dim # TCN outputs hidden_dim channels
        self.risk = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x): # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1) # (batch, input_dim, seq_len)
        tcn_out_features = self.tcn_network(x) # (batch, hidden_dim, seq_len)
        last_time_features = tcn_out_features[:, :, -1] # (batch, hidden_dim)

        risk_pred = self.risk(last_time_features).squeeze(-1)
        classification_logits = self.classifier(last_time_features)
        return classification_logits, risk_pred


##############################################
# Loss and Evaluation Functions
##############################################

def cox_ph_loss(risk_scores, event_times, event_observed):
    """
    Compute the negative log partial likelihood for Cox proportional hazards.
    Handles batches with no events.
    Args:
        risk_scores (torch.Tensor): Predicted risk scores from the model. Higher score = higher risk.
        event_times (torch.Tensor): Time to event or censoring.
        event_observed (torch.Tensor): Event indicator (1 if event, 0 if censored). Float type.
    Returns:
        torch.Tensor: The calculated Cox loss for the batch.
    """
    device = risk_scores.device
    # Ensure event_observed is boolean for indexing, but keep original float for calculations if needed by sum()
    event_observed_bool = event_observed.bool()

    if not torch.any(event_observed_bool): # No events in this batch
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Sort by event_times in descending order (important for calculating risk sets)
    # Ties in event_times are typically handled by specific methods (breslow, efron),
    # this is a simplified version often used (equivalent to Breslow for unique times).
    desc_indices = torch.argsort(event_times, descending=True)
    risk_scores_sorted = risk_scores[desc_indices]
    event_times_sorted = event_times[desc_indices] # Not directly used in loss but good to have
    event_observed_sorted = event_observed[desc_indices]
    event_observed_bool_sorted = event_observed_bool[desc_indices]


    # Calculate log of sum of exp(risk_scores) for risk sets
    # logsumexp is more numerically stable than log(sum(exp()))
    # For each event time i, the risk set R_i includes all individuals j where T_j >= T_i.
    # exp_risk_scores = torch.exp(risk_scores_sorted)
    # log_risk_set_sum = torch.log(torch.cumsum(exp_risk_scores, dim=0)) # This is log(sum_{j in R_i} exp(risk_j))

    # A more stable way for the denominator term:
    # For each event i, calculate log(sum_{j >= i} exp(risk_scores_sorted_j))
    # This can be done by reversing, cumsum, reversing, then log. Or use logcumsumexp.
    # Let's use a direct logsumexp for each risk set (can be slow for large batches if done per-event naively)
    # Or, the standard way:
    hazard_ratios = torch.exp(risk_scores_sorted)
    log_hazard_sum_terms = torch.log(torch.cumsum(hazard_ratios, dim=0) + 1e-9) # Denominators for each observation if it were an event

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


def concordance_index(event_times, predicted_scores, event_observed):
    """
    Calculates the concordance index (C-index).
    Higher predicted_scores should correlate with earlier event_times.
    Args:
        event_times (np.ndarray): Time to event or censoring.
        predicted_scores (np.ndarray): Predicted risk scores. Higher score = higher risk.
        event_observed (np.ndarray): Event indicator (1 if event, 0 if censored). Int type.
    Returns:
        float: Concordance index.
    """
    # Ensure inputs are numpy arrays
    event_times = np.asarray(event_times)
    predicted_scores = np.asarray(predicted_scores)
    event_observed = np.asarray(event_observed).astype(int)

    # Remove NaNs if any - crucial for correct calculation
    nan_mask = np.isnan(event_times) | np.isnan(predicted_scores) | np.isnan(event_observed)
    if np.any(nan_mask):
        # logger.debug(f"NaNs found in C-index input. Removing {np.sum(nan_mask)} entries.")
        event_times = event_times[~nan_mask]
        predicted_scores = predicted_scores[~nan_mask]
        event_observed = event_observed[~nan_mask]

    if len(event_times) < 2: return 0.5 # Not enough data for comparison

    concordant_pairs = 0
    discordant_pairs = 0
    num_comparable_pairs = 0 # Using a broader definition of comparable for clarity

    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            # Check for permissible pairs:
            # 1. Both events observed: pair is permissible.
            # 2. One event observed, one censored:
            #    - If event_i=1, event_j=0: permissible if time_i < time_j (i fails before j is censored)
            #    - If event_i=0, event_j=1: permissible if time_j < time_i (j fails before i is censored)
            # 3. Both censored: not traditionally permissible for risk score comparison.

            # Case 1: Both events occurred (event_observed[i]=1 and event_observed[j]=1)
            if event_observed[i] == 1 and event_observed[j] == 1:
                num_comparable_pairs +=1
                if event_times[i] < event_times[j]: # i failed before j
                    if predicted_scores[i] > predicted_scores[j]: concordant_pairs += 1
                    elif predicted_scores[i] < predicted_scores[j]: discordant_pairs += 1
                    else: concordant_pairs += 0.5 # Tie in scores
                elif event_times[j] < event_times[i]: # j failed before i
                    if predicted_scores[j] > predicted_scores[i]: concordant_pairs += 1
                    elif predicted_scores[j] < predicted_scores[i]: discordant_pairs += 1
                    else: concordant_pairs += 0.5 # Tie in scores
                # If event_times[i] == event_times[j] and both observed, it's a tie in time.
                # Some definitions exclude these or handle score ties.
                # Here, if times are equal, scores should ideally be equal for concordance.
                # If scores differ, it's ambiguous. For simplicity with tied times and events:
                # if predicted_scores[i] == predicted_scores[j]: concordant_pairs += 0.5 (counted as half)
                # else: # scores differ, could be discordant or neutral depending on definition.
                # This simplified version does not add to discordant for this specific sub-case.

            # Case 2: Patient i had an event, patient j was censored (event_observed[i]=1, event_observed[j]=0)
            elif event_observed[i] == 1 and event_observed[j] == 0:
                if event_times[i] < event_times[j]: # i failed before j was censored
                    num_comparable_pairs +=1
                    if predicted_scores[i] > predicted_scores[j]: concordant_pairs += 1
                    elif predicted_scores[i] < predicted_scores[j]: discordant_pairs += 1
                    else: concordant_pairs += 0.5
                # If event_times[i] == event_times[j] (i failed at same time j was censored)
                # this is also comparable.
                elif event_times[i] == event_times[j]:
                    num_comparable_pairs +=1
                    if predicted_scores[i] > predicted_scores[j]: concordant_pairs += 1
                    elif predicted_scores[i] < predicted_scores[j]: discordant_pairs += 1 # j should have lower risk
                    else: concordant_pairs += 0.5


            # Case 3: Patient j had an event, patient i was censored (event_observed[j]=1, event_observed[i]=0)
            elif event_observed[j] == 1 and event_observed[i] == 0:
                if event_times[j] < event_times[i]: # j failed before i was censored
                    num_comparable_pairs +=1
                    if predicted_scores[j] > predicted_scores[i]: concordant_pairs += 1
                    elif predicted_scores[j] < predicted_scores[i]: discordant_pairs += 1
                    else: concordant_pairs += 0.5
                elif event_times[j] == event_times[i]:
                    num_comparable_pairs +=1
                    if predicted_scores[j] > predicted_scores[i]: concordant_pairs += 1
                    elif predicted_scores[j] < predicted_scores[i]: discordant_pairs += 1
                    else: concordant_pairs += 0.5

    if num_comparable_pairs == 0:
        return 0.5 # Or np.nan, depends on how you want to handle no comparable pairs
    return concordant_pairs / num_comparable_pairs


def compute_metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    try:
        cm = confusion_matrix(labels, preds)
        if cm.size == 1: # Only one class predicted and present
             if labels[0] == 0 : tn, fp, fn, tp = len(labels), 0,0,0
             else: tn, fp, fn, tp = 0,0,0,len(labels)
        else:
             tn, fp, fn, tp = cm.ravel()

    except ValueError:
        # Fallback if ravel fails (e.g. only one class in labels or preds)
        tp = np.sum((labels == 1) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        # logger.warning("Confusion matrix could not be raveled normally. Using manual calculation.")

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
        # logger.warning("Empty labels/probs for bootstrapping.")
        all_keys = ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]
        return {k: (np.nan, np.nan, np.nan) for k in all_keys}

    all_keys = ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]
    metric_samples = {k: [] for k in all_keys}

    for _ in range(n_boot): # tqdm might be too verbose here, can disable
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_labels = labels[indices]
        sample_probs = probs[indices]

        if len(sample_labels) == 0 or len(np.unique(sample_labels)) < 1 : # Should not happen with replace=True if n_samples > 0
             for k in all_keys: metric_samples[k].append(np.nan)
             continue
        if len(np.unique(sample_labels)) < 2 and (("auroc" in all_keys) or ("auprc" in all_keys)): # AUROC/AUPRC need >1 class
            # For other metrics, they might still be valid (e.g. accuracy if all are one class and predicted as such)
            # Compute what we can, others will be NaN from compute_metrics_at_threshold or here
            temp_result = compute_metrics_at_threshold(sample_labels, sample_probs, threshold)
            for k_single in all_keys:
                 metric_samples[k_single].append(temp_result.get(k_single, np.nan))
            continue


        sample_result = compute_metrics_at_threshold(sample_labels, sample_probs, threshold)
        for k_all in all_keys:
            metric_samples[k_all].append(sample_result[k_all])

    ci_results = {}
    for k_ci in all_keys:
        arr = np.array(metric_samples[k_ci])
        arr_clean = arr[~np.isnan(arr)] # Remove NaNs for percentile calculation
        if len(arr_clean) < 2: # Not enough data points for CI
            lower, upper, meanv = np.nan, np.nan, np.nan
            if len(arr_clean) == 1: meanv = arr_clean[0]
        else:
            lower = np.percentile(arr_clean, 2.5)
            upper = np.percentile(arr_clean, 97.5)
            meanv = np.mean(arr_clean)
        ci_results[k_ci] = (meanv, lower, upper)
    return ci_results


def time_to_event_preprocessing(meta_df_input, log_transform_tte=False):
    meta = meta_df_input.copy()
    meta["EventDate"] = pd.to_datetime(meta["EventDate"])
    meta = meta.sort_values(by=["PatientID", "EventDate"]).reset_index(drop=True)

    meta["time_to_first_progression"] = np.nan # Time from current visit to patient's first progression
    meta["event_for_cox"] = 0 # 1 if this visit IS the first progression, 0 otherwise (censored or pre-event)

    for pid, group in tqdm(meta.groupby("PatientID"), desc="Preprocessing TTE for Cox", leave=False, disable=True):
        # 'label' here is the original CKD stage-based label (1 if stage >=4)
        first_prog_rows = group[group["label"] == 1]
        if not first_prog_rows.empty:
            first_prog_date = first_prog_rows["EventDate"].min()
            # Get the actual index in the original `meta` DataFrame for the first progression event of this patient
            first_prog_meta_idx = first_prog_rows["EventDate"].idxmin()

            for current_visit_idx in group.index: # Iterate using original index from `meta`
                current_date = meta.loc[current_visit_idx, "EventDate"]
                if current_date <= first_prog_date:
                    delta_days = (first_prog_date - current_date).days
                    meta.loc[current_visit_idx, "time_to_first_progression"] = float(delta_days)
                    if current_visit_idx == first_prog_meta_idx: # This is the exact first progression visit
                        meta.loc[current_visit_idx, "event_for_cox"] = 1
                    else: # Visits before the first progression event
                        meta.loc[current_visit_idx, "event_for_cox"] = 0 # Event hasn't happened yet
                else: # Visits after the first progression event
                    meta.loc[current_visit_idx, "time_to_first_progression"] = np.nan # Or 0, or -1. Task dependent.
                    meta.loc[current_visit_idx, "event_for_cox"] = 0 # Event already passed
        else:
            # No progression event (label=1) for this patient in the dataset
            # All visits are censored relative to the "first progression" event type
            # Time to event is time to last follow-up for this patient from each visit
            if not group.empty:
                last_observed_date_for_patient = group["EventDate"].max()
                for current_visit_idx in group.index:
                    current_date = meta.loc[current_visit_idx, "EventDate"]
                    delta_to_last_obs = (last_observed_date_for_patient - current_date).days
                    meta.loc[current_visit_idx, "time_to_first_progression"] = float(delta_to_last_obs)
                    meta.loc[current_visit_idx, "event_for_cox"] = 0 # Censored

    if log_transform_tte:
        # Add 1 before log to handle 0-day TTE and ensure positive values
        meta["time_to_first_progression"] = np.log(meta["time_to_first_progression"] + 1.0)
        # Handle -inf from log(0) if any NaNs resulted in log(0+1) then NaN again, or if TTE was negative.
        meta.loc[np.isneginf(meta["time_to_first_progression"]), "time_to_first_progression"] = np.nan
        logger.info("Applied log transformation to 'time_to_first_progression'.")
    return meta


def build_sequences_1year_future_label(metadata_df, window_size, prediction_horizon_days, for_multitask=False):
    data_records = []
    required_cols = ["PatientID", "EventDate", "embedding", "label"]
    if for_multitask:
        required_cols.extend(["time_to_first_progression", "event_for_cox"])
    if not all(col in metadata_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in metadata_df.columns]
        raise ValueError(f"Metadata missing required columns for sequence building: {missing}. Have: {metadata_df.columns}")

    metadata_df["EventDate"] = pd.to_datetime(metadata_df["EventDate"]) # Ensure datetime
    prediction_timedelta = timedelta(days=prediction_horizon_days)

    for pid, group in tqdm(metadata_df.groupby("PatientID"), desc=f"Building {prediction_horizon_days}-day future label sequences", leave=False, disable=True):
        group = group.sort_values(by="EventDate").reset_index(drop=True) # Use reset_index for clean 0-based indexing within group
        
        embeddings_list = list(group["embedding"])
        original_labels_list = list(group["label"]) # Original CKD stage-derived label
        event_dates_list = list(group["EventDate"])
        
        if for_multitask:
            tte_cox_list = list(group["time_to_first_progression"])
            event_cox_list = list(group["event_for_cox"])

        for i in range(len(group)): # Current visit index within the patient's group
            # Context: `window_size` visits up to and including the current visit `i`
            start_idx_context = max(0, i - window_size + 1)
            context_embeddings = embeddings_list[start_idx_context : i + 1]

            if not context_embeddings: # Should not happen if i is valid and window_size >=1
                continue

            current_visit_date = event_dates_list[i]
            horizon_end_date = current_visit_date + prediction_timedelta
            
            # Determine the label for the 1-year future prediction
            future_event_occurs = 0
            # Look for an original 'label' == 1 in subsequent visits within the horizon
            for j in range(i + 1, len(group)): # Look at visits *after* current
                future_visit_date = event_dates_list[j]
                if future_visit_date <= horizon_end_date: # If future visit is within horizon
                    if original_labels_list[j] == 1: # And it's a progression event
                        future_event_occurs = 1
                        break # Found an event, no need to look further for this horizon
                else: # Future visit is beyond the horizon
                    break 
            
            # local_idx is the index of the current visit 'i' within the patient's sequence of visits
            # This helps in aligning predictions back to specific visits if needed later.
            if for_multitask:
                data_records.append((
                    context_embeddings,
                    future_event_occurs, # Label for 1-year classification task
                    pid,
                    i, # Local index within patient's sorted visits
                    tte_cox_list[i],    # TTE for Cox loss (time to *first* overall progression)
                    event_cox_list[i]   # Event indicator for Cox loss
                ))
            else:
                data_records.append((
                    context_embeddings,
                    future_event_occurs, # Label for 1-year classification task
                    pid,
                    i # Local index
                ))
    return data_records


def train_and_evaluate_deepsurv(model, device, train_loader, val_loader, test_loader, args, model_name):
    logger.info(f"Starting {model_name} training (DeepSurv: {args.prediction_horizon_days}-day future classification + TTE to first progression).")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, verbose=False)
    classification_criterion = nn.CrossEntropyLoss()
    cox_loss_w = args.cox_loss_weight

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_dir = os.path.dirname(f"{args.output_model_prefix}_{model_name}.pt")
    if model_dir == "": model_dir = "." # Handle cases where prefix has no path
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"


    for epoch in range(args.epochs):
        model.train()
        train_total_losses, train_class_losses, train_surv_losses = [], [], []
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            x_batch, classification_target, _, _, tte_cox, event_cox = batch_data
            
            x_batch = x_batch.to(device)
            classification_target = classification_target.to(device) # N-day future label
            tte_cox = tte_cox.to(device)         # Time to first progression
            event_cox = event_cox.to(device)       # Event indicator for first progression (0 or 1)

            cl_logits, risk_scores = model(x_batch)

            class_loss = classification_criterion(cl_logits, classification_target.long())
            
            # Cox loss for time to *first* progression
            # Mask for valid (non-NaN) TTE data for Cox loss and non-NaN risk scores
            valid_cox_mask = ~torch.isnan(tte_cox) & ~torch.isnan(risk_scores) & ~torch.isnan(event_cox)
            
            surv_loss = torch.tensor(0.0, device=device, requires_grad=True) # Default if no valid data/events
            if valid_cox_mask.sum() > 0:
                masked_risk = risk_scores[valid_cox_mask]
                masked_tte = tte_cox[valid_cox_mask]
                masked_event = event_cox[valid_cox_mask] # Should be float (0.0 or 1.0)
                if masked_event.sum() > 0: # Ensure there's at least one event in the masked batch
                    surv_loss = cox_ph_loss(masked_risk, masked_tte, masked_event)
            
            total_loss = class_loss + cox_loss_w * surv_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN/Inf loss in {model_name} train epoch {epoch+1}, batch {batch_idx}. CL:{class_loss.item():.4f} SL:{surv_loss.item():.4f}. Skipping update.")
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
            for batch_data_val in val_loader:
                x_val, class_target_val, _, _, tte_cox_val, event_cox_val = batch_data_val
                x_val, class_target_val = x_val.to(device), class_target_val.to(device)
                tte_cox_val, event_cox_val = tte_cox_val.to(device), event_cox_val.to(device)

                cl_logits_val, risk_val = model(x_val)
                class_l_val = classification_criterion(cl_logits_val, class_target_val.long())
                
                valid_cox_mask_val = ~torch.isnan(tte_cox_val) & ~torch.isnan(risk_val) & ~torch.isnan(event_cox_val)
                surv_l_val = torch.tensor(0.0, device=device)
                if valid_cox_mask_val.sum() > 0:
                    m_risk_v, m_tte_v, m_event_v = risk_val[valid_cox_mask_val], tte_cox_val[valid_cox_mask_val], event_cox_val[valid_cox_mask_val]
                    if m_event_v.sum() > 0: surv_l_val = cox_ph_loss(m_risk_v, m_tte_v, m_event_v)
                
                total_l_val = class_l_val + cox_loss_w * surv_l_val
                if not (torch.isnan(total_l_val) or torch.isinf(total_l_val)):
                    val_total_losses.append(total_l_val.item())
                    val_class_losses.append(class_l_val.item())
                    val_surv_losses.append(surv_l_val.item())

        avg_train_loss = np.nanmean(train_total_losses) if train_total_losses else float('inf')
        avg_train_cl = np.nanmean(train_class_losses) if train_class_losses else float('inf')
        avg_train_sl = np.nanmean(train_surv_losses) if train_surv_losses else float('inf')
        avg_val_loss = np.nanmean(val_total_losses) if val_total_losses else float('inf')
        avg_val_cl = np.nanmean(val_class_losses) if val_class_losses else float('inf')
        avg_val_sl = np.nanmean(val_surv_losses) if val_surv_losses else float('inf')

        logger.info(f"{model_name} Ep {epoch+1}/{args.epochs} -> Tr L: {avg_train_loss:.4f}(C:{avg_train_cl:.4f} S:{avg_train_sl:.4f}), "
                    f"Val L: {avg_val_loss:.4f}(C:{avg_val_cl:.4f} S:{avg_val_sl:.4f})")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"{model_name}: Val loss improved to {best_val_loss:.4f}. Model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"{model_name}: Early stopping. No improvement in val loss for {args.patience} epochs.")
                break
            if np.isnan(avg_val_loss) and epoch > args.scheduler_patience : # Stop if val loss is persistently NaN after a few epochs
                logger.warning(f"{model_name}: Val loss is NaN. Early stopping.")
                break
    
    # Load best model for evaluation
    if os.path.exists(best_model_path):
        logger.info(f"{model_name}: Loading best model from {best_model_path} for final test evaluation.")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning(f"{model_name}: Best model not found at {best_model_path}. Using current model state (last epoch).")

    # Final Test Evaluation
    model.eval()
    all_cl_targets_test, all_cl_probs_test, all_cl_logits_test = [], [], []
    all_risk_scores_test, all_tte_cox_test, all_event_cox_test = [], [], []

    with torch.no_grad():
        for batch_test in test_loader:
            x_test, cl_target_t, _, _, tte_cox_t, event_cox_t = batch_test
            x_test = x_test.to(device)
            
            cl_logits_t, risk_t = model(x_test)
            cl_probs_t = nn.Softmax(dim=1)(cl_logits_t)[:, 1] # Prob of class 1 for N-day future

            all_cl_targets_test.extend(cl_target_t.cpu().numpy())
            all_cl_probs_test.extend(cl_probs_t.cpu().numpy())
            all_cl_logits_test.extend(cl_logits_t.cpu().numpy())
            all_risk_scores_test.extend(risk_t.cpu().numpy())
            all_tte_cox_test.extend(tte_cox_t.cpu().numpy())
            all_event_cox_test.extend(event_cox_t.cpu().numpy())

    all_cl_targets_test = np.array(all_cl_targets_test)
    all_cl_probs_test = np.array(all_cl_probs_test)
    all_cl_logits_test = np.array(all_cl_logits_test)
    all_risk_scores_test = np.array(all_risk_scores_test)
    all_tte_cox_test = np.array(all_tte_cox_test)
    all_event_cox_test = np.array(all_event_cox_test).astype(int)

    # --- Classification Metrics (N-day future prediction) ---
    logger.info(f"--- {model_name}: Classification Metrics ({args.prediction_horizon_days}-day future) ---")
    final_results_dict = {"model_name": model_name}
    if len(all_cl_targets_test) > 0:
        prevalence_cl = np.mean(all_cl_targets_test) if len(all_cl_targets_test) > 0 else 0.0
        logger.info(f"{model_name} Test Prevalence ({args.prediction_horizon_days}-day future label): {prevalence_cl:.4f}")
        threshold_cl = prevalence_cl if 0 < prevalence_cl < 1 else 0.5 # Use prevalence or 0.5 if extreme

        cl_metrics_raw = compute_metrics_at_threshold(all_cl_targets_test, all_cl_probs_test, threshold_cl)
        cl_metrics_ci = bootstrap_metrics(all_cl_targets_test, all_cl_probs_test, threshold_cl, random_state=args.random_seed)
        
        logger.info(f"{model_name} Classification Threshold set to {threshold_cl:.4f}.")
        for k_metric in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]:
            raw_val = cl_metrics_raw.get(k_metric, np.nan)
            ci_mean, ci_low, ci_high = cl_metrics_ci.get(k_metric, (np.nan, np.nan, np.nan))
            logger.info(f"{model_name} Class-{k_metric.upper()}: {raw_val:.4f} (CI Mean: {ci_mean:.4f} [{ci_low:.4f}-{ci_high:.4f}])")
            final_results_dict[f"class_{k_metric}"] = raw_val # Store raw point estimate
    else: logger.warning(f"{model_name}: No classification targets in test set.")

    # --- Survival Metric (C-index for time-to-first-progression) ---
    logger.info(f"--- {model_name}: Survival Metric (Time-to-First-Progression) ---")
    valid_cox_test_mask = ~np.isnan(all_tte_cox_test) & ~np.isnan(all_risk_scores_test) & ~np.isnan(all_event_cox_test)
    if np.sum(valid_cox_test_mask) > 1 and np.sum(all_event_cox_test[valid_cox_test_mask]) > 0:
        c_idx_tte = concordance_index(
            all_tte_cox_test[valid_cox_test_mask],
            all_risk_scores_test[valid_cox_test_mask], # Higher score = higher risk
            all_event_cox_test[valid_cox_test_mask]
        )
        logger.info(f"{model_name} Concordance Index (TTE to first prog): {c_idx_tte:.4f}")
        final_results_dict["concordance_index_tte"] = c_idx_tte
    else:
        logger.warning(f"{model_name}: Not enough valid data or no events to calculate C-index for TTE.")
        final_results_dict["concordance_index_tte"] = np.nan

    # Save detailed outputs
    output_dir_details = f"./{args.prediction_horizon_days}day_future_prediction_outputs"
    os.makedirs(output_dir_details, exist_ok=True)
    df_details = pd.DataFrame({
        "cl_logit_0": all_cl_logits_test[:, 0] if len(all_cl_logits_test.shape) == 2 else ([np.nan]*len(all_cl_targets_test) if len(all_cl_logits_test)>0 else []),
        "cl_logit_1": all_cl_logits_test[:, 1] if len(all_cl_logits_test.shape) == 2 else ([np.nan]*len(all_cl_targets_test) if len(all_cl_logits_test)>0 else []),
        "cl_prob_1": all_cl_probs_test,
        "cl_true_label": all_cl_targets_test,
        "tte_cox_risk_score": all_risk_scores_test,
        "tte_cox_true_time": all_tte_cox_test,
        "tte_cox_true_event": all_event_cox_test
    })
    out_csv_path = os.path.join(output_dir_details, f"{model_name}_detailed_outputs.csv")
    df_details.to_csv(out_csv_path, index=False)
    logger.info(f"{model_name}: Detailed test outputs saved to {out_csv_path}")

    return final_results_dict


def train_and_evaluate(model, device, train_loader, val_loader, test_loader, args, model_name):
    logger.info(f"Starting {model_name} training (Classification: {args.prediction_horizon_days}-day future label).")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, verbose=False)
    classification_criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    model_dir = os.path.dirname(f"{args.output_model_prefix}_{model_name}.pt")
    if model_dir == "": model_dir = "."
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = f"{args.output_model_prefix}_{model_name}.pt"

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            x_batch, y_batch, _, _ = batch_data # y_batch is N-day future label
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            logits = model(x_batch)
            loss = classification_criterion(logits, y_batch.long())

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss in {model_name} train epoch {epoch+1}, batch {batch_idx}. Loss:{loss.item():.4f}. Skipping update.")
                continue
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_val in val_loader:
                x_val, y_val, _, _ = batch_val
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits_val = model(x_val)
                loss_val = classification_criterion(logits_val, y_val.long())
                if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                    val_losses.append(loss_val.item())
        
        avg_train_loss = np.nanmean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.nanmean(val_losses) if val_losses else float('inf')
        logger.info(f"{model_name} Ep {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"{model_name}: Val loss improved to {best_val_loss:.4f}. Model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"{model_name}: Early stopping triggered.")
                break
            if np.isnan(avg_val_loss) and epoch > args.scheduler_patience:
                logger.warning(f"{model_name}: Val loss is NaN. Early stopping.")
                break

    if os.path.exists(best_model_path):
        logger.info(f"{model_name}: Loading best model from {best_model_path} for test evaluation.")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning(f"{model_name}: Best model not found. Using current model state (last epoch).")

    model.eval()
    all_targets_test, all_probs_test, all_logits_test = [], [], []
    with torch.no_grad():
        for batch_t in test_loader:
            x_t, y_t, _, _ = batch_t
            x_t = x_t.to(device)
            logits_t = model(x_t)
            probs_t = nn.Softmax(dim=1)(logits_t)[:, 1]
            all_targets_test.extend(y_t.cpu().numpy())
            all_probs_test.extend(probs_t.cpu().numpy())
            all_logits_test.extend(logits_t.cpu().numpy())

    all_targets_test = np.array(all_targets_test)
    all_probs_test = np.array(all_probs_test)
    all_logits_test = np.array(all_logits_test)
    
    final_results_dict = {"model_name": model_name}
    if len(all_targets_test) > 0 :
        prevalence = np.mean(all_targets_test) if len(all_targets_test) > 0 else 0.0
        logger.info(f"{model_name} Test Prevalence ({args.prediction_horizon_days}-day future label): {prevalence:.4f}")
        threshold = prevalence if 0 < prevalence < 1 else 0.5

        metrics_raw = compute_metrics_at_threshold(all_targets_test, all_probs_test, threshold)
        metrics_ci = bootstrap_metrics(all_targets_test, all_probs_test, threshold, random_state=args.random_seed)

        logger.info(f"{model_name} Classification Threshold set to {threshold:.4f}.")
        for k_m in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]:
            raw_v = metrics_raw.get(k_m, np.nan)
            ci_m, ci_l, ci_h = metrics_ci.get(k_m, (np.nan,np.nan,np.nan))
            logger.info(f"{model_name} {k_m.upper()}: {raw_v:.4f} (CI Mean: {ci_m:.4f} [{ci_l:.4f}-{ci_h:.4f}])")
            final_results_dict[k_m] = raw_v # Store raw point estimate
    else: logger.warning(f"{model_name}: No targets in test set for evaluation.")


    output_dir_dets = f"./{args.prediction_horizon_days}day_future_prediction_outputs"
    os.makedirs(output_dir_dets, exist_ok=True)
    df_dets = pd.DataFrame({
        "logit_0": all_logits_test[:, 0] if len(all_logits_test.shape) == 2 else ([np.nan]*len(all_targets_test) if len(all_logits_test)>0 else []),
        "logit_1": all_logits_test[:, 1] if len(all_logits_test.shape) == 2 else ([np.nan]*len(all_targets_test) if len(all_logits_test)>0 else []),
        "prob_positive": all_probs_test,
        "true_label": all_targets_test
    })
    out_csv_p = os.path.join(output_dir_dets, f"{model_name}_detailed_outputs.csv")
    df_dets.to_csv(out_csv_p, index=False)
    logger.info(f"{model_name}: Detailed test outputs saved to {out_csv_p}")
    
    return final_results_dict


def predict_label_switches(model, loader, device, is_deepsurv_model=False):
    model.eval()
    records = []
    with torch.no_grad():
        for batch_data in loader:
            if is_deepsurv_model:
                x_batch, y_batch, pid_batch, idx_batch, _, _ = batch_data # y_batch is N-day future label
            else:
                x_batch, y_batch, pid_batch, idx_batch = batch_data # y_batch is N-day future label
            
            x_batch = x_batch.to(device)
            if is_deepsurv_model:
                classification_logits, _ = model(x_batch)
            else:
                classification_logits = model(x_batch)
            
            probs = nn.Softmax(dim=1)(classification_logits)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            # pid_batch and idx_batch are lists/tuples of tensors if not collated carefully, or just lists of numbers
            # Assuming they are already simple lists/arrays of PIDs and indices
            for pid_val, true_lbl, pred_lbl, local_idx_val in zip(pid_batch, y_batch.cpu().numpy(), preds, idx_batch):
                records.append((
                    pid_val.item() if isinstance(pid_val, torch.Tensor) else pid_val, # Ensure PID is a simple value
                    local_idx_val.item() if isinstance(local_idx_val, torch.Tensor) else local_idx_val, # Ensure local_idx is simple
                    true_lbl, 
                    pred_lbl
                ))
    df = pd.DataFrame(records, columns=["PatientID", "LocalIndex", f"TrueLabel_{args.prediction_horizon_days}DayFuture", f"PredLabel_{args.prediction_horizon_days}DayFuture"])
    return df

def analyze_switches_Nday_future(df_preds_Nday, N_days_horizon):
    analysis_records = []
    # Column names depend on N_days_horizon used in predict_label_switches
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
    global args # Make args global so predict_label_switches can access prediction_horizon_days for col names
    args = parse_args()
    logger.info(f"Running with configuration for {args.prediction_horizon_days}-DAY FUTURE PREDICTION:")
    for key, val in vars(args).items(): logger.info(f"{key}: {val}")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        # For full reproducibility, but may impact performance slightly
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    logger.info("Loading metadata...")
    metadata_path = os.path.join(args.embedding_root, args.metadata_file)
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}. Exiting."); return
    metadata = pd.read_csv(metadata_path)
    logger.info(f"Initial metadata rows: {len(metadata)}")

    metadata['CKD_stage_clean'] = metadata['CKD_stage'].apply(clean_ckd_stage)
    metadata = metadata.sort_values(by=['PatientID', 'EventDate'])
    metadata['CKD_stage_clean'] = metadata.groupby('PatientID')['CKD_stage_clean'].bfill().ffill()
    metadata = metadata.dropna(subset=['CKD_stage_clean'])
    metadata['CKD_stage_clean'] = metadata['CKD_stage_clean'].astype(int)
    metadata['label'] = metadata['CKD_stage_clean'].apply(lambda x: 1 if x >= 4 else 0) # Original progression label
    logger.info(f"Rows after CKD stage cleaning & 'label' creation: {len(metadata)}")

    logger.info("Filtering for existing embedding files...")
    metadata = metadata[metadata.apply(lambda row: embedding_exists(row, args.embedding_root), axis=1)]
    logger.info(f"Rows after checking embedding existence: {len(metadata)}")
    if metadata.empty: logger.error("No valid data after filtering for embeddings. Exiting."); return

    unique_pids_initial = sorted(metadata['PatientID'].unique())
    if args.max_patients is not None and args.max_patients < len(unique_pids_initial):
        subset_pids = unique_pids_initial[:args.max_patients]
        metadata = metadata[metadata['PatientID'].isin(subset_pids)]
        logger.info(f"Using only {args.max_patients} patients. Rows: {len(metadata)}")

    embedding_cache_dict = {}
    logger.info("Loading embeddings (this may take a while)...")
    metadata['embedding'] = metadata.apply(lambda r: load_embedding(os.path.join(args.embedding_root, r['embedding_file']), embedding_cache_dict), axis=1)
    logger.info("Embeddings loaded.")

    logger.info("Preprocessing TTE data for Cox loss component (time to first overall progression).")
    metadata = time_to_event_preprocessing(metadata, log_transform_tte=args.log_tte)

    unique_pids_processed = sorted(metadata['PatientID'].unique())
    if not unique_pids_processed: logger.error("No patients left after TTE preprocessing. Exiting."); return
    
    train_pids, temp_pids = train_test_split(unique_pids_processed, test_size=0.3, random_state=args.random_seed) # 70% train
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=args.random_seed) # 15% val, 15% test

    train_meta = metadata[metadata['PatientID'].isin(train_pids)].copy()
    val_meta = metadata[metadata['PatientID'].isin(val_pids)].copy()
    test_meta = metadata[metadata['PatientID'].isin(test_pids)].copy()
    logger.info(f"Data split: Train PIDs={len(train_pids)} ({len(train_meta)} recs), Val PIDs={len(val_pids)} ({len(val_meta)} recs), Test PIDs={len(test_pids)} ({len(test_meta)} recs)")

    # Build sequences for standard classification models (N-day future label)
    logger.info(f"Building sequences for {args.prediction_horizon_days}-day future classification task...")
    train_seq_cl = build_sequences_1year_future_label(train_meta, args.window_size, args.prediction_horizon_days, for_multitask=False)
    val_seq_cl = build_sequences_1year_future_label(val_meta, args.window_size, args.prediction_horizon_days, for_multitask=False)
    test_seq_cl = build_sequences_1year_future_label(test_meta, args.window_size, args.prediction_horizon_days, for_multitask=False)

    if not all([train_seq_cl, val_seq_cl, test_seq_cl]): 
        logger.error("One or more classification sequence sets are empty. Check data/logic. Exiting."); return
    
    train_ds_cl = CKDSequenceDataset(train_seq_cl, args.window_size, args.embed_dim, is_multitask_data=False)
    val_ds_cl = CKDSequenceDataset(val_seq_cl, args.window_size, args.embed_dim, is_multitask_data=False)
    test_ds_cl = CKDSequenceDataset(test_seq_cl, args.window_size, args.embed_dim, is_multitask_data=False)
    train_loader_cl = DataLoader(train_ds_cl, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader_cl = DataLoader(val_ds_cl, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    test_loader_cl = DataLoader(test_ds_cl, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    logger.info(f"Classification datasets: Train={len(train_ds_cl)}, Val={len(val_ds_cl)}, Test={len(test_ds_cl)} sequences.")

    # Build sequences for DeepSurv multitask models
    logger.info(f"Building sequences for DeepSurv multitask ({args.prediction_horizon_days}-day classif. + TTE Cox)...")
    train_seq_ds = build_sequences_1year_future_label(train_meta, args.window_size, args.prediction_horizon_days, for_multitask=True)
    val_seq_ds = build_sequences_1year_future_label(val_meta, args.window_size, args.prediction_horizon_days, for_multitask=True)
    test_seq_ds = build_sequences_1year_future_label(test_meta, args.window_size, args.prediction_horizon_days, for_multitask=True)

    if not all([train_seq_ds, val_seq_ds, test_seq_ds]):
        logger.error("One or more DeepSurv sequence sets are empty. Check data/logic. Exiting."); return

    train_ds_ds = CKDSequenceDataset(train_seq_ds, args.window_size, args.embed_dim, is_multitask_data=True)
    val_ds_ds = CKDSequenceDataset(val_seq_ds, args.window_size, args.embed_dim, is_multitask_data=True)
    test_ds_ds = CKDSequenceDataset(test_seq_ds, args.window_size, args.embed_dim, is_multitask_data=True)
    train_loader_ds = DataLoader(train_ds_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader_ds = DataLoader(val_ds_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    test_loader_ds = DataLoader(test_ds_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    logger.info(f"DeepSurv multitask datasets: Train={len(train_ds_ds)}, Val={len(val_ds_ds)}, Test={len(test_ds_ds)} sequences.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Define Models ---
    cl_models_defs = {
        "RNN": lambda: LongitudinalRNN(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "LSTM": lambda: LongitudinalLSTM(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "Transformer": lambda: LongitudinalTransformer(args.embed_dim, args.num_layers, args.transformer_nhead, args.transformer_dim_feedforward, args.transformer_dropout),
        "MLP": lambda: MLPSimple(args.embed_dim, args.window_size, args.hidden_dim, args.num_layers, args.rnn_dropout), # Assuming rnn_dropout for MLP dropout too
        "TCN": lambda: TCN(args.embed_dim, args.hidden_dim, args.num_layers, kernel_size=3, dropout=args.rnn_dropout) # Assuming rnn_dropout for TCN
    }
    ds_models_defs = {
        "DeepSurv_RNN": lambda: DeepSurvRNN(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "DeepSurv_LSTM": lambda: DeepSurvLSTM(args.embed_dim, args.hidden_dim, args.num_layers, args.rnn_dropout, args.rnn_bidir),
        "DeepSurv_Transformer": lambda: DeepSurvTransformer(args.embed_dim, args.num_layers, args.transformer_nhead, args.transformer_dim_feedforward, args.transformer_dropout),
        "DeepSurv_MLP": lambda: DeepSurvMLP(args.embed_dim, args.window_size, args.hidden_dim, args.num_layers, args.rnn_dropout),
        "DeepSurv_TCN": lambda: DeepSurvTCN(args.embed_dim, args.hidden_dim, args.num_layers, kernel_size=3, dropout=args.rnn_dropout)
    }
    
    all_cl_results, all_ds_results = [], []
    trained_cl_models, trained_ds_models = {}, {}


    if args.epochs > 0:
        logger.info(f"--- Training Classification-Only Models ({args.prediction_horizon_days}-Day Future Prediction) ---")
        if len(train_loader_cl) > 0 and len(val_loader_cl) > 0 and len(test_loader_cl) > 0:
            for name, model_fn in cl_models_defs.items():
                model_instance = model_fn()
                results = train_and_evaluate(model_instance, device, train_loader_cl, val_loader_cl, test_loader_cl, args, f"{name}_{args.prediction_horizon_days}DayFuture")
                all_cl_results.append(results)
                trained_cl_models[name] = model_instance # Store trained model
        else: logger.warning("Skipping classification model training due to empty/insufficient data loaders.")

        logger.info(f"--- Training DeepSurv Multi-Task Models ({args.prediction_horizon_days}-Day Classif. + TTE Cox) ---")
        if len(train_loader_ds) > 0 and len(val_loader_ds) > 0 and len(test_loader_ds) > 0:
            for name, model_fn in ds_models_defs.items():
                model_instance = model_fn()
                results = train_and_evaluate_deepsurv(model_instance, device, train_loader_ds, val_loader_ds, test_loader_ds, args, f"{name}_{args.prediction_horizon_days}DayFuture")
                all_ds_results.append(results)
                trained_ds_models[name] = model_instance # Store trained model
        else: logger.warning("Skipping DeepSurv model training due to empty/insufficient data loaders.")
    else: logger.info("Skipping all model training as epochs is 0.")


    # --- Summarize Results ---
    logger.info(f"\n--- Summary: Final Test Metrics (Classification Models - {args.prediction_horizon_days}-Day Future) ---")
    for res in all_cl_results:
        if res and isinstance(res, dict):
            log_line = f"Model={res.get('model_name', 'N/A')} "
            for met in ["auroc", "auprc", "f1", "accuracy", "precision", "recall", "ppv", "npv"]:
                log_line += f"{met.upper()}={res.get(met, float('nan')):.4f} "
            logger.info(log_line)

    logger.info(f"\n--- Summary: Final Test Metrics (DeepSurv Multi-Task Models - {args.prediction_horizon_days}-Day Future) ---")
    for res_ds in all_ds_results:
        if res_ds and isinstance(res_ds, dict):
            log_line_ds = f"Model={res_ds.get('model_name', 'N/A')} "
            for met_ds in ["class_auroc", "class_auprc", "class_f1", "class_accuracy"]: # Key classification metrics
                log_line_ds += f"{met_ds.upper()}={res_ds.get(met_ds, float('nan')):.4f} "
            log_line_ds += f"C_INDEX_TTE={res_ds.get('concordance_index_tte', float('nan')):.4f}"
            logger.info(log_line_ds)

    # --- Analyze "Switches" for N-Day Future Predictions ---
    logger.info(f"\n--- Analyzing First Prediction of {args.prediction_horizon_days}-Day Future Event on Test Set ---")
    all_switch_dfs = []
    # Classification models
    if len(test_loader_cl) > 0 and args.epochs > 0 : # Only if test data and models were trained
        for name, model_obj in trained_cl_models.items():
            df_preds = predict_label_switches(model_obj, test_loader_cl, device, is_deepsurv_model=False)
            if not df_preds.empty:
                df_sw = analyze_switches_Nday_future(df_preds, args.prediction_horizon_days)
                df_sw["ModelType"] = name
                all_switch_dfs.append(df_sw)
                logger.info(f"{args.prediction_horizon_days}-Day Future Event Pred Switch Analysis for {name}:\n{df_sw.head(3)}")
                valid_diffs = df_sw[f"SwitchDifference_{args.prediction_horizon_days}DayFuture"].dropna()
                if not valid_diffs.empty: logger.info(f"{name} - Mean SwitchDiff: {valid_diffs.mean():.2f}, Median: {valid_diffs.median():.2f} visits")
    # DeepSurv models
    if len(test_loader_ds) > 0 and args.epochs > 0:
        for name, model_obj in trained_ds_models.items():
            df_preds_ds = predict_label_switches(model_obj, test_loader_ds, device, is_deepsurv_model=True)
            if not df_preds_ds.empty:
                df_sw_ds = analyze_switches_Nday_future(df_preds_ds, args.prediction_horizon_days)
                df_sw_ds["ModelType"] = name
                all_switch_dfs.append(df_sw_ds)
                logger.info(f"{args.prediction_horizon_days}-Day Future Event Pred Switch Analysis for {name}:\n{df_sw_ds.head(3)}")
                valid_diffs_ds = df_sw_ds[f"SwitchDifference_{args.prediction_horizon_days}DayFuture"].dropna()
                if not valid_diffs_ds.empty: logger.info(f"{name} - Mean SwitchDiff: {valid_diffs_ds.mean():.2f}, Median: {valid_diffs_ds.median():.2f} visits")
    
    if all_switch_dfs:
        combined_sw_df = pd.concat(all_switch_dfs, ignore_index=True)
        sw_out_path = os.path.join(f"./{args.prediction_horizon_days}day_future_prediction_outputs", f"all_models_{args.prediction_horizon_days}day_future_switch_analysis.csv")
        combined_sw_df.to_csv(sw_out_path, index=False)
        logger.info(f"Combined {args.prediction_horizon_days}-day future switch analysis saved to: {sw_out_path}")

    logger.info("Script finished.")

if __name__ == "__main__":
    main()
