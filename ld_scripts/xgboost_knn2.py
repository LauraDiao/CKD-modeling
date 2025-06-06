#!/usr/bin/env python

import os
import logging
import argparse
import numpy as np
import pandas as pd
# Removed: import torch
# Removed: import torch.nn as nn
from tqdm import tqdm
# Removed: from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import xgboost as xgb
import joblib
from datetime import timedelta

# changes
prediction_period = 1095 # 365, 730, 1095
embedding_size = "/ckd_embeddings_full"
embedding_path =  "./../../../commonfilesharePHI/slee/ckd" + embedding_size
years = str(round(prediction_period/365))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"xgboost_survival_tte_{years}year_future.log", mode='w'), # Updated log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Highlight Start: Removed DeepSurv PyTorch model and related classes/functions
# No DeepSurv class or SurvivalDataset class needed for XGBoost
# No negative_log_likelihood, train_deepsurv functions needed for XGBoost

def evaluate_survival_model(risk_scores, events, times):
    # Calculate concordance index (C-index)
    # This C-index calculation is adapted from the previous DeepSurv version
    # and can be replaced with a more robust library like `lifelines` if available.
    num_pairs = 0
    concordant_pairs = 0
    tied_pairs = 0

    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            time_i, event_i, risk_i = times[i], events[i], risk_scores[i]
            time_j, event_j, risk_j = times[j], events[j], risk_scores[j]

            # Only consider comparable pairs
            if (time_i < time_j and event_i == 1) or \
               (time_j < time_i and event_j == 1):
                num_pairs += 1
                if (risk_i > risk_j and time_i < time_j) or \
                   (risk_j > risk_i and time_j < time_i):
                    concordant_pairs += 1
                elif risk_i == risk_j:
                    tied_pairs += 1
            elif time_i == time_j and event_i == 1 and event_j == 1:
                # Both events occurred at the same time and are events.
                if risk_i == risk_j:
                    tied_pairs += 1
                # else: (risk_i != risk_j) - this is a discordant pair, but common C-index definitions
                # don't strictly count this as concordant or discordant for tied times.
                # The lifelines library handles these nuances better.
                pass

    if num_pairs == 0:
        c_index = 0.5
    else:
        c_index = (concordant_pairs + 0.5 * tied_pairs) / num_pairs
    return c_index
# Highlight End

def parse_args():
    parser = argparse.ArgumentParser(description="CKD survival analysis with XGBoost Cox model.") # Modified description
    parser.add_argument("--embedding-root", type=str, default="./ckd_embeddings_full", help="Path to embeddings.")
    parser.add_argument("--prediction-horizon-days", type=int, default=1095, help="Prediction horizon in days for survival analysis.")
    # Removed: --epochs, --batch-size, --learning-rate as these are for PyTorch
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds for XGBoost.") # New argument
    parser.add_argument("--learning-rate-xgb", type=float, default=0.1, help="Learning rate for XGBoost.") # New argument
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum tree depth for XGBoost.") # New argument
    return parser.parse_args()

def load_data(embedding_path, prediction_horizon_days):
    logger.info(f"Loading data from {embedding_path}...")
    all_embeddings = []
    all_pids = []
    all_event_status = []
    all_time_to_event = []

    # Placeholder for actual data loading logic
    # In a real scenario, you would load your embeddings and corresponding
    # 'event' (e.g., CKD progression within X days) and 'time_to_event' labels.
    # For demonstration, let's create dummy data.
    num_samples = 1000
    embed_dim = 768 # Assuming the default embedding dimension from tte_1year.py
    rng = np.random.default_rng(42)

    for i in range(num_samples):
        all_embeddings.append(rng.random(embed_dim))
        all_pids.append(f"patient_{i}")

        event_occurred = rng.choice([0, 1], p=[0.7, 0.3])
        all_event_status.append(event_occurred)

        if event_occurred:
            time_to_event = rng.integers(1, prediction_horizon_days + 1)
        else:
            time_to_event = prediction_horizon_days

        all_time_to_event.append(time_to_event)

    df_data = pd.DataFrame({
        'embedding': all_embeddings,
        'pid': all_pids,
        'event': all_event_status,
        'time_to_event': all_time_to_event
    })
    logger.info(f"Loaded {len(df_data)} samples.")
    return df_data

# Removed: predict_label_switches_sklearn
# Removed: analyze_switches_Nday_future

def main():
    args = parse_args()
    # Removed: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Removed: logger.info(f"Using device: {device}")

    df_data = load_data(embedding_path, args.prediction_horizon_days)

    X = np.array(df_data['embedding'].tolist())
    event = df_data['event'].values
    time_to_event = df_data['time_to_event'].values
    pids = df_data['pid'].values

    # Highlight Start: Prepare labels for XGBoost survival objective
    # XGBoost's `survival:cox` objective expects a specific format for labels:
    # - Time for uncensored samples should be positive.
    # - Time for censored samples should be negative.
    # We will encode event=0 (censored) as negative time, event=1 (event) as positive time.
    # This also aligns with the output structure you want for `tte_cox_true_time`.
    y_survival = np.where(event == 1, time_to_event, -time_to_event)

    X_train, X_test, y_train_survival, y_test_survival, pids_train, pids_test = train_test_split(
        X, y_survival, pids, test_size=0.2, random_state=42, stratify=event # Stratify by original event
    )

    # Re-extract original event and time for evaluation metrics for the test set
    # This is because y_test_survival has been modified for XGBoost's objective.
    event_test_original = np.where(y_test_survival > 0, 1, 0)
    time_test_original = np.abs(y_test_survival)


    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Highlight Start: Define and train XGBoost survival model
    model = xgb.XGBRegressor(
        objective='survival:cox',
        eval_metric='cox-nloglik', # Or 'cindex' if using a custom metric
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate_xgb,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist' # Often faster for larger datasets
    )

    logger.info("--- Starting XGBoost Survival Training ---")
    model.fit(X_train, y_train_survival,
              # Optional: Add early stopping for better performance
              # eval_set=[(X_test, y_test_survival)],
              # early_stopping_rounds=10,
              verbose=False) # Set to True for verbose output during training

    logger.info("--- Evaluating XGBoost Survival on Test Set ---")
    # Get risk scores (relative risk)
    all_risk_scores_test = model.predict(X_test)

    # Use the original event and time for C-index calculation
    c_index = evaluate_survival_model(all_risk_scores_test, event_test_original, time_test_original)
    logger.info(f"XGBoost Survival Test C-index: {c_index:.4f}")

    # You might want to save the trained model
    model_save_path = os.path.join(f"./{args.prediction_horizon_days}day_future_prediction_outputs",
                                   f"xgboost_survival_model_tte_{years}year.json") # XGBoost models can be saved as JSON
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path) # Use save_model for XGBoost
    logger.info(f"Trained XGBoost Survival model saved to {model_save_path}")

    # Prepare the output DataFrame for tte_cox risk data
    df_output = pd.DataFrame({
        "PatientID": pids_test,
        # Classification logits/probs are not directly from a survival model, set to NaN or remove if not applicable
        "cl_logit_0": [np.nan] * len(pids_test),
        "cl_logit_1": [np.nan] * len(pids_test),
        "cl_prob_1": [np.nan] * len(pids_test),
        "cl_true_label": [np.nan] * len(pids_test), # Survival analysis doesn't have a single 'true_label' for classification
        "tte_cox_risk_score": all_risk_scores_test,
        "tte_cox_true_time": time_test_original, # Original time_to_event
        "tte_cox_true_event": event_test_original # Original event status
    })

    output_dir = f"./{args.prediction_horizon_days}day_future_prediction_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"xgboost_survival_predictions_{years}year.csv")
    df_output.to_csv(output_filename, index=False)
    logger.info(f"XGBoost survival predictions saved to {output_filename}")
    # Highlight End


if __name__ == "__main__":
    main()