# %%
#!/usr/bin/env python
# test for xgboost with pytortch

import os
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    make_scorer # change
)
# Removed: from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
# Removed: import lightgbm as lgb
import joblib
from datetime import timedelta
import warnings

# change variables
prediction_period = 365 # 365, 730, 1095
years = str(round(prediction_period/365))
window_size = 365
filtering_stage = True
output_dir = ""


#%%
# comment out
import sys
sys.argv=['']
print("test")
#%%

subset = False
if not subset: 
    # print(f"cuda:{str(cuda_num)}")
    tab_path = f"./tabular_full/processed_tab_eskd_v5.csv"    
    output_dir += "_full"
if subset: 
    # print(f"cuda:{str(cuda_num)}")
    size = "1000" # 10, 100, full 
    tab_path = f"./tabular_subset_{size}/processed_tab_eskd_v5.csv"
    output_dir += f"_subset_{size}"

if filtering_stage: 
    output_dir += "_stage_filter" # ""

baseline = '_eskd'
output_dir += baseline

version = '_v5'
output_dir += version
print(output_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"./log_files/xgboost_only_model_tte_{years}year_future.log", mode='w'), # Updated log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# redirect warnings to the logger
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Custom warning handler that redirects all warnings to the logger's info level.
    """
    logger.info(f"Warning: {category.__name__}: {message} (File: {filename}, Line: {lineno})")

# Set the custom warning handler
warnings.showwarning = custom_showwarning

def parse_args():
    parser = argparse.ArgumentParser(description="CKD classification (n-year future window) and time-to-event training with XGBoost models (no early stopping).") # Updated description
    parser.add_argument("--tabular-data-file", type=str, default=tab_path, help="Path to the processed tabular CKD data CSV file.")
    parser.add_argument("--window-size", type=int, default=window_size, help="Sequence window size.")
    parser.add_argument("--embed-dim", type=int, default=768, help="Dimensionality of embeddings.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-patients", type=int, default=None, help="If set, only use data for up to this many patients.")
    parser.add_argument("--output-model-prefix", type=str, default=f"best_tab_model_{years}yr_future_xgboost_only_no_es", help="Filename prefix for saved models.") # Updated prefix
    parser.add_argument("--log-tte", action="store_true", help="Apply log transformation to time-to-event targets for Cox modeling.")
    parser.add_argument("--prediction-horizon-days", type=int, default=prediction_period, help="Number of days into the future to check for an event for label generation.")

    # XGBoost specific arguments
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Number of estimators for XGBoost.")
    parser.add_argument("--xgb-max-depth", type=int, default=6, help="Max depth for XGBoost.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1, help="Learning rate for XGBoost.")

    # Removed LightGBM and KNN arguments

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


def concordance_index(event_times, predicted_scores, event_observed):
    event_times = np.asarray(event_times)
    predicted_scores = np.asarray(predicted_scores)
    event_observed = np.asarray(event_observed).astype(int)

    nan_mask = np.isnan(event_times) | np.isnan(predicted_scores) | np.isnan(event_observed)
    if np.any(nan_mask):
        event_times = event_times[~nan_mask]
        predicted_scores = predicted_scores[~nan_mask]
        event_observed = event_observed[~nan_mask]

    if len(event_times) < 2: return 0.5

    concordant_pairs = 0
    num_comparable_pairs = 0

    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            is_comparable = False
            if event_observed[i] == 1 and event_observed[j] == 1:
                if event_times[i] != event_times[j]:
                    is_comparable = True
            elif event_observed[i] == 1 and event_observed[j] == 0:
                if event_times[i] < event_times[j]:
                    is_comparable = True
            elif event_observed[j] == 1 and event_observed[i] == 0:
                if event_times[j] < event_times[i]:
                    is_comparable = True
            
            if is_comparable:
                num_comparable_pairs += 1
                if event_times[i] < event_times[j]: 
                    if predicted_scores[i] > predicted_scores[j]: concordant_pairs += 1
                    elif predicted_scores[i] == predicted_scores[j]: concordant_pairs += 0.5
                elif event_times[j] < event_times[i]: 
                    if predicted_scores[j] > predicted_scores[i]: concordant_pairs += 1
                    elif predicted_scores[j] == predicted_scores[i]: concordant_pairs += 0.5
    if num_comparable_pairs == 0:
        return 0.5
    return concordant_pairs / num_comparable_pairs


def compute_metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    try:
        cm = confusion_matrix(labels, preds)
        if cm.size == 1: 
             if labels[0] == 0 : tn, fp, fn, tp = len(labels), 0,0,0
             else: tn, fp, fn, tp = 0,0,0,len(labels)
        elif cm.shape == (2,2): 
            tn, fp, fn, tp = cm.ravel()
        else: 
            tp = np.sum((labels == 1) & (preds == 1))
            tn = np.sum((labels == 0) & (preds == 0))
            fp = np.sum((labels == 0) & (preds == 1))
            fn = np.sum((labels == 1) & (preds == 0))
    except ValueError: 
        tp = np.sum((labels == 1) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
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

        if len(sample_labels) == 0: continue 
        
        temp_result = compute_metrics_at_threshold(sample_labels, sample_probs, threshold)
        for k_single in all_keys:
            metric_samples[k_single].append(temp_result.get(k_single, np.nan))
            
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


#  dead function
def build_sequences_1year_future_label(metadata_df, window_size, prediction_horizon_days, for_multitask=False):
    data_records = [] 
    required_cols = ["PatientID", "EventDate", "embedding", "label"]
    if for_multitask:
        required_cols.extend(["time_to_first_progression", "event_for_cox"])
    if not all(col in metadata_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in metadata_df.columns]
        raise ValueError(f"Metadata missing required columns: {missing}. Have: {metadata_df.columns}")

    metadata_df["EventDate"] = pd.to_datetime(metadata_df["EventDate"])
    prediction_timedelta = timedelta(days=prediction_horizon_days)

    for pid, group in tqdm(metadata_df.groupby("PatientID"), desc=f"Building sequences", leave=False, disable=True):
        group = group.sort_values(by="EventDate").reset_index(drop=True)
        
        embeddings_list = list(group["embedding"])
        original_labels_list = list(group["label"]) 
        event_dates_list = list(group["EventDate"])
        
        if for_multitask:
            tte_cox_list = list(group["time_to_first_progression"])
            event_cox_list = list(group["event_for_cox"])

        for i in range(len(group)): 
            start_idx_context = max(0, i - window_size + 1)
            context_embeddings = embeddings_list[start_idx_context : i + 1]

            if not context_embeddings: continue

            current_visit_date = event_dates_list[i]
            horizon_end_date = current_visit_date + prediction_timedelta
            
            future_event_occurs = 0
            for j in range(i + 1, len(group)): 
                future_visit_date = event_dates_list[j]
                if future_visit_date <= horizon_end_date: 
                    if original_labels_list[j] == 1: 
                        future_event_occurs = 1
                        break 
                else: 
                    break 
            
            record_tuple = (
                context_embeddings,
                future_event_occurs, 
                pid,
                i 
            )
            if for_multitask:
                record_tuple += (tte_cox_list[i], event_cox_list[i])
            data_records.append(record_tuple)
    return data_records

# Adapted build_sequences for the new 1-year future label
def build_sequences(meta, window_size, target_label_col=f'label_ckd_{years}_year_future', feature_col='embedding'):
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
        event_dates = list(group["EventDate"])
        ckd_stages = list(group["CKD_stage_clean"])

        for k in range(len(group)): # For each visit k
            context_end_idx = k + 1
            context_start_idx = max(0, k - window_size + 1)
            
            # context is a list of feature lists/arrays from previous time steps
            context = feature_sequences[context_start_idx : context_end_idx]
            target = classification_labels[k] # label_ckd_{years}_year_future for visit k
            
            if not context: 
                # This case should ideally not be hit if group is non-empty and window_size >= 1
                # For window_size=0 (current features only), context would be features_sequences[k:k+1]
                # Let's ensure context is never empty, if k=0 and window_size=1, context is features_sequences[0:1]
                logger.warning(f"Empty context for PatientID {pid}, visit index {k}. Skipping.")
                continue
            
            # sequence_data.append((context, target, pid, k)) 
            sequence_data.append((context, target, pid, k, event_dates[k], ckd_stages[k]))

    return sequence_data

# Adapted build_sequences_for_multitask
def build_sequences_for_multitask0(meta, window_size, 
                                  classification_target_col='label_ckd_1_year_future',
                                  tte_time_col='time_until_progression', # Was 'time_until_progression'
                                  # Removed tte_event_col as it's taken from classification_target_col by current DeepSurv train loop
                                  tte_event_col='event_for_cox_indicator', 
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
        event_dates = list(group["EventDate"])
        ckd_stages = list(group["CKD_stage_clean"])

        for k in range(len(group)):
            context_end_idx = k + 1
            context_start_idx = max(0, k - window_size + 1)
            context = feature_sequences[context_start_idx : context_end_idx]
            
            classification_target = classification_labels[k]
            time_for_tte = tte_values[k] 
            # The 'event' for Cox loss in the existing train_and_evaluate_deepsurv is `event_batch`,
            # which will be `classification_target`.
            
            if not context: continue
            # duplicate clf target for event
            # 5 items
            # data_records.append((context, classification_target, pid, k, time_for_tte, classification_target))
            # 7 items
            data_records.append((
                context, 
                classification_target, 
                pid, 
                k, 
                time_for_tte, 
                classification_target, 
                event_dates[k], 
                ckd_stages[k]
                ))

    return data_records

def build_sequences_for_multitask(meta, window_size, 
                                  classification_target_col='label_ckd_1_year_future',
                                  tte_time_col='time_until_progression',
                                  tte_event_col='event_for_cox_indicator',  # update; event
                                  feature_col='embedding'):
    data_records = []
    required_cols = [classification_target_col, tte_time_col, tte_event_col, feature_col]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Required column '{col}' not found for build_sequences_for_multitask.")

    for pid, group in meta.groupby("PatientID"):
        group = group.sort_values(by="EventDate").reset_index(drop=True)
        
        feature_sequences = list(group[feature_col])
        classification_labels = list(group[classification_target_col])
        tte_values = list(group[tte_time_col])
        cox_event_indicators = list(group[tte_event_col])  # update
        event_dates = list(group["EventDate"])
        ckd_stages = list(group["CKD_stage_clean"])

        for k in range(len(group)):
            context_end_idx = k + 1
            context_start_idx = max(0, k - window_size + 1)
            context = feature_sequences[context_start_idx : context_end_idx]
            
            classification_target = classification_labels[k]
            time_for_tte = tte_values[k]
            cox_event = cox_event_indicators[k]  # update
            
            if not context: continue
            # record[5] is cox event indicator
            data_records.append((
                context, 
                classification_target, 
                pid, 
                k, 
                time_for_tte, 
                cox_event, # update
                event_dates[k], 
                ckd_stages[k]
                ))

    return data_records

def prepare_sklearn_data(sequence_records, window_size, embed_dim, for_survival=False):
    X_list, y_cls_list, pids_list, local_indices_list = [], [], [], []
    # added
    dates_list, stages_list = [], []
    y_time_list, y_event_list = [], [] 

    for record in sequence_records:
        context_embeddings = record[0]
        classification_target_label = record[1]
        pid = record[2]
        local_idx = record[3]
        # added
        event_date = record[6] if for_survival else record[4]
        event_stage = record[7] if for_survival else record[5]

        context_padded = pad_sequence(list(context_embeddings), window_size, embed_dim)
        X_list.append(context_padded.flatten()) 
        y_cls_list.append(classification_target_label)
        pids_list.append(pid)
        local_indices_list.append(local_idx)
        dates_list.append(event_date)
        stages_list.append(event_stage)

        if for_survival:
            # break
            tte_for_cox = record[4]
            event_for_cox = record[5]
            y_time_list.append(tte_for_cox)
            y_event_list.append(event_for_cox)

    X_array = np.array(X_list)
    y_cls_array = np.array(y_cls_list)
    
    # add dates list 
    if for_survival:
        y_time_array = np.array(y_time_list)
        y_event_array = np.array(y_event_list)
        valid_survival_mask = ~np.isnan(y_time_array) & ~np.isnan(y_event_array)
        
        return (X_array, y_cls_array, pids_list, local_indices_list, dates_list, stages_list, 
                y_time_array, y_event_array, valid_survival_mask)
    else:
        return X_array, y_cls_array, pids_list, local_indices_list, dates_list, stages_list


def train_and_evaluate_classifier(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test_cls, pids_test, local_indices_test, dates_test, stages_test, args):
    logger.info(f"Starting {model_name} training (Classification: {args.prediction_horizon_days}-day future label).")
    
    # model.fit(X_train, y_train)
    # early stopping for baseline
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    logger.info(f"{model_name}: Trained for specified number of estimators/iterations.")
    
    model_path = f"./joblib_files/{args.output_model_prefix}_{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"{model_name}: Model saved to {model_path}")

    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_preds_test_logits = model.predict_proba(X_test) 

    final_results_dict = {"model_name": model_name}
    if len(y_test_cls) > 0:
        prevalence = np.mean(y_test_cls)
        logger.info(f"{model_name} Test Prevalence ({args.prediction_horizon_days}-day future label): {prevalence:.4f}")
        # update 
        print("prevalence: ", prevalence)
        threshold = prevalence if 0 < prevalence < 1 else 0.5

        metrics_raw = compute_metrics_at_threshold(y_test_cls, y_probs_test, threshold)
        metrics_ci = bootstrap_metrics(y_test_cls, y_probs_test, threshold, random_state=args.random_seed)

        logger.info(f"{model_name} Classification Threshold set to {threshold:.4f}.")
        for k_m in ["accuracy", "precision", "recall", "f1", "ppv", "npv", "auroc", "auprc", "tp", "tn", "fp", "fn"]:
            raw_v = metrics_raw.get(k_m, np.nan)
            ci_m, ci_l, ci_h = metrics_ci.get(k_m, (np.nan, np.nan, np.nan))
            logger.info(f"{model_name} {k_m.upper()}: {raw_v:.4f} (CI Mean: {ci_m:.4f} [{ci_l:.4f}-{ci_h:.4f}])")
            final_results_dict[k_m] = raw_v
    else:
        logger.warning(f"{model_name}: No targets in test set for classification evaluation.")

    output_dir_dets = f"./{args.prediction_horizon_days}day_future_prediction_outputs" + output_dir
    os.makedirs(output_dir_dets, exist_ok=True)
    logit_0_col = ([np.nan] * len(y_test_cls))
    logit_1_col = ([np.nan] * len(y_test_cls))
    if y_preds_test_logits.ndim == 2:
        if y_preds_test_logits.shape[1] > 0:
            logit_0_col = y_preds_test_logits[:, 0]
        if y_preds_test_logits.shape[1] > 1:
            logit_1_col = y_preds_test_logits[:, 1]
        elif y_preds_test_logits.shape[1] == 1 : 
            logit_1_col = y_preds_test_logits[:, 0] 
            logit_0_col = 1 - logit_1_col

    df_dets = pd.DataFrame({
        "PatientID": pids_test,
        "EventDate": dates_test,
        "CKD_stage_numeric": stages_test,
        "LocalIndex": local_indices_test,
        "cl_logit_0": logit_0_col,
        "cl_logit_1": logit_1_col,
        "cl_prob_1": y_probs_test, # "prob_positive"
        "cl_true_label": y_test_cls,
        
    })
    out_csv_p = os.path.join(output_dir_dets, f"{model_name}_detailed_outputs_classification.csv")
    df_dets.to_csv(out_csv_p, index=False)
    logger.info(f"{model_name}: Detailed classification test outputs saved to {out_csv_p}")
    
    return final_results_dict, model

def train_and_evaluate_xgboost_survival(model, model_name, X_train_s, y_train_time_s, y_train_event_s, 
                                   X_val_s, y_val_time_s, y_val_event_s,
                                   X_test_s, y_test_time_s, y_test_event_s, 
                                   pids_test_s, local_indices_test_s, 
                                   dates_test_s, stages_test_s,
                                   y_test_cls_s,
                                   args):
    logger.info(f"Starting {model_name} training (Survival TTE to first progression).")
    y_train_xgb_surv = np.where(y_train_event_s == 1, y_train_time_s, -y_train_time_s)
    y_val_xgb_surv = np.where(y_val_event_s == 1, y_val_time_s, -y_val_time_s)

    # change
    # true_label 1
    # surv_true_label = y_test_cls_s
    # true_label 2
    surv_true_label = np.where(y_test_event_s == 1, y_test_time_s, -y_test_time_s)
    
    # fit 1
    # model.fit(X_train_s, y_train_xgb_surv)
    
    model.fit(
        X_train_s, y_train_xgb_surv,
        eval_set=[(X_val_s, y_val_xgb_surv)],
        verbose=False
    )
    # fit 2
    # model.fit(X_train_s, y_train_time_s, sample_weight = y_train_event_s)
    
    logger.info(f"{model_name}: Trained for specified number of estimators.")

    model_path = f"./joblib_files/{args.output_model_prefix}_{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"{model_name}: Survival model saved to {model_path}")

    # change
    risk_scores_test = model.predict(X_test_s)
    cl_prob_1_survival = 1 / (1 + np.exp(-risk_scores_test)) # sigmoid function

    final_results_dict = {"model_name": model_name + "_Survival"}
    if len(y_test_time_s) > 1 and np.sum(y_test_event_s) > 0:
        c_idx_tte = concordance_index(y_test_time_s, risk_scores_test, y_test_event_s)
        logger.info(f"{model_name} Concordance Index (TTE to first prog): {c_idx_tte:.4f}")
        final_results_dict["concordance_index_tte"] = c_idx_tte
    else:
        logger.warning(f"{model_name}: Not enough valid data or no events to calculate C-index for TTE.")
        final_results_dict["concordance_index_tte"] = np.nan

    output_dir_dets = f"./{args.prediction_horizon_days}day_future_prediction_outputs" + output_dir
    os.makedirs(output_dir_dets, exist_ok=True)
    df_surv_dets = pd.DataFrame({
        "PatientID": pids_test_s,
        "EventDate": dates_test_s,
        "CKD_stage_numeric": stages_test_s,
        "LocalIndex": local_indices_test_s,
        "cl_prob_1":  risk_scores_test, 
        "cl_true_label": surv_true_label,
        "tte_cox_risk_score": risk_scores_test,
        "tte_cox_true_time": y_test_time_s,
        "tte_cox_true_event": y_test_event_s
        
    })
    out_csv_surv_p = os.path.join(output_dir_dets, f"{model_name}_detailed_outputs_survival.csv")
    df_surv_dets.to_csv(out_csv_surv_p, index=False)
    logger.info(f"{model_name}: Detailed survival test outputs saved to {out_csv_surv_p}")

    return final_results_dict

# Removed train_and_evaluate_lgbm_survival function

def predict_label_switches_sklearn(model, X_data, y_true_labels, pids, local_indices, N_days_horizon):
    preds = model.predict(X_data) 
    records = []
    for pid_val, true_lbl, pred_lbl, local_idx_val in zip(pids, y_true_labels, preds, local_indices):
        records.append((
            pid_val,
            local_idx_val,
            true_lbl,
            pred_lbl
        ))
    df = pd.DataFrame(records, columns=["PatientID", "LocalIndex", f"TrueLabel_{N_days_horizon}DayFuture", f"PredLabel_{N_days_horizon}DayFuture"])
    return df


def analyze_switches_Nday_future(df_preds_Nday, N_days_horizon):
    analysis_records = []
    true_label_col = f"TrueLabel_{N_days_horizon}DayFuture"
    pred_label_col = f"PredLabel_{N_days_horizon}DayFuture"

    if not {true_label_col, pred_label_col}.issubset(df_preds_Nday.columns):
        logger.error(f"Required columns missing: Need {true_label_col}, {pred_label_col}. Got {df_preds_Nday.columns}")
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
    global args 
    args = parse_args()
    logger.info(f"Running with configuration for {args.prediction_horizon_days}-DAY FUTURE PREDICTION (XGBoost ONLY - NO EARLY STOPPING):") # Updated description
    for key, val in vars(args).items(): logger.info(f"{key}: {val}")

    np.random.seed(args.random_seed)

    logger.info(f"Loading tabular CKD data from: {args.tabular_data_file}")
    try:
        metadata = pd.read_csv(args.tabular_data_file, parse_dates=["EventDate"])
    except FileNotFoundError:
        logger.error(f"Tabular data file not found: {args.tabular_data_file}. Exiting.")
        return

    # CKD Stage Cleaning (as in original tabular script, adapted)
    if 'CKD_stage_numeric' in metadata.columns:
        metadata['CKD_stage_clean'] = metadata['CKD_stage_numeric'].apply(clean_ckd_stage)
        # Fill missing stages within a patient's record
        metadata['CKD_stage_clean'] = metadata.groupby('PatientID')['CKD_stage_clean'].bfill().ffill()
        metadata = metadata.dropna(subset=['CKD_stage_clean']) # Remove patients with no stage info
        metadata['CKD_stage_clean'] = metadata['CKD_stage_clean'].astype(int)
    else:
        logger.error("'CKD_stage' column not found in tabular data. Cannot proceed with label generation.")
        return true

    # Label 1: Current CKD stage >= 4
    metadata['label_ckd_stage_4_plus'] = metadata['CKD_stage_clean'].apply(lambda x: 1 if x >= 4 else 0)
    logger.info(f"Value counts for 'label_ckd_stage_4_plus':\n{metadata['label_ckd_stage_4_plus'].value_counts(dropna=False).to_string()}")


    # Time-to-event preprocessing (using the adapted tabular version)
    logger.info("Preprocessing time-to-event data (time_until_progression to first CKD stage 4+).")
    metadata = time_to_event_preprocessing_tabular(metadata, source_label_col='label_ckd_stage_4_plus',
                                             time_col_name='time_until_progression',
                                             event_indicator_col_name='event_for_cox_indicator') # This event_indicator is not directly used by current deepsurv loop

    # print(metadata.head())
    # Label 2: CKD stage 4+ within 1 year (args.prediction_horizon_days)
    metadata = add_future_event_label_column(
        metadata,
        source_label_col='label_ckd_stage_4_plus',
        new_label_col=f'label_ckd_{years}_year_future',
        horizon_days=args.prediction_horizon_days
    )
    logger.info(f"Value counts for 'label_ckd_{years}_year_future':\n{metadata[f'label_ckd_{years}_year_future'].value_counts(dropna=False).to_string()}")
    logger.info(f"Metadata with new labels (first 3 rows):\n{metadata.head(3).to_string()}")

    # Feature Selection: Use all columns except identifiers, raw date/stage, and created labels/TTE info
    exclude_cols = ['PatientID', 
                    'EventDate', 
                    'EventMonth',
                    'CKD_stage', # Raw stage column
                    'CKD_stage_clean', # Intermediate cleaned stage
                    'CKD_stage_numeric',
                    # 'CKD_stage_numeric_right'
                    'max_stage',
                    # 'max_stage_right'
                    "META_1",
                    'label_ckd_stage_4_plus', f'label_ckd_{years}_year_future', # Generated labels
                    'time_until_progression', 'event_for_cox_indicator'] # Generated TTE info
    
    # Also consider other known non-feature columns from the original script if any (e.g. 'GFR_combined')
    # changed to ICD
    if 'ICD_combined' in metadata.columns: # Example if it was a target or identifier
         exclude_cols.append('ICD_combined')
# %%
    feature_cols = []
    for c in potential_feature_cols:
        if metadata[c].dtype.kind in 'biufc':  # bool/int/uint/float/complex
            feature_cols.append(c)
            continue

        coerced = pd.to_numeric(metadata[c], errors='coerce')
        n_bad = coerced.isna().sum() - metadata[c].isna().sum()  # newly-NaN, not originally-NaN
        if coerced.notna().any():
            logger.warning(
                f"Column {c} (dtype={metadata[c].dtype}) coerced to numeric; "
                f"{n_bad} of {len(coerced)} values could not be parsed and became NaN."
            )
            metadata[c] = coerced
            feature_cols.append(c)
        else:
            logger.warning(f"Dropping column {c}: no valid numeric values after coercion.")
            
    # Convert boolean columns to int (0 or 1)
    for col in feature_cols:
        if metadata[col].dtype == 'bool':
            metadata[col] = metadata[col].astype(int)
    # print(feature_cols)
    # if not feature_cols:
    #     logger.error("No feature columns selected. Check data and exclude_cols list. Exiting.")
    #     return
    if len(feature_cols) == 0:
        print("No feature columns selected. ")

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

    # print(metadata)

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

    print(train_metadata.head())
    print(test_metadata.head())
    # Fill any remaining NaNs in TTE columns after splits (e.g., for patients with no progression)
    # TTE values (time_until_progression) that are NaN because a patient never progressed (or progressed after last obs)
    # might need specific handling (e.g. fill with a large number if CoxPH requires non-NaN, or ensure masking handles it)
    # For now, the dataloader's getitem handles NaN tte_val.
    # The `time_to_event_preprocessing_tabular` already handles this for censored patients by giving time to last obs.
    # Nan-filling for feature columns was done when creating 'embedding' list.

    # tte processing

    # logger.info("Preprocessing TTE data for survival modeling component.")
    # metadata = time_to_event_preprocessing(metadata, log_transform_tte=args.log_tte)

    logger.info(f"Building sequences and preparing data for sklearn models...")
    train_sequences_class = build_sequences(train_metadata, args.window_size, target_label_col=f'label_ckd_{years}_year_future', feature_col='embedding')
    val_sequences_class = build_sequences(val_metadata, args.window_size, target_label_col=f'label_ckd_{years}_year_future', feature_col='embedding')
    test_sequences_class = build_sequences(test_metadata, args.window_size, target_label_col=f'label_ckd_{years}_year_future', feature_col='embedding')

    if not all([train_sequences_class, val_sequences_class, test_sequences_class]): 
        logger.error("One or more sequence sets are empty. Exiting."); return

    # print(train_sequences_class)
    train_sequences_surv = build_sequences_for_multitask(train_metadata, args.window_size, 
                                                          classification_target_col='label_ckd_1_year_future',
                                                          tte_time_col='time_until_progression',
                                                          tte_event_col='event_for_cox_indicator',
                                                          feature_col='embedding')
    val_sequences_surv = build_sequences_for_multitask(val_metadata, args.window_size,
                                                        classification_target_col='label_ckd_1_year_future',
                                                        tte_time_col='time_until_progression',
                                                        tte_event_col='event_for_cox_indicator',                                                        
                                                        feature_col='embedding')
    test_sequences_surv = build_sequences_for_multitask(test_metadata, args.window_size,
                                                         classification_target_col='label_ckd_1_year_future',
                                                         tte_time_col='time_until_progression',
                                                         tte_event_col='event_for_cox_indicator',
                                                         feature_col='embedding')

    # modified for date, stage output
    (X_train, y_train_cls, _, _, 
    _, _,
     y_train_time, y_train_event, train_survival_mask) = prepare_sklearn_data(train_sequences_surv, args.window_size, args.embed_dim, for_survival=True)
    (X_val, y_val_cls, _, _, 
    _, _,
     y_val_time, y_val_event, val_survival_mask) = prepare_sklearn_data(val_sequences_surv, args.window_size, args.embed_dim, for_survival=True)
    (X_test, y_test_cls, pids_test, local_indices_test, 
    dates_test, stages_test,
     y_test_time, y_test_event, test_survival_mask) = prepare_sklearn_data(test_sequences_surv, args.window_size, args.embed_dim, for_survival=True)

    logger.info(f"Sklearn data shapes: X_train: {X_train.shape}, y_train_cls: {y_train_cls.shape}")
    logger.info(f"Train survival data: {np.sum(train_survival_mask)} valid samples.")
    logger.info(f"Val survival data: {np.sum(val_survival_mask)} valid samples (not used for fitting).")
    logger.info(f"Test survival data: {np.sum(test_survival_mask)} valid samples.")

    all_results = []
    trained_classification_models = {} 

    # --- XGBoost Classifier ---
    # xgb_params = {
    # 'objective': 'binary:logistic',
    # 'eval_metric': ['logloss', 'aucpr'],
    # 'learning_rate': 0.05,
    # 'max_depth': 6,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
    # 'early_stopping_rounds':50,
    # 'n_estimators': 1000,
    # 'use_label_encoder': False,
    # # 'base_score':0.5,
    # }
    # xgb_clf = xgb.XGBClassifier(**xgb_params)
    
    # update 
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],
        random_state=args.random_seed,
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
    )
    
    results_xgb_cls, trained_xgb_cls = train_and_evaluate_classifier(
        xgb_clf, f"XGBoost_{args.prediction_horizon_days}DayFuture_Classifier",
        X_train, y_train_cls, X_val, y_val_cls, X_test, y_test_cls, 
        pids_test, local_indices_test, dates_test, stages_test,
        args
    )
    all_results.append(results_xgb_cls)
    trained_classification_models["XGBoost_Classifier"] = trained_xgb_cls

    # --- Survival Models ---
    X_train_s, y_train_time_s, y_train_event_s = X_train[train_survival_mask], y_train_time[train_survival_mask], y_train_event[train_survival_mask]
    X_val_s, y_val_time_s, y_val_event_s = X_val[val_survival_mask], y_val_time[val_survival_mask], y_val_event[val_survival_mask] 
    X_test_s, y_test_time_s, y_test_event_s = X_test[test_survival_mask], y_test_time[test_survival_mask], y_test_event[test_survival_mask]
    pids_test_s = [p for i, p in enumerate(pids_test) if test_survival_mask[i]]
    local_indices_test_s = [idx for i, idx in enumerate(local_indices_test) if test_survival_mask[i]]
    # added for survival
    dates_test_s = [d for i, d in enumerate(dates_test) if test_survival_mask[i]]
    # added
    stages_test_s = [s for i, s in enumerate(stages_test) if test_survival_mask[i]]
    # change
    #  true classification labels for the test set (before the survival mask is applied)
    y_test_cls_s = y_test_cls[test_survival_mask]

    # --- XGBoost Survival Model ---
    if X_train_s.shape[0] > 0 and X_test_s.shape[0] > 0: 
        # xgb_surv = xgb.XGBModel( 
        #     n_estimators=args.xgb_n_estimators,
        #     max_depth=args.xgb_max_depth,
        #     learning_rate=args.xgb_learning_rate,
        #     objective='survival:cox',
        #     # eval_metric='cox-nloglik', # leave out 
        #     random_state=args.random_seed,
        # )

        xgb_params = {
            'objective': 'survival:cox',
            'tree_method': 'hist',       # Highly recommended for speed/memory
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 1000,
            'eval_metric': 'cox-nloglik', # CRITICAL: Metric for survival
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50
        }

        # Use XGBRegressor for survival tasks
        xgb_surv = xgb.XGBRegressor(**xgb_params)

        results_xgb_surv = train_and_evaluate_xgboost_survival(
            xgb_surv, f"XGBoost_TTE_Survival",
            X_train_s, y_train_time_s, y_train_event_s,
            X_val_s if X_val_s.shape[0] > 0 else None, 
            y_val_time_s if X_val_s.shape[0] > 0 else None, 
            y_val_event_s if X_val_s.shape[0] > 0 else None,
            X_test_s, y_test_time_s, y_test_event_s,
            pids_test_s, local_indices_test_s,
            dates_test_s, stages_test_s,
            y_test_cls_s,
            args
        )
        all_results.append(results_xgb_surv)
    else:
        logger.warning("Skipping XGBoost Survival model training due to insufficient valid survival data.")

    # Removed LightGBM Survival model training section

    logger.info(f"\n--- Summary: Final Test Metrics ({args.prediction_horizon_days}-Day Future Prediction & TTE) ---")
    for res in all_results:
        if res and isinstance(res, dict):
            log_line = f"Model={res.get('model_name', 'N/A')} "
            if "concordance_index_tte" in res: 
                log_line += f"C_INDEX_TTE={res.get('concordance_index_tte', float('nan')):.4f}"
            else: 
                for met in ["auroc", "auprc", "f1", "accuracy", "precision", "recall", "ppv", "npv"]:
                    log_line += f"{met.upper()}={res.get(met, float('nan')):.4f} "
            logger.info(log_line)

    logger.info(f"\n--- Analyzing First Prediction of {args.prediction_horizon_days}-Day Future Event on Test Set ---")
    all_switch_dfs = []
    if len(X_test) > 0 and "XGBoost_Classifier" in trained_classification_models: 
        name = "XGBoost_Classifier"
        model_obj = trained_classification_models[name]
        df_preds = predict_label_switches_sklearn(model_obj, X_test, y_test_cls, pids_test, local_indices_test, args.prediction_horizon_days)
        if not df_preds.empty:
            df_sw = analyze_switches_Nday_future(df_preds, args.prediction_horizon_days)
            df_sw["ModelType"] = name
            all_switch_dfs.append(df_sw)
            logger.info(f"{args.prediction_horizon_days}-Day Future Event Pred Switch Analysis for {name}:\n{df_sw.head(3)}")
            valid_diffs = df_sw[f"SwitchDifference_{args.prediction_horizon_days}DayFuture"].dropna()
            if not valid_diffs.empty: logger.info(f"{name} - Mean SwitchDiff: {valid_diffs.mean():.2f}, Median: {valid_diffs.median():.2f} visits")
    
    if all_switch_dfs:
        combined_sw_df = pd.concat(all_switch_dfs, ignore_index=True)
        sw_out_path = os.path.join(f"./{args.prediction_horizon_days}day_future_prediction_outputs" + output_dir, f"xgboost_only_{args.prediction_horizon_days}day_future_switch_analysis.csv") # Updated filename
        combined_sw_df.to_csv(sw_out_path, index=False)
        logger.info(f"Combined {args.prediction_horizon_days}-day future switch analysis saved to: {sw_out_path}")

    logger.info("Script finished.")


# %%
import sys

# %%
sys.argv = ['']

# %%
if __name__ == "__main__":
    main()

# %%



