#!/usr/bin/env python

import os
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)
# Removed: from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
# Removed: import lightgbm as lgb
import joblib
from datetime import timedelta

# change variables
prediction_period = 1 # 365, 730, 1095
embedding_size = "10" # 10, 100, full
embedding_path =  "./../../../commonfilesharePHI/slee/ckd-optum/ckd_embeddings_" + embedding_size
years = str(round(prediction_period/365))
window_size = 50

# script_folder = f"xgb_{years}year_embeddings_{embedding_size}_files"
# try:
#     os.mkdir(script_folder)
# except FileExistsError:
#     pass
# end of variables

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"xgboost_only_model_tte_{years}year_future.log", mode='w'), # Updated log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CKD classification (n-year future window) and time-to-event training with XGBoost models (no early stopping).") # Updated description
    parser.add_argument("--embedding-root", type=str, default=embedding_path, help="Path to embeddings.")
    parser.add_argument("--window-size", type=int, default=window_size, help="Sequence window size.")
    parser.add_argument("--embed-dim", type=int, default=768, help="Dimensionality of embeddings.")
    parser.add_argument("--metadata-file", type=str, default="patient_embedding_metadata.csv", help="CSV with metadata.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-patients", type=int, default=None, help="If set, only load embeddings for up to this many patients.")
    parser.add_argument("--output-model-prefix", type=str, default=f"best_model_{years}yr_future_xgboost_only_no_es", help="Filename prefix for saved models.") # Updated prefix
    parser.add_argument("--log-tte", action="store_true", help="Apply log transformation to time-to-event targets for Cox modeling.")
    parser.add_argument("--prediction-horizon-days", type=int, default=prediction_period, help="Number of days into the future to check for an event for label generation.")

    # XGBoost specific arguments
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Number of estimators for XGBoost.")
    parser.add_argument("--xgb-max-depth", type=int, default=3, help="Max depth for XGBoost.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1, help="Learning rate for XGBoost.")

    # Removed LightGBM and KNN arguments

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
    if not seq:
        return np.zeros((length, dim), dtype=np.float32)
    if len(seq) < length:
        padding = [np.zeros(dim, dtype=np.float32)] * (length - len(seq))
        seq = padding + seq
    processed_seq = []
    for item in seq[-length:]:
        if isinstance(item, np.ndarray) and item.shape == (dim,):
            processed_seq.append(item)
        else:
            logger.warning(f"Unexpected item type or shape in sequence for padding. Expected ({dim},), got {type(item)} {getattr(item, 'shape', 'N/A')}. Using zeros.")
            processed_seq.append(np.zeros(dim, dtype=np.float32))
    return np.stack(processed_seq, axis=0)


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


def time_to_event_preprocessing(meta_df_input, log_transform_tte=False):
    meta = meta_df_input.copy()
    meta["EventDate"] = pd.to_datetime(meta["EventDate"])
    meta = meta.sort_values(by=["PatientID", "EventDate"]).reset_index(drop=True)

    meta["time_to_first_progression"] = np.nan
    meta["event_for_cox"] = 0

    for pid, group in tqdm(meta.groupby("PatientID"), desc="Preprocessing TTE", leave=False, disable=True):
        first_prog_rows = group[group["label"] == 1]
        if not first_prog_rows.empty:
            first_prog_date = first_prog_rows["EventDate"].min()
            first_prog_meta_idx = first_prog_rows["EventDate"].idxmin()

            for current_visit_idx in group.index:
                current_date = meta.loc[current_visit_idx, "EventDate"]
                if current_date <= first_prog_date:
                    delta_days = (first_prog_date - current_date).days
                    meta.loc[current_visit_idx, "time_to_first_progression"] = float(delta_days)
                    if current_visit_idx == first_prog_meta_idx:
                        meta.loc[current_visit_idx, "event_for_cox"] = 1
                    else:
                        meta.loc[current_visit_idx, "event_for_cox"] = 0
                else: 
                    meta.loc[current_visit_idx, "time_to_first_progression"] = np.nan 
                    meta.loc[current_visit_idx, "event_for_cox"] = 0 
        else: 
            if not group.empty:
                last_observed_date_for_patient = group["EventDate"].max()
                for current_visit_idx in group.index:
                    current_date = meta.loc[current_visit_idx, "EventDate"]
                    delta_to_last_obs = (last_observed_date_for_patient - current_date).days
                    meta.loc[current_visit_idx, "time_to_first_progression"] = float(delta_to_last_obs)
                    meta.loc[current_visit_idx, "event_for_cox"] = 0 

    if log_transform_tte:
        meta["time_to_first_progression"] = np.log(meta["time_to_first_progression"] + 1.0)
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

def prepare_sklearn_data(sequence_records, window_size, embed_dim, for_survival=False):
    X_list, y_cls_list, pids_list, local_indices_list = [], [], [], []
    y_time_list, y_event_list = [], [] 

    for record in sequence_records:
        context_embeddings = record[0]
        classification_target_label = record[1]
        pid = record[2]
        local_idx = record[3]

        context_padded = pad_sequence(list(context_embeddings), window_size, embed_dim)
        X_list.append(context_padded.flatten()) 
        y_cls_list.append(classification_target_label)
        pids_list.append(pid)
        local_indices_list.append(local_idx)

        if for_survival:
            tte_for_cox = record[4]
            event_for_cox = record[5]
            y_time_list.append(tte_for_cox)
            y_event_list.append(event_for_cox)

    X_array = np.array(X_list)
    y_cls_array = np.array(y_cls_list)
    
    if for_survival:
        y_time_array = np.array(y_time_list)
        y_event_array = np.array(y_event_list)
        valid_survival_mask = ~np.isnan(y_time_array) & ~np.isnan(y_event_array)
        
        return (X_array, y_cls_array, pids_list, local_indices_list, 
                y_time_array, y_event_array, valid_survival_mask)
    else:
        return X_array, y_cls_array, pids_list, local_indices_list


def train_evaluate_classifier(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test_cls, pids_test, local_indices_test, args):
    logger.info(f"Starting {model_name} training (Classification: {args.prediction_horizon_days}-day future label).")
    
    model.fit(X_train, y_train)
    logger.info(f"{model_name}: Trained for specified number of estimators/iterations.")
    
    model_path = f"{args.output_model_prefix}_{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"{model_name}: Model saved to {model_path}")

    y_probs_test = model.predict_proba(X_test)[:, 1]
    y_preds_test_logits = model.predict_proba(X_test) 

    final_results_dict = {"model_name": model_name}
    if len(y_test_cls) > 0:
        prevalence = np.mean(y_test_cls)
        logger.info(f"{model_name} Test Prevalence ({args.prediction_horizon_days}-day future label): {prevalence:.4f}")
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

    output_dir_dets = f"./{args.prediction_horizon_days}day_future_prediction_outputs"
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
        "LocalIndex": local_indices_test,
        "logit_0": logit_0_col,
        "logit_1": logit_1_col,
        "prob_positive": y_probs_test,
        "true_label": y_test_cls,
        
    })
    out_csv_p = os.path.join(output_dir_dets, f"{model_name}_detailed_outputs_classification.csv")
    df_dets.to_csv(out_csv_p, index=False)
    logger.info(f"{model_name}: Detailed classification test outputs saved to {out_csv_p}")
    
    return final_results_dict, model

def train_evaluate_xgboost_survival(model, model_name, X_train_s, y_train_time_s, y_train_event_s, 
                                   X_val_s, y_val_time_s, y_val_event_s,
                                   X_test_s, y_test_time_s, y_test_event_s, 
                                   pids_test_s, local_indices_test_s, 
                                   y_test_cls_s,
                                   args):
    logger.info(f"Starting {model_name} training (Survival TTE to first progression).")
    y_train_xgb_surv = np.where(y_train_event_s == 1, y_train_time_s, -y_train_time_s)
    
    model.fit(X_train_s, y_train_xgb_surv)
    logger.info(f"{model_name}: Trained for specified number of estimators.")

    model_path = f"{args.output_model_prefix}_{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"{model_name}: Survival model saved to {model_path}")

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

    output_dir_dets = f"./{args.prediction_horizon_days}day_future_prediction_outputs"
    os.makedirs(output_dir_dets, exist_ok=True)
    df_surv_dets = pd.DataFrame({
        "PatientID": pids_test_s,
        "LocalIndex": local_indices_test_s,
        "cl_prob_1":  cl_prob_1_survival, 
        "cl_true_label": y_test_cls_s,
        "tte_cox_risk_score": risk_scores_test,
        "tte_cox_true_time": y_test_time_s,
        "tte_cox_true_event": y_test_event_s
        
    })
    out_csv_surv_p = os.path.join(output_dir_dets, f"{model_name}_detailed_outputs_survival.csv")
    df_surv_dets.to_csv(out_csv_surv_p, index=False)
    logger.info(f"{model_name}: Detailed survival test outputs saved to {out_csv_surv_p}")

    return final_results_dict

# Removed train_evaluate_lgbm_survival function

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
    metadata['label'] = metadata['CKD_stage_clean'].apply(lambda x: 1 if x >= 4 else 0)
    logger.info(f"Rows after CKD stage cleaning & 'label' creation: {len(metadata)}")

    logger.info("Filtering for existing embedding files...")
    metadata = metadata[metadata.apply(lambda row: embedding_exists(row, args.embedding_root), axis=1)]
    logger.info(f"Rows after checking embedding existence: {len(metadata)}")
    if metadata.empty: logger.error("No valid data after filtering. Exiting."); return

    unique_pids_initial = sorted(metadata['PatientID'].unique())
    if args.max_patients is not None and args.max_patients < len(unique_pids_initial):
        subset_pids = unique_pids_initial[:args.max_patients]
        metadata = metadata[metadata['PatientID'].isin(subset_pids)]
        logger.info(f"Using only {args.max_patients} patients. Rows: {len(metadata)}")

    embedding_cache_dict = {}
    logger.info("Loading embeddings...")
    metadata['embedding'] = metadata.apply(lambda r: load_embedding(os.path.join(args.embedding_root, r['embedding_file']), embedding_cache_dict), axis=1)
    logger.info("Embeddings loaded.")

    logger.info("Preprocessing TTE data for survival modeling component.")
    metadata = time_to_event_preprocessing(metadata, log_transform_tte=args.log_tte)

    unique_pids_processed = sorted(metadata['PatientID'].unique())
    if not unique_pids_processed: logger.error("No patients left. Exiting."); return
    
    train_pids, temp_pids = train_test_split(unique_pids_processed, test_size=0.3, random_state=args.random_seed)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=args.random_seed)

    train_meta = metadata[metadata['PatientID'].isin(train_pids)].copy()
    val_meta = metadata[metadata['PatientID'].isin(val_pids)].copy() 
    test_meta = metadata[metadata['PatientID'].isin(test_pids)].copy()
    logger.info(f"Data split: Train PIDs={len(train_pids)}, Val PIDs={len(val_pids)}, Test PIDs={len(test_pids)}")

    logger.info(f"Building sequences and preparing data for sklearn models...")
    train_seq_records = build_sequences_1year_future_label(train_meta, args.window_size, args.prediction_horizon_days, for_multitask=True)
    val_seq_records = build_sequences_1year_future_label(val_meta, args.window_size, args.prediction_horizon_days, for_multitask=True)
    test_seq_records = build_sequences_1year_future_label(test_meta, args.window_size, args.prediction_horizon_days, for_multitask=True)

    if not all([train_seq_records, val_seq_records, test_seq_records]): 
        logger.error("One or more sequence sets are empty. Exiting."); return

    (X_train, y_train_cls, _, _, 
     y_train_time, y_train_event, train_survival_mask) = prepare_sklearn_data(train_seq_records, args.window_size, args.embed_dim, for_survival=True)
    (X_val, y_val_cls, _, _, 
     y_val_time, y_val_event, val_survival_mask) = prepare_sklearn_data(val_seq_records, args.window_size, args.embed_dim, for_survival=True)
    (X_test, y_test_cls, pids_test, local_indices_test, 
     y_test_time, y_test_event, test_survival_mask) = prepare_sklearn_data(test_seq_records, args.window_size, args.embed_dim, for_survival=True)

    logger.info(f"Sklearn data shapes: X_train: {X_train.shape}, y_train_cls: {y_train_cls.shape}")
    logger.info(f"Train survival data: {np.sum(train_survival_mask)} valid samples.")
    logger.info(f"Val survival data: {np.sum(val_survival_mask)} valid samples (not used for fitting).")
    logger.info(f"Test survival data: {np.sum(test_survival_mask)} valid samples.")

    all_results = []
    trained_classification_models = {} 

    # --- XGBoost Classifier ---
    xgb_clf = xgb.XGBClassifier(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=args.random_seed
    )
    results_xgb_cls, trained_xgb_cls = train_evaluate_classifier(
        xgb_clf, f"XGBoost_{args.prediction_horizon_days}DayFuture_Classifier",
        X_train, y_train_cls, X_val, y_val_cls, X_test, y_test_cls, 
        pids_test, local_indices_test, args
    )
    all_results.append(results_xgb_cls)
    trained_classification_models["XGBoost_Classifier"] = trained_xgb_cls

    # Removed LightGBM Classifier training section
    # Removed KNN Classifier training section

    # --- Survival Models ---
    X_train_s, y_train_time_s, y_train_event_s = X_train[train_survival_mask], y_train_time[train_survival_mask], y_train_event[train_survival_mask]
    X_val_s, y_val_time_s, y_val_event_s = X_val[val_survival_mask], y_val_time[val_survival_mask], y_val_event[val_survival_mask] 
    X_test_s, y_test_time_s, y_test_event_s = X_test[test_survival_mask], y_test_time[test_survival_mask], y_test_event[test_survival_mask]
    pids_test_s = [p for i, p in enumerate(pids_test) if test_survival_mask[i]]
    local_indices_test_s = [idx for i, idx in enumerate(local_indices_test) if test_survival_mask[i]]

    #  true classification labels for the test set (before the survival mask is applied)
    y_test_cls_s = y_test_cls[test_survival_mask]

    # --- XGBoost Survival Model ---
    if X_train_s.shape[0] > 0 and X_test_s.shape[0] > 0: 
        xgb_surv = xgb.XGBModel( 
            objective='survival:cox',
            # eval_metric='cox-nloglik', # leave out 
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            random_state=args.random_seed,
        )
        results_xgb_surv = train_evaluate_xgboost_survival(
            xgb_surv, f"XGBoost_TTE_Survival",
            X_train_s, y_train_time_s, y_train_event_s,
            X_val_s if X_val_s.shape[0] > 0 else None, 
            y_val_time_s if X_val_s.shape[0] > 0 else None, 
            y_val_event_s if X_val_s.shape[0] > 0 else None,
            X_test_s, y_test_time_s, y_test_event_s,
            pids_test_s, local_indices_test_s,
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
        sw_out_path = os.path.join(f"./{args.prediction_horizon_days}day_future_prediction_outputs", f"xgboost_only_{args.prediction_horizon_days}day_future_switch_analysis.csv") # Updated filename
        combined_sw_df.to_csv(sw_out_path, index=False)
        logger.info(f"Combined {args.prediction_horizon_days}-day future switch analysis saved to: {sw_out_path}")

    logger.info("Script finished.")

if __name__ == "__main__":
    main()