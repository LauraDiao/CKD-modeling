# %% [markdown]
# # Survival TTE Based Evaluations

# %%
# import os

# # path to embeddings from workingdir
# embedding_path =  "./../../../commonfilesharePHI/slee/ckd-optum"
# # os.listdir(embedding_path)
# embedding_size = "/ckd_embeddings_full" # 10:  10 - 15 minutes, full: 2 hours
# folder_path = embedding_path + embedding_size
# """  """
# # List all items in the folder and filter only subdirectories
# subfolders = [entry for entry in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, entry))]

# # Count the number of subfolders
# num_subfolders = len(subfolders)

# print(f"Number of subfolders in '{folder_path}': {num_subfolders}")


# %%
import os

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix
)
from scipy.special import softmax
from sklearn.utils import resample

sns.set(style="whitegrid")

def load_and_process_file_polars(filepath):
    """
    Loads data using Polars for faster CSV parsing.
    """
    # Using scan_csv for larger-than-memory potential, 
    # but collect() brings it into memory for sklearn compatibility
    df = pl.read_csv(filepath)
    
    # Extract logits and apply softmax via NumPy
    logits = df.select(["cl_logit_0", "cl_logit_1"]).to_numpy()
    probs = softmax(logits, axis=1)[:, 1]
    
    # Cast labels to int
    labels = df.select(pl.col("cl_true_label").cast(pl.Int64)).to_series().to_numpy()
    
    return probs, labels

def find_optimal_threshold(y_true, y_probs, max_threshold=1.0, step=0.01):
    # This logic remains largely NumPy/Sklearn based as it iterates 
    # over scalar thresholds
    thresholds = np.arange(0.0, max_threshold + step, step)
    best_f1 = -1.0
    best_thresh = 0.0
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        if np.sum(preds) == 0:
            continue
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

def compute_metrics(y_true, y_probs, threshold):
    preds = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    metrics = {
        "AUROC": roc_auc_score(y_true, y_probs),
        "AUPRC": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, preds),
        "PPV": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    return metrics

def evaluate_models_polars(filepaths, n_boot=1000, n_workers=8, threshold_cap=1.0,  verbose=False):
    results = {}
    df_list = []
    all_boot_metrics = {}
    for filepath in filepaths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\nProcessing: {name}")
        
        # Polars Loading
        y_probs, y_true = load_and_process_file_polars(filepath)
        
        threshold = find_optimal_threshold(y_true, y_probs, max_threshold=threshold_cap)
        metrics = compute_metrics(y_true, y_probs, threshold)
        
        print("Metrics: ")
        for keys,values in metrics.items():
            print(keys, values)

        # boot_metrics = bootstrap_metrics(y_true, y_probs, threshold, n_iterations=n_boot, n_workers=n_workers)
        boot_metrics = bootstrap_metrics_gpu(y_true, y_probs, threshold, n_iterations=n_boot, gpu_id=3)
        
        if verbose:
            print("Bootstrapped Metrics with 95% Confidence Intervals:")
            for k, (mean, low, high) in boot_metrics.items():
                print(f"  {k}: {mean:.4f} [{low:.4f}, {high:.4f}]")
        
        # Create result row
        row = {"Model": name, "Threshold": threshold}
        for k, v in metrics.items():
            if isinstance(v, dict):  # Handle nested Confusion Matrix dictionary
                for cm_key, cm_val in v.items():
                    row[cm_key] = cm_val
            else:
                row[k] = v
        for k, (mean, low, high) in boot_metrics.items():
            row[f"{k}_boot_mean"] = mean
            row[f"{k}_ci_lower"] = low
            row[f"{k}_ci_upper"] = high

        df_list.append(row)
        
        results[name] = {
            "y_true": y_true,
            "y_probs": y_probs,
            "metrics": metrics
        }

        all_boot_metrics[name] = boot_metrics

    plot_roc_pr_curves(results)
    plot_metric_bars(all_boot_metrics)
    metrics_df = pl.DataFrame(df_list)

    return metrics_df


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import torch

from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix
)
from scipy.special import softmax
from sklearn.utils import resample

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_and_process_file(filepath):
    df = pd.read_csv(filepath)
    logits = df[["cl_logit_0", "cl_logit_1"]].values
    probs = softmax(logits, axis=1)[:, 1]
    labels = df["cl_true_label"].astype(int).values
    return probs, labels

def find_optimal_threshold(y_true, y_probs, max_threshold=1.0, step=0.01):
    thresholds = np.arange(0.0, max_threshold + step, step)
    best_f1 = -1.0
    best_thresh = 0.0
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        if np.sum(preds) == 0:
            continue
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

def compute_confusion_matrix(y_true, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    confusion_m = {
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp
    }
    return confusion_m

def compute_metrics(y_true, y_probs, threshold):
    preds = (y_probs >= threshold).astype(int)
    metrics = {
        "AUROC": roc_auc_score(y_true, y_probs),
        "AUPRC": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, preds),
        "PPV": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "Avg Precision": average_precision_score(y_true, y_probs),
        "Avg Recall": recall_score(y_true, preds),
        "Confusion Matrix": compute_confusion_matrix(y_true, preds)
    }
    return metrics

def bootstrap_metrics_gpu(y_true, y_probs, threshold, n_iterations=1000, gpu_id=3):

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Bootstrapping on {torch.cuda.get_device_name(device)} (ID: {gpu_id})")

    y_true_gpu = torch.tensor(y_true, dtype=torch.float32, device=device)
    y_probs_gpu = torch.tensor(y_probs, dtype=torch.float32, device=device)
    n_samples = len(y_true)
    
    boot_results = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]}
    
    # Process in batches of 100 iterations to maximize parallel math
    batch_size = 1024
    for b in range(0, n_iterations, batch_size):
        current_batch_size = min(batch_size, n_iterations - b)
        
        # 2. Vectorized Resampling: Generate (Batch x 1.45M) indices
        indices = torch.randint(0, n_samples, (current_batch_size, n_samples), device=device)
        
        # 3. Massively Parallel Metric Calculation
        # These operations happen across all 100 bootstrap samples simultaneously
        yb_true = y_true_gpu[indices]
        yb_probs = y_probs_gpu[indices]
        
        preds = (yb_probs >= threshold).float()
        tp = (preds * yb_true).sum(dim=1)
        fp = (preds * (1 - yb_true)).sum(dim=1)
        fn = ((1 - preds) * yb_true).sum(dim=1)
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        boot_results["PPV"].extend(precision.cpu().tolist())
        boot_results["Recall"].extend(recall.cpu().tolist())
        boot_results["Avg Recall"].append(recall.cpu().tolist())
        boot_results["F1"].extend(f1.cpu().tolist())

        # 4. Complex Metrics (AUROC/AUPRC) 
        # Sklearn is still the gold standard for these; we compute them per sample
        for i in range(current_batch_size):
            y_t_cpu = yb_true[i].cpu().numpy()
            y_p_cpu = yb_probs[i].cpu().numpy()
            auroc = roc_auc_score(y_t_cpu, y_p_cpu)
            auprc = average_precision_score(y_t_cpu, y_p_cpu)
            boot_results["AUROC"].append(auroc)
            boot_results["AUPRC"].append(auprc)
            boot_results["Avg Precision"].append(auprc)

    final_stats = {}
    for k, v in boot_results.items():

        mean = np.mean(v)
        low = np.percentile(v, 2.5)
        high = np.percentile(v, 97.5)
        
        # Store raw numbers for plotting/math
        final_stats[k] = (mean, low, high)

    return final_stats

def bootstrap_metrics_gpu_0(y_true, y_probs, threshold, n_iterations=1000, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Bootstrapping with {n_iterations} iterations on {device}...")
    
    # Move data to GPU
    y_true_gpu = torch.tensor(y_true, dtype=torch.float32, device=device)
    y_probs_gpu = torch.tensor(y_probs, dtype=torch.float32, device=device)
    n_samples = len(y_true)
    
    # Initialize containers for results
    boot_results = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]}

    for i in range(n_iterations):
        # 1. Resample indices on GPU
        indices = torch.randint(0, n_samples, (n_samples,), device=device)
        yb_true = y_true_gpu[indices]
        yb_probs = y_probs_gpu[indices]
        
        # 2. Binary predictions
        preds = (yb_probs >= threshold).float()
        
        # 3. Calculate basic metrics (F1, PPV, Recall) on GPU
        tp = (preds * yb_true).sum()
        fp = (preds * (1 - yb_true)).sum()
        fn = ((1 - preds) * yb_true).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        boot_results["PPV"].append(precision.item())
        boot_results["Recall"].append(recall.item())
        boot_results["Avg Recall"].append(recall.item())
        boot_results["F1"].append(f1.item())

        # 4. Calculate AUROC/AUPRC 
        # (Note: Standard sklearn requires CPU; we move only the small resampled batch back)
        yb_true_cpu = yb_true.cpu().numpy()
        yb_probs_cpu = yb_probs.cpu().numpy()
        
        auroc = roc_auc_score(yb_true_cpu, yb_probs_cpu)
        auprc = average_precision_score(yb_true_cpu, yb_probs_cpu)
        
        boot_results["AUROC"].append(auroc)
        boot_results["AUPRC"].append(auprc)
        boot_results["Avg Precision"].append(auprc)

    # 5. Compute mean and 95% Confidence Intervals
    final_metrics = {}
    for k, v in boot_results.items():
        mean = np.mean(v)
        low = np.percentile(v, 2.5)
        high = np.percentile(v, 97.5)
        final_metrics[k] = (mean, low, high)
        
    print("Bootstrapping completed.")
    return final_metrics

def bootstrap_once(seed, y_true, y_probs, threshold):
    np.random.seed(seed)
    idx = resample(np.arange(len(y_true)))
    yb_true, yb_probs = y_true[idx], y_probs[idx]
    preds = (yb_probs >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(yb_true, yb_probs),
        "AUPRC": average_precision_score(yb_true, yb_probs),
        "F1": f1_score(yb_true, preds),
        "PPV": precision_score(yb_true, preds),
        "Recall": recall_score(yb_true, preds),
        "Avg Precision": average_precision_score(yb_true, yb_probs),
        "Avg Recall": recall_score(yb_true, preds),
        "Confusion Matrix": confusion_matrix(y_true, preds) # different format for boostrapping
    }

def bootstrap_metrics(y_true, y_probs, threshold, n_iterations=1000, n_workers=8):
    print(f"Bootstrapping with {n_iterations} iterations using {n_workers} workers...")
    boot_metrics = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall",
                                        "Confusion Matrix"]}
    seeds = np.random.randint(0, 100000, size=n_iterations)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(bootstrap_once, s, y_true, y_probs, threshold) for s in seeds]
        for f in futures:
            result = f.result()
            for k, v in result.items():
                boot_metrics[k].append(v)

    print("Bootstrapping completed.")
    return {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot_metrics.items()}

def plot_roc_pr_curves(results):
    # Plot ROC curves on the same figure
    plt.figure()
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_probs'])
        auroc = res['metrics']['AUROC']
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()

    figname = f"{modifier}_ROC_Curves.png"
    figpath = os.path.join(out_path, figname)
    plt.savefig(figpath)
    plt.show()

    # Plot PR curves on the same figure
    plt.figure()
    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(res['y_true'], res['y_probs'])
        auprc = res['metrics']['AUPRC']
        plt.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()

    figname = f"{modifier}_Precision_Recall_Curves.png"
    figpath = os.path.join(out_path, figname)
    plt.savefig(figpath)
    plt.show()


def plot_metric_bars(all_boot_metrics):
    metrics = ["F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]
    model_names = list(all_boot_metrics.keys())

    for metric in metrics:
        means = [all_boot_metrics[m][metric][0] for m in model_names]
        lowers = [all_boot_metrics[m][metric][0] - all_boot_metrics[m][metric][1] for m in model_names]
        uppers = [all_boot_metrics[m][metric][2] - all_boot_metrics[m][metric][0] for m in model_names]

        plt.figure()
        plt.bar(model_names, means, yerr=[lowers, uppers], capsize=5)
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison with 95% CI")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        figname = f"{modifier}_{metric}.png"
        figpath = os.path.join(out_path, figname)
        plt.savefig(figpath)
        plt.show()


def evaluate_models(filepaths, n_boot=1000, n_workers=8, threshold_cap=1.0, threshold_step=0.01, verbose=False):
    # print(verbose)
    # print(n_workers)
    results = {}
    all_boot_metrics = {}
    df_list = []
    for filepath in filepaths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\nProcessing file: {name}")
        y_probs, y_true = load_and_process_file(filepath)
        threshold = find_optimal_threshold(y_true, y_probs, max_threshold=threshold_cap, step=threshold_step)
        
        print(f"Optimal threshold selected: {threshold:.3f}")
        metrics = compute_metrics(y_true, y_probs, threshold)

        print("Metrics: ")
        for keys,values in metrics.items():
            print(keys, values)

        boot_metrics = bootstrap_metrics(y_true, y_probs, threshold, n_iterations=n_boot, n_workers=n_workers)
        # boot_metrics = bootstrap_metrics_gpu(y_true, y_probs, threshold, n_iterations=n_boot, gpu_id=3)
        
        if verbose:
            print("Bootstrapped Metrics with 95% Confidence Intervals:")
            for k, (mean, low, high) in boot_metrics.items():
                print(f"  {k}: {mean:.4f} [{low:.4f}, {high:.4f}]")

        #-----
        row = {"Model": name, "Threshold": threshold}
        
        for k, v in metrics.items():
            
            if isinstance(v, dict):
                for cm_key, cm_val in v.items():
                    # Converts 'True Negative' to 'TN'
                    row[cm_key] = cm_val
            else:
                row[k] = v
            
        # Add bootstrapped statistics
        for k, (mean, low, high) in boot_metrics.items():
            row[f"{k}_boot_mean"] = mean
            row[f"{k}_ci_lower"] = low
            row[f"{k}_ci_upper"] = high
        
        df_list.append(row)
        #----
        results[name] = {
            "y_true": y_true,
            "y_probs": y_probs,
            "threshold": threshold,
            "metrics": metrics
        }
        all_boot_metrics[name] = boot_metrics

    plot_roc_pr_curves(results)
    plot_metric_bars(all_boot_metrics)
    metrics_df = pd.DataFrame(df_list)

    return metrics_df


# %%
version = 8
n = 100

# %%
# subset event level
# fp = f"./365day_future_prediction_outputs_50_subset_{n}_stage_filter_v{version}"

# full event level
fp = f"./365day_future_prediction_outputs_50_full_stage_filter_v8"

# subset patient level
# fp = "./365day_future_prediction_outputs_50_subset_1000_stage_filter_patient_level_v2"

# full patient level 
# fp = "./365day_future_prediction_outputs_50_full_stage_filter_patient_level_v2"


# dirs = [
#     "/LSTM_365DayFutureTarget_detailed_outputs.csv",
#     "/MLP_365DayFutureTarget_detailed_outputs.csv",
#     "/RNN_365DayFutureTarget_detailed_outputs.csv",
#     "/TCN_365DayFutureTarget_detailed_outputs.csv",
#     "/Transformer_365DayFutureTarget_detailed_outputs.csv",
# ]

dirs = [
    "/DeepSurv_LSTM_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_MLP_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_RNN_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_TCN_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_Transformer_365DayFutureTarget_detailed_outputs.csv",
]

# %%
# baseline event level
# fp_eskd = f"./365day_future_prediction_outputs_stage_filter_full_stage_filter_eskd_v2"
# dirs_eskd = ['/XGBoost_365DayFuture_Classifier_detailed_outputs_classification.csv']

# baseline patient level
# fp_eskd = f"./365day_future_prediction_outputs_stage_filter_full_stage_filter_eskd_v2_patient_level"
# dirs_eskd = ['/XGBoost_365DayFuture_Classifier_detailed_outputs_classification_pt_lvl.csv']

# ['/XGBoost_365DayFuture_Classifier_detailed_outputs_classification.csv',
# #  '/XGBoost_TTE_Survival_detailed_outputs_survival.csv',
# #  'xgboost_only_365day_future_switch_analysis.csv'
#  ]

# %%
# filepaths2 = [fp_eskd + i for i in dirs_eskd]
# filepaths2

# %%
filepaths1 = [fp + i for i in dirs]
filepaths1

# %%
filepaths = filepaths1 # + filepaths2
filepaths

# %%
import logging
import sys

modifier = "deepsurv" if "DeepSurv" in dirs[0] else "classification"
log_path = f'results_logs/{fp}_results_{modifier}_metrics_output.log'

# %%
modifier = "deepsurv" if "DeepSurv" in dirs[0] else "classification"
out_folder = "results_files_v2"
out_path =  os.path.join("results_files_v2", fp.strip("./"))
print(out_path)
try:
    os.mkdir(out_path)
except FileExistsError:
    pass

# %%


# %%
csv_path = os.path.join(out_path, f"{modifier}_metrics.csv")
csv_path

# %%
from contextlib import redirect_stdout, redirect_stderr

with open(log_path, 'w') as f:
    with redirect_stdout(f), redirect_stderr(f):
        metrics = evaluate_models(filepaths, threshold_cap = 0.0239, n_workers=20, verbose = True)
        # metrics = evaluate_models_polars( filepaths, threshold_cap = 0.0239, n_workers=20, verbose = True)

# log_file.close()

# %%
try:
    # Attempt Polars syntax first
    metrics.write_csv(csv_path)
except AttributeError:
    # Fallback to Pandas syntax
    metrics.to_csv(csv_path, index=False)

# %%
metrics.columns

# %%
# performance
metrics[['Model', 'AUROC', 'AUPRC', 'PPV', 'Recall', 'F1']]

# %%
# boostrap confidence intervals
metrics
# add thresdhold column

# %%
metrics.columns

# %%
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_metrics_comparison(df, metrics=['AUROC', 'AUPRC', 'F1', 'PPV']):
#     models = df['Model'].tolist()
#     n_models = len(models)
#     n_metrics = len(metrics)
    
#     # Set bar width and positions
#     width = 0.8 / n_models
#     x = np.arange(n_metrics)
    
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     for i, model_name in enumerate(models):
#         model_row = df[df['Model'] == model_name]
        
#         # Extract means and calculate relative error lengths
#         means = [model_row[f'{m}_boot_mean'].values[0] for m in metrics]
#         lower = [model_row[f'{m}_ci_lower'].values[0] for m in metrics]
#         upper = [model_row[f'{m}_ci_upper'].values[0] for m in metrics]
        
#         # Matplotlib error bars expect (mean - lower, upper - mean)
#         yerr = [
#             [m - l for m, l in zip(means, lower)],
#             [u - m for m, l, u in zip(means, lower, upper)]
#         ]
        
#         # Position each model's bars
#         pos = x - 0.4 + (i * width) + (width / 2)
#         ax.bar(pos, means, width, label=model_name, yerr=yerr, capsize=4, alpha=0.8)

#     # Styling
#     ax.set_ylabel('Score')
#     ax.set_title('Model Performance with 95% Confidence Intervals')
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics)
#     ax.set_ylim(0, 1.1)
#     ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.grid(axis='y', linestyle='--', alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# %%
# plot_metrics_comparison(metrics)

# %%
# def plot_single_metric_leaderboard(df, metric='AUROC'):
#     # Sort data by mean performance
#     df_sorted = df.sort_values(f'{metric}_boot_mean', ascending=False)
    
#     models = df_sorted['Model'].tolist()
#     means = df_sorted[f'{metric}_boot_mean'].values
#     lower = df_sorted[f'{metric}_ci_lower'].values
#     upper = df_sorted[f'{metric}_ci_upper'].values
    
#     # Calculate error lengths
#     yerr = [means - lower, upper - means]
    
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(models, means, yerr=yerr, capsize=8, color='steelblue', edgecolor='black', alpha=0.7)
    
#     # Add value labels on top of bars
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

#     plt.title(f'Comparative Analysis: {metric}', fontsize=14, fontweight='bold')
#     plt.ylabel(f'{metric} (95% CI)')
#     plt.xticks(rotation=45)
#     plt.ylim(0, 1.1)
#     plt.grid(axis='y', linestyle=':', alpha=0.6)
    
#     plt.tight_layout()
#     plt.show()

# %%
# for i in ['AUROC', 'AUPRC', 'F1', 'PPV', 'Recall']: 
#     plot_single_metric_leaderboard(metrics, metric=i)

# %% [markdown]
# # Normal Modeling

# %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import argparse
# from concurrent.futures import ProcessPoolExecutor
# from sklearn.metrics import (
#     precision_recall_curve, roc_curve,
#     average_precision_score, roc_auc_score,
#     f1_score, precision_score, recall_score
# )
# from scipy.special import softmax
# from sklearn.utils import resample

# sns.set(style="whitegrid")
# plt.rcParams["figure.figsize"] = (10, 6)

# def load_classification_file(filepath):
#     df = pd.read_csv(filepath)
#     logits = df[["cl_logit_0", "cl_logit_1"]].values
#     probs = softmax(logits, axis=1)[:, 1]
#     labels = df["label"].astype(int).values
#     return probs, labels

# def find_optimal_threshold(y_true, y_probs, max_threshold=1.0, step=0.01):
#     thresholds = np.arange(0.0, max_threshold + step, step)
#     best_f1 = -1.0
#     best_thresh = 0.0
#     for t in thresholds:
#         preds = (y_probs >= t).astype(int)
#         if np.sum(preds) == 0:
#             continue
#         f1 = f1_score(y_true, preds)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = t
#     return best_thresh

# def compute_metrics(y_true, y_probs, threshold):
#     preds = (y_probs >= threshold).astype(int)
#     metrics = {
#         "AUROC": roc_auc_score(y_true, y_probs),
#         "AUPRC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds),
#         "PPV": precision_score(y_true, preds),
#         "Recall": recall_score(y_true, preds),
#         "Avg Precision": average_precision_score(y_true, y_probs),
#         "Avg Recall": recall_score(y_true, preds)
#     }
#     return metrics

# def bootstrap_once(seed, y_true, y_probs, threshold):
#     np.random.seed(seed)
#     idx = resample(np.arange(len(y_true)))
#     yb_true, yb_probs = y_true[idx], y_probs[idx]
#     preds = (yb_probs >= threshold).astype(int)
#     return {
#         "AUROC": roc_auc_score(yb_true, yb_probs),
#         "AUPRC": average_precision_score(yb_true, yb_probs),
#         "F1": f1_score(yb_true, preds),
#         "PPV": precision_score(yb_true, preds),
#         "Recall": recall_score(yb_true, preds),
#         "Avg Precision": average_precision_score(yb_true, yb_probs),
#         "Avg Recall": recall_score(yb_true, preds),
#     }

# def bootstrap_metrics(y_true, y_probs, threshold, n_iterations=1000, n_workers=8):
#     print(f"Bootstrapping with {n_iterations} iterations using {n_workers} workers...")
#     boot_metrics = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]}
#     seeds = np.random.randint(0, 100000, size=n_iterations)

#     with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         futures = [executor.submit(bootstrap_once, s, y_true, y_probs, threshold) for s in seeds]
#         for f in futures:
#             result = f.result()
#             for k, v in result.items():
#                 boot_metrics[k].append(v)

#     print("Bootstrapping completed.")
#     return {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot_metrics.items()}

# def plot_roc_pr_curves(results):
#     plt.figure()
#     for name, res in results.items():
#         fpr, tpr, _ = roc_curve(res['y_true'], res['y_probs'])
#         auroc = res['metrics']['AUROC']
#         plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
#     plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curves")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure()
#     for name, res in results.items():
#         precision, recall, _ = precision_recall_curve(res['y_true'], res['y_probs'])
#         auprc = res['metrics']['AUPRC']
#         plt.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curves")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def plot_metric_bars(all_boot_metrics):
#     metrics = ["F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]
#     model_names = list(all_boot_metrics.keys())

#     for metric in metrics:
#         means = [all_boot_metrics[m][metric][0] for m in model_names]
#         lowers = [all_boot_metrics[m][metric][0] - all_boot_metrics[m][metric][1] for m in model_names]
#         uppers = [all_boot_metrics[m][metric][2] - all_boot_metrics[m][metric][0] for m in model_names]

#         plt.figure()
#         plt.bar(model_names, means, yerr=[lowers, uppers], capsize=5)
#         plt.ylabel(metric)
#         plt.title(f"{metric} Comparison with 95% CI")
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()

# def evaluate_models(filepaths, n_boot=1000, n_workers=8, threshold_cap=1.0, threshold_step=0.01, verbose=False):
#     results = {}
#     all_boot_metrics = {}
#     for filepath in filepaths:
#         name = os.path.splitext(os.path.basename(filepath))[0]
#         print(f"\nProcessing file: {name}")
#         y_probs, y_true = load_classification_file(filepath)
#         threshold = find_optimal_threshold(y_true, y_probs, max_threshold=threshold_cap, step=threshold_step)
#         print(f"Optimal threshold selected: {threshold:.4f}")
#         metrics = compute_metrics(y_true, y_probs, threshold)

#         boot_metrics = bootstrap_metrics(y_true, y_probs, threshold, n_iterations=n_boot, n_workers=n_workers)
#         if verbose:
#             print("Bootstrapped Metrics with 95% Confidence Intervals:")
#             for k, (mean, low, high) in boot_metrics.items():
#                 print(f"  {k}: {mean:.4f} [{low:.4f}, {high:.4f}]")
#         results[name] = {
#             "y_true": y_true,
#             "y_probs": y_probs,
#             "threshold": threshold,
#             "metrics": metrics
#         }
#         all_boot_metrics[name] = boot_metrics
#     plot_roc_pr_curves(results)
#     plot_metric_bars(all_boot_metrics)

# %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import argparse
# from concurrent.futures import ProcessPoolExecutor

# from sklearn.metrics import (
#     precision_recall_curve, roc_curve,
#     average_precision_score, roc_auc_score,
#     f1_score, precision_score, recall_score
# )
# from scipy.special import softmax
# from sklearn.utils import resample

# sns.set(style="whitegrid")
# plt.rcParams["figure.figsize"] = (10, 6)

# def load_and_process_file(filepath):
#     df = pd.read_csv(filepath)
#     logits = df[["cl_logit_0", "cl_logit_1"]].values
#     probs = softmax(logits, axis=1)[:, 1]
#     labels = df["cl_true_label"].astype(int).values
#     return probs, labels

# def find_optimal_threshold(y_true, y_probs, max_threshold=1.0, step=0.01):
#     thresholds = np.arange(0.0, max_threshold + step, step)
#     best_f1 = -1.0
#     best_thresh = 0.0
#     for t in thresholds:
#         preds = (y_probs >= t).astype(int)
#         if np.sum(preds) == 0:
#             continue
#         f1 = f1_score(y_true, preds)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = t
#     return best_thresh

# def compute_metrics(y_true, y_probs, threshold):
#     preds = (y_probs >= threshold).astype(int)
#     metrics = {
#         "AUROC": roc_auc_score(y_true, y_probs),
#         "AUPRC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds),
#         "PPV": precision_score(y_true, preds),
#         "Recall": recall_score(y_true, preds),
#         "Avg Precision": average_precision_score(y_true, y_probs),
#         "Avg Recall": recall_score(y_true, preds)
#     }
#     return metrics

# def bootstrap_once(seed, y_true, y_probs, threshold):
#     np.random.seed(seed)
#     idx = resample(np.arange(len(y_true)))
#     yb_true, yb_probs = y_true[idx], y_probs[idx]
#     preds = (yb_probs >= threshold).astype(int)
#     return {
#         "AUROC": roc_auc_score(yb_true, yb_probs),
#         "AUPRC": average_precision_score(yb_true, yb_probs),
#         "F1": f1_score(yb_true, preds),
#         "PPV": precision_score(yb_true, preds),
#         "Recall": recall_score(yb_true, preds),
#         "Avg Precision": average_precision_score(yb_true, yb_probs),
#         "Avg Recall": recall_score(yb_true, preds),
#     }

# def bootstrap_metrics(y_true, y_probs, threshold, n_iterations=1000, n_workers=8):
#     print(f"Bootstrapping with {n_iterations} iterations using {n_workers} workers...")
#     boot_metrics = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]}
#     seeds = np.random.randint(0, 100000, size=n_iterations)

#     with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         futures = [executor.submit(bootstrap_once, s, y_true, y_probs, threshold) for s in seeds]
#         for f in futures:
#             result = f.result()
#             for k, v in result.items():
#                 boot_metrics[k].append(v)

#     print("Bootstrapping completed.")
#     return {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot_metrics.items()}

# def plot_roc_pr_curves(results):
#     # Plot ROC curves on the same figure
#     plt.figure()
#     for name, res in results.items():
#         fpr, tpr, _ = roc_curve(res['y_true'], res['y_probs'])
#         auroc = res['metrics']['AUROC']
#         plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
#     plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curves")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # Plot PR curves on the same figure
#     plt.figure()
#     for name, res in results.items():
#         precision, recall, _ = precision_recall_curve(res['y_true'], res['y_probs'])
#         auprc = res['metrics']['AUPRC']
#         plt.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curves")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def plot_metric_bars(all_boot_metrics):
#     metrics = ["F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]
#     model_names = list(all_boot_metrics.keys())

#     for metric in metrics:
#         means = [all_boot_metrics[m][metric][0] for m in model_names]
#         lowers = [all_boot_metrics[m][metric][0] - all_boot_metrics[m][metric][1] for m in model_names]
#         uppers = [all_boot_metrics[m][metric][2] - all_boot_metrics[m][metric][0] for m in model_names]

#         plt.figure()
#         plt.bar(model_names, means, yerr=[lowers, uppers], capsize=5)
#         plt.ylabel(metric)
#         plt.title(f"{metric} Comparison with 95% CI")
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()


# def evaluate_models(filepaths, n_boot=1000, n_workers=8, threshold_cap=1.0, threshold_step=0.01, verbose=False):
#     results = {}
#     all_boot_metrics = {}
#     for filepath in filepaths:
#         name = os.path.splitext(os.path.basename(filepath))[0]
#         print(f"\nProcessing file: {name}")
#         y_probs, y_true = load_and_process_file(filepath)
#         threshold = find_optimal_threshold(y_true, y_probs, max_threshold=threshold_cap, step=threshold_step)
#         print(f"Optimal threshold selected: {threshold:.3f}")
#         metrics = compute_metrics(y_true, y_probs, threshold)

#         # boot_metrics = bootstrap_metrics(y_true, y_probs, threshold, n_iterations=n_boot, n_workers=n_workers)
#         # boot_metrics = bootstrap_metrics(y_true, y_probs, threshold, n_iterations=n_boot, n_workers=n_workers)
#         boot_metrics = bootstrap_metrics_gpu(y_true, y_probs, threshold, n_iterations=n_boot, gpu_id=3)
#         if verbose:
#             print("Bootstrapped Metrics with 95% Confidence Intervals:")
#             for k, (mean, low, high) in boot_metrics.items():
#                 print(f"  {k}: {mean:.4f} [{low:.4f}, {high:.4f}]")

#         results[name] = {
#             "y_true": y_true,
#             "y_probs": y_probs,
#             "threshold": threshold,
#             "metrics": metrics
#         }
#         all_boot_metrics[name] = boot_metrics

#     plot_roc_pr_curves(results)
#     plot_metric_bars(all_boot_metrics)


# %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import argparse
# from concurrent.futures import ProcessPoolExecutor
# from sklearn.metrics import (
#     precision_recall_curve, roc_curve,
#     average_precision_score, roc_auc_score,
#     f1_score, precision_score, recall_score
# )
# from scipy.special import softmax
# from sklearn.utils import resample

# sns.set(style="whitegrid")
# plt.rcParams["figure.figsize"] = (10, 6)

# def load_classification_file(filepath):
#     df = pd.read_csv(filepath)
#     logits = df[["cl_logit_0", "cl_logit_1"]].values
#     probs = softmax(logits, axis=1)[:, 1]
#     labels = df["label"].astype(int).values
#     return probs, labels

# def find_optimal_threshold(y_true, y_probs, max_threshold=1.0, step=0.01):
#     thresholds = np.arange(0.0, max_threshold + step, step)
#     best_f1 = -1.0
#     best_thresh = 0.0
#     for t in thresholds:
#         preds = (y_probs >= t).astype(int)
#         if np.sum(preds) == 0:
#             continue
#         f1 = f1_score(y_true, preds)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = t
#     return best_thresh

# def compute_metrics(y_true, y_probs, threshold):
#     preds = (y_probs >= threshold).astype(int)
#     metrics = {
#         "AUROC": roc_auc_score(y_true, y_probs),
#         "AUPRC": average_precision_score(y_true, y_probs),
#         "F1": f1_score(y_true, preds),
#         "PPV": precision_score(y_true, preds),
#         "Recall": recall_score(y_true, preds),
#         "Avg Precision": average_precision_score(y_true, y_probs),
#         "Avg Recall": recall_score(y_true, preds)
#     }
#     return metrics

# def bootstrap_once(seed, y_true, y_probs, threshold):
#     np.random.seed(seed)
#     idx = resample(np.arange(len(y_true)))
#     yb_true, yb_probs = y_true[idx], y_probs[idx]
#     preds = (yb_probs >= threshold).astype(int)
#     return {
#         "AUROC": roc_auc_score(yb_true, yb_probs),
#         "AUPRC": average_precision_score(yb_true, yb_probs),
#         "F1": f1_score(yb_true, preds),
#         "PPV": precision_score(yb_true, preds),
#         "Recall": recall_score(yb_true, preds),
#         "Avg Precision": average_precision_score(yb_true, yb_probs),
#         "Avg Recall": recall_score(yb_true, preds),
#     }

# def bootstrap_metrics(y_true, y_probs, threshold, n_iterations=1000, n_workers=8):
#     print(f"Bootstrapping with {n_iterations} iterations using {n_workers} workers...")
#     boot_metrics = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]}
#     seeds = np.random.randint(0, 100000, size=n_iterations)

#     with ProcessPoolExecutor(max_workers=n_workers) as executor:
#         futures = [executor.submit(bootstrap_once, s, y_true, y_probs, threshold) for s in seeds]
#         for f in futures:
#             result = f.result()
#             for k, v in result.items():
#                 boot_metrics[k].append(v)

#     print("Bootstrapping completed.")
#     return {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot_metrics.items()}

# def plot_roc_pr_curves(results):
#     plt.figure()
#     for name, res in results.items():
#         fpr, tpr, _ = roc_curve(res['y_true'], res['y_probs'])
#         auroc = res['metrics']['AUROC']
#         plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
#     plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curves")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure()
#     for name, res in results.items():
#         precision, recall, _ = precision_recall_curve(res['y_true'], res['y_probs'])
#         auprc = res['metrics']['AUPRC']
#         plt.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curves")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def plot_metric_bars(all_boot_metrics):
#     metrics = ["F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]
#     model_names = list(all_boot_metrics.keys())

#     for metric in metrics:
#         means = [all_boot_metrics[m][metric][0] for m in model_names]
#         lowers = [all_boot_metrics[m][metric][0] - all_boot_metrics[m][metric][1] for m in model_names]
#         uppers = [all_boot_metrics[m][metric][2] - all_boot_metrics[m][metric][0] for m in model_names]

#         plt.figure()
#         plt.bar(model_names, means, yerr=[lowers, uppers], capsize=5)
#         plt.ylabel(metric)
#         plt.title(f"{metric} Comparison with 95% CI")
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()

# def evaluate_models(filepaths, n_boot=1000, n_workers=8, threshold_cap=1.0, threshold_step=0.01, verbose=False):
#     results = {}
#     all_boot_metrics = {}
#     for filepath in filepaths:
#         name = os.path.splitext(os.path.basename(filepath))[0]
#         print(f"\nProcessing file: {name}")
#         y_probs, y_true = load_classification_file(filepath)
#         threshold = find_optimal_threshold(y_true, y_probs, max_threshold=threshold_cap, step=threshold_step)
#         print(f"Optimal threshold selected: {threshold:.4f}")
#         metrics = compute_metrics(y_true, y_probs, threshold)

#         boot_metrics = bootstrap_metrics(y_true, y_probs, threshold, n_iterations=n_boot, n_workers=n_workers)
#         if verbose:
#             print("Bootstrapped Metrics with 95% Confidence Intervals:")
#             for k, (mean, low, high) in boot_metrics.items():
#                 print(f"  {k}: {mean:.4f} [{low:.4f}, {high:.4f}]")
#         results[name] = {
#             "y_true": y_true,
#             "y_probs": y_probs,
#             "threshold": threshold,
#             "metrics": metrics
#         }
#         all_boot_metrics[name] = boot_metrics
#     plot_roc_pr_curves(results)
#     plot_metric_bars(all_boot_metrics)

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



