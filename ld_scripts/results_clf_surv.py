# %% [markdown]
# # Survival TTE Based Evaluations

# %%
import os

# path to embeddings from workingdir
embedding_path =  "./../../../commonfilesharePHI/slee/ckd-optum"
# os.listdir(embedding_path)
embedding_size = "/ckd_embeddings_full" # 10:  10 - 15 minutes, full: 2 hours
folder_path = embedding_path + embedding_size
"""  """
# List all items in the folder and filter only subdirectories
subfolders = [entry for entry in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, entry))]

# Count the number of subfolders
num_subfolders = len(subfolders)

print(f"Number of subfolders in '{folder_path}': {num_subfolders}")


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
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
# subset event level
# fp = f"./365day_future_prediction_outputs_50_subset_{n}_stage_filter_v{version}"
# full event level
fp = f"./365day_future_prediction_outputs_50_full_stage_filter_v{version}"

# subset patient level
# fp = "./365day_future_prediction_outputs_50_subset_1000_stage_filter_patient_level_v2"
# full patient level 
# fp = "./365day_future_prediction_outputs_50_full_stage_filter_patient_level_v2"
print(fp)

# %%
dirs = [
    "/LSTM_365DayFutureTarget_detailed_outputs.csv",
    "/MLP_365DayFutureTarget_detailed_outputs.csv",
    "/RNN_365DayFutureTarget_detailed_outputs.csv",
    "/TCN_365DayFutureTarget_detailed_outputs.csv",
    "/Transformer_365DayFutureTarget_detailed_outputs.csv",
]

# dirs = [
#     "/DeepSurv_LSTM_365DayFutureTarget_detailed_outputs.csv",
#     "/DeepSurv_MLP_365DayFutureTarget_detailed_outputs.csv",
#     "/DeepSurv_RNN_365DayFutureTarget_detailed_outputs.csv",
#     "/DeepSurv_TCN_365DayFutureTarget_detailed_outputs.csv",
#     "/DeepSurv_Transformer_365DayFutureTarget_detailed_outputs.csv",
# ]

filepaths = [fp + i for i in dirs]
filepaths

# %%
pd.read_csv(filepaths[0]).head()

# %%
import logging
import sys

log_file = open(f'results_files/{fp}_results_clf_surv_output.log', 'w') 

# Redirect stdout to the log file
sys.stdout = log_file
# print("check")

metrics = evaluate_models( filepaths, threshold_cap = 0.0239, n_workers=20, verbose = True)

sys.stdout = sys.__stdout__ # Restore original stdout
log_file.close()

# %%
metrics.columns
out_path = os.path.join("results_files", fp + "metrics.csv")
metrics.to_csv(fp + "metrics.csv")

# %%
print("performance")
metrics[['Model', 'Threshold', 'AUROC', 'AUPRC', 'F1', 'PPV', 'Recall']]

# %%
metrics.columns

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_comparison(df, metrics=['AUROC', 'AUPRC', 'F1', 'PPV']):
    models = df['Model'].tolist()
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Set bar width and positions
    width = 0.8 / n_models
    x = np.arange(n_metrics)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, model_name in enumerate(models):
        model_row = df[df['Model'] == model_name]
        
        # Extract means and calculate relative error lengths
        means = [model_row[f'{m}_boot_mean'].values[0] for m in metrics]
        lower = [model_row[f'{m}_ci_lower'].values[0] for m in metrics]
        upper = [model_row[f'{m}_ci_upper'].values[0] for m in metrics]
        
        # Matplotlib error bars expect (mean - lower, upper - mean)
        yerr = [
            [m - l for m, l in zip(means, lower)],
            [u - m for m, l, u in zip(means, lower, upper)]
        ]
        
        # Position each model's bars
        pos = x - 0.4 + (i * width) + (width / 2)
        ax.bar(pos, means, width, label=model_name, yerr=yerr, capsize=4, alpha=0.8)

    # Styling
    ax.set_ylabel('Score')
    ax.set_title('Model Performance with 95% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
plot_metrics_comparison(metrics)

# %%
def plot_single_metric_leaderboard(df, metric='AUROC'):
    # Sort data by mean performance
    df_sorted = df.sort_values(f'{metric}_boot_mean', ascending=False)
    
    models = df_sorted['Model'].tolist()
    means = df_sorted[f'{metric}_boot_mean'].values
    lower = df_sorted[f'{metric}_ci_lower'].values
    upper = df_sorted[f'{metric}_ci_upper'].values
    
    # Calculate error lengths
    yerr = [means - lower, upper - means]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, means, yerr=yerr, capsize=8, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.title(f'Comparative Analysis: {metric}', fontsize=14, fontweight='bold')
    plt.ylabel(f'{metric} (95% CI)')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# # %%
# for i in
#     plot_single_metric_leaderboard(metrics)
