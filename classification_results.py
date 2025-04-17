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
    f1_score, precision_score, recall_score
)
from scipy.special import softmax
from sklearn.utils import resample

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_classification_file(filepath):
    df = pd.read_csv(filepath)
    logits = df[["logit_0", "logit_1"]].values
    probs = softmax(logits, axis=1)[:, 1]
    labels = df["label"].astype(int).values
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

def compute_metrics(y_true, y_probs, threshold):
    preds = (y_probs >= threshold).astype(int)
    metrics = {
        "AUROC": roc_auc_score(y_true, y_probs),
        "AUPRC": average_precision_score(y_true, y_probs),
        "F1": f1_score(y_true, preds),
        "PPV": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "Avg Precision": average_precision_score(y_true, y_probs),
        "Avg Recall": recall_score(y_true, preds)
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
    }

def bootstrap_metrics(y_true, y_probs, threshold, n_iterations=1000, n_workers=8):
    print(f"Bootstrapping with {n_iterations} iterations using {n_workers} workers...")
    boot_metrics = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]}
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
    results = {}
    all_boot_metrics = {}
    for filepath in filepaths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\nProcessing file: {name}")
        y_probs, y_true = load_classification_file(filepath)
        threshold = find_optimal_threshold(y_true, y_probs, max_threshold=threshold_cap, step=threshold_step)
        print(f"Optimal threshold selected: {threshold:.4f}")
        metrics = compute_metrics(y_true, y_probs, threshold)

        boot_metrics = bootstrap_metrics(y_true, y_probs, threshold, n_iterations=n_boot, n_workers=n_workers)
        if verbose:
            print("Bootstrapped Metrics with 95% Confidence Intervals:")
            for k, (mean, low, high) in boot_metrics.items():
                print(f"  {k}: {mean:.4f} [{low:.4f}, {high:.4f}]")
        results[name] = {
            "y_true": y_true,
            "y_probs": y_probs,
            "threshold": threshold,
            "metrics": metrics
        }
        all_boot_metrics[name] = boot_metrics
    plot_roc_pr_curves(results)
    plot_metric_bars(all_boot_metrics)
