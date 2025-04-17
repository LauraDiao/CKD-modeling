import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score
)
from scipy.special import softmax
from sklearn.utils import resample

# Global config
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_and_process_file(filepath):
    df = pd.read_csv(filepath)
    logits = df[["logit_0", "logit_1"]].values
    probs = softmax(logits, axis=1)[:, 1]
    labels = df["event"].astype(int).values
    return probs, labels

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

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

def bootstrap_metrics(y_true, y_probs, threshold, n_iterations=1000):
    boot_metrics = {k: [] for k in ["AUROC", "AUPRC", "F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]}
    for _ in range(n_iterations):
        idx = resample(np.arange(len(y_true)))
        yb_true, yb_probs = y_true[idx], y_probs[idx]
        boot_metrics["AUROC"].append(roc_auc_score(yb_true, yb_probs))
        boot_metrics["AUPRC"].append(average_precision_score(yb_true, yb_probs))
        preds = (yb_probs >= threshold).astype(int)
        boot_metrics["F1"].append(f1_score(yb_true, preds))
        boot_metrics["PPV"].append(precision_score(yb_true, preds))
        boot_metrics["Recall"].append(recall_score(yb_true, preds))
        boot_metrics["Avg Precision"].append(average_precision_score(yb_true, yb_probs))
        boot_metrics["Avg Recall"].append(recall_score(yb_true, preds))
    return {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot_metrics.items()}

def plot_roc_pr_curves(results):
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_probs'])
        precision, recall, _ = precision_recall_curve(res['y_true'], res['y_probs'])

        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUROC={res['metrics']['AUROC']:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(recall, precision, label=f"{name} (AUPRC={res['metrics']['AUPRC']:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {name}")
        plt.legend()
        plt.show()

def plot_metric_bars(all_boot_metrics):
    metrics = ["F1", "PPV", "Recall", "Avg Precision", "Avg Recall"]
    for metric in metrics:
        names, means, lowers, uppers = [], [], [], []
        for name, bm in all_boot_metrics.items():
            mean, low, high = bm[metric]
            names.append(name)
            means.append(mean)
            lowers.append(mean - low)
            uppers.append(high - mean)
        plt.figure()
        plt.bar(names, means, yerr=[lowers, uppers], capsize=5)
        plt.ylabel(metric)
        plt.title(f"{metric} with 95% CI")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def evaluate_models(filepaths):
    results = {}
    all_boot_metrics = {}
    for filepath in filepaths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        y_probs, y_true = load_and_process_file(filepath)
        threshold = find_optimal_threshold(y_true, y_probs)
        metrics = compute_metrics(y_true, y_probs, threshold)
        boot_metrics = bootstrap_metrics(y_true, y_probs, threshold)
        results[name] = {
            "y_true": y_true,
            "y_probs": y_probs,
            "threshold": threshold,
            "metrics": metrics
        }
        all_boot_metrics[name] = boot_metrics
    plot_roc_pr_curves(results)
    plot_metric_bars(all_boot_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model outputs from CSVs with logits.")
    parser.add_argument("files", nargs="+", help="Paths to one or more CSV files.")
    args = parser.parse_args()
    evaluate_models(args.files)
