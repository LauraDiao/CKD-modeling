# %% [markdown]
# # Survival TTE Based Evaluations

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

dirs = [
    "/DeepSurv_LSTM_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_MLP_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_RNN_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_TCN_365DayFutureTarget_detailed_outputs.csv",
    "/DeepSurv_Transformer_365DayFutureTarget_detailed_outputs.csv",
]

# %%
filepaths = [fp + i for i in dirs]
print(filepaths)

# %%
import logging
import sys
import os

modifier = "deepsurv_tte" 
log_path = f'results_logs/{fp}_results_{modifier}_metrics_output.log'

out_folder = "results_files_v2"
out_path =  os.path.join(out_folder, fp.strip("./"))
print(out_path)
try:
    os.mkdir(out_path)
except FileExistsError:
    pass


# %%
metric_path_1 = os.path.join(out_path, f"{modifier}_metrics_pt1.csv")
print(metric_path_1)
metric_path_2 = os.path.join(out_path, f"{modifier}_metrics_pt2.csv")
print(metric_path_2)
metric_path_3 = os.path.join(out_path, f"{modifier}_metrics_pt3.csv")
print(metric_path_3)

# %%
# def analyze_dataframes(list_of_dfs):
#     all_results = []
#     for i, filepath in enumerate(list_of_dfs):
#         df = pd.read_csv(filepath)
#         df_columns = list(df.columns)
#         # print(df_columns)
#         df_columns.remove('PatientID')
#         df['EncounterDate'] = pd.to_datetime(df['EncounterDate'])
#         try:
#             df_columns.remove('LocalIndex')
#         except:
#             print()
#         fn = os.path.basename(filepath)
#         modeln = os.path.splitext(fn)[0]
#         for column in df_columns:
#             column_results = {
#                 'DataFrame_Index': modeln,
#                 'Column_Name': column,
#                 'Max': None,
#                 'Min': None,
#                 'Mode': None,
#                 'Median': None
#             }
#             column_results['Max'] = df[column].max()
#             column_results['Min'] = df[column].min()
#             mode_values = df[column].mode().tolist()
#             column_results['Mode'] = ', '.join(map(str, mode_values)) if mode_values else None
#             column_results['Median'] = df[column].median()
#             all_results.append(column_results)
#     return pd.DataFrame(all_results)
# df_stats = analyze_dataframes(filepaths)
# df_stats.head()

# %%
import pandas as pd
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
import warnings
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
# Suppress all warnings
warnings.filterwarnings("ignore")

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df[["tte_cox_true_time", "tte_cox_true_event", "tte_cox_risk_score"]].dropna()

def create_surv_object(df: pd.DataFrame):
    times = df["tte_cox_true_time"].values
    events = df["tte_cox_true_event"].values.astype(bool)
    surv = Surv.from_arrays(events, times)
    risks = df["tte_cox_risk_score"].values
    return surv, risks

def compute_confusion_matrix(y_true, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    confusion_m = {
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp
    }
    return confusion_m


def evaluate_day_365(surv, risks: np.ndarray, threshold: float = 0.2) -> dict:
    eval_day = [365]
    c_index = concordance_index_ipcw(surv, surv, risks)[0]
    _, auc_vals = cumulative_dynamic_auc(surv, surv, risks, eval_day)
    auc_365 = float(auc_vals[0]) if isinstance(auc_vals, (list, np.ndarray)) else float(auc_vals)

    true_events = [int(pair[0] & (pair[1] <= 365)) for pair in surv]#.astype(int)
    # true_events = np.array(true_events, dtype='int')
    # print(true_events)
    predicted_events = (risks > threshold).astype(int)
    # print(predicted_events)

    # confusion matrix
    matrix = compute_confusion_matrix(true_events, predicted_events)
    

    return {"AUC@365": auc_365, "C-index": float(c_index), "Confusion Matrix": matrix}

# %%
# eval pt1
def evaluate_models_pt1(filepaths, verbose=False):
    df_list = []
    print("\n--- Time-dependent AUC and C-index at Day 365 ---")
    for filepath in tqdm(filepaths, desc="Evaluating models at Day 365"):
        name = os.path.splitext(os.path.basename(filepath))[0]
        fn = os.path.basename(filepath)
        df = load_and_clean_data(filepath)
        surv, risks = create_surv_object(df)

        metrics = evaluate_day_365(surv, risks)
        
        row = {"Model": name}
        for k, v in metrics.items():
            
            if isinstance(v, dict):
                for cm_key, cm_val in v.items():
                    # Converts 'True Negative' to 'TN'
                    row[cm_key] = cm_val
            else:
                row[k] = v

        df_list.append(row)

        if verbose:
            print(f"\nModel: {os.path.splitext(fn)[0]}")
            print(f"  AUC@365: {metrics['AUC@365']:.4f}")
            print(f"  C-index: {metrics['C-index']:.4f}")
            print(f"  Confusion Matrix: {metrics['Confusion Matrix']}")
    
    metrics_df = pd.DataFrame(df_list)
    return metrics_df


# %%
# eval pt2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def evaluate_models_pt2(filepaths, verbose=False):
    model_names = [fp.split("/")[-1].replace("_365DayFuture_detailed_outputs.csv", "") for fp in filepaths]
    eval_times = np.arange(30, 366, 30)

    combined_brier_scores = []

    for fp, model_name in zip(filepaths, model_names):
        df = pd.read_csv(fp)
        df_clean = df.dropna(subset=['tte_cox_true_time', 'tte_cox_true_event', 'cl_prob_1'])

        event_times = df_clean['tte_cox_true_time'].values
        event_observed = df_clean['tte_cox_true_event'].values
        predicted_probs = df_clean['cl_prob_1'].values

        kmf_censor = KaplanMeierFitter()
        kmf_censor.fit(event_times, event_observed == 0)

        for t in eval_times:
            y_true = (event_times > t).astype(int)
            y_pred = 1 - predicted_probs  # survival prob

            G_t = kmf_censor.predict(t)
            weights = (event_times >= t).astype(float) / np.clip(G_t, 1e-5, None)

            brier_score_t = np.mean(weights * (y_pred - y_true) ** 2)
            combined_brier_scores.append({
                'model': model_name,
                'time': t,
                'brier_score': brier_score_t                
            })

    # Convert to DataFrame
    brier_df = pd.DataFrame(combined_brier_scores)

    
    # Plot
    plt.figure(figsize=(10, 6))
    for model_name in brier_df['model'].unique():
        df_plot = brier_df[brier_df['model'] == model_name]
        plt.plot(df_plot['time'], df_plot['brier_score'], marker='o', label=model_name)
    
    title = "Temporal Brier Scores by Model"
    plt.title(title)
    plt.xlabel("Time (days)")
    plt.ylabel("Brier Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    figname = f"{modifier}_{title}.png"
    figpath = os.path.join(out_path, figname)
    plt.savefig(figpath)
    plt.show()

    return brier_df


# %%
# eval pt3
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, brier_score



def sample_shared_test_set(df, horizon, N_PROGRESSOR_SAMPLES, RANDOM_SEED):
    """
    For each patient, sample up to N_PROGRESSOR_SAMPLES distinct failure times
    if they progressed, or one time >= horizon if they did not.
    """
    samples = []
    for pid, group in df.groupby("PatientID"):
        if group['cl_true_label'].iloc[0] == 1:
            # only keep rows where failure actually occurred
            fails = group[group['tte_cox_true_event'] == 1]
            for i in range(min(N_PROGRESSOR_SAMPLES, len(fails))):
                row = fails.sample(n=1, random_state=RANDOM_SEED + i).iloc[0].copy()
                samples.append(row)
        else:
            # sample a censored time that survives at least to the horizon
            cens = group[(group['tte_cox_true_event'] == 0) &
                         (group['tte_cox_true_time'] >= horizon)]
            if cens.empty:
                # if none survive past horizon, skip patient
                continue
            row = cens.sample(n=1, random_state=RANDOM_SEED).iloc[0].copy()
            samples.append(row)
    return pd.DataFrame(samples).drop_duplicates(subset=["PatientID", "tte_cox_true_time"])


def evaluate_model_fullsample(df, eval_day, bootstrap=False, random_state=None):
    """
    Compute C-index, time-dependent AUC@eval_day, and IPCW Brier@eval_day
    using the entire test sample without further dropping.
    """
    if bootstrap:
        df = df.sample(n=len(df), replace=True, random_state=random_state)

    df = df.dropna(subset=['tte_cox_true_time', 'tte_cox_true_event', 'tte_cox_risk_score'])
    T = df['tte_cox_true_time'].values
    E = df['tte_cox_true_event'].astype(bool).values
    R = df['tte_cox_risk_score'].values

    # surv = Surv.from_arrays(E, T)
    surv, risks = create_surv_object(df)

    # Concordance
    try:
        # c_index = concordance_index(T, -R, event_observed=E)
        # check
        c_index = concordance_index_ipcw(surv, surv, R)[0]
    except Exception:
        c_index = np.nan

    # Time-dependent AUC
    try:
        _, auc_vals = cumulative_dynamic_auc(surv, surv, risks, eval_day)
        auc_365 = float(auc_vals[0]) if isinstance(auc_vals, (list, np.ndarray)) else float(auc_vals)
    except Exception:
        auc_365 = np.nan

    # IPCW-weighted Brier score
    try:
        # Check if cl_prob_1 exists for Brier score calculation
        if 'cl_prob_1' in df.columns:
            df_brier = df.dropna(subset=['cl_prob_1'])
            event_times = df_brier['tte_cox_true_time'].values
            event_observed = df_brier['tte_cox_true_event'].values
            predicted_probs = df_brier['cl_prob_1'].values
            
            # Fit censoring distribution
            kmf_censor = KaplanMeierFitter()
            kmf_censor.fit(event_times, event_observed == 0)
            
            # Calculate Brier score at eval_day
            y_true = (event_times > eval_day).astype(int)
            y_pred = 1 - predicted_probs  # survival probability
            
            G_t = kmf_censor.predict(eval_day)
            weights = (event_times >= eval_day).astype(float) / np.clip(G_t, 1e-5, None)
            
            brier_365 = np.mean(weights * (y_pred - y_true) ** 2)
    except Exception:
        brier_365 = np.nan

    # return c_index, auc_365, brier_365
    return {"AUC@365": auc_365, "C-index": float(c_index), "brier_score": brier_365}
    
def evaluate_model_fullsample_v0(df, eval_day):
    """
    Compute C-index, time-dependent AUC@eval_day, and IPCW Brier@eval_day
    using the entire test sample without further dropping.
    """
    df = df.dropna(subset=['tte_cox_true_time', 'tte_cox_true_event', 'tte_cox_risk_score'])
    T = df['tte_cox_true_time'].values
    E = df['tte_cox_true_event'].astype(bool).values
    R = df['tte_cox_risk_score'].values

    surv = Surv.from_arrays(E, T)

    # Concordance
    try:
        c_index = concordance_index(T, -R, event_observed=E)
    except Exception:
        c_index = np.nan

    # Time-dependent AUC
    try:
        _, auc_vals = cumulative_dynamic_auc(
            surv_train=surv,
            surv_test=surv,
            risk_scores_train=-R,
            risk_scores_test=-R,
            times=np.array([eval_day])
        )
        auc_365 = float(auc_vals[0])
    except Exception:
        auc_365 = np.nan

    # IPCW-weighted Brier score
    try:
        _, brier_vals = brier_score(
            surv_train=surv,
            surv_test=surv,
            pred_scores=-R,
            times=np.array([eval_day])
        )
        brier_365 = float(brier_vals[0])
    except Exception:
        brier_365 = np.nan

    # return c_index, auc_365, brier_365
    return {"AUC@365": auc_365, "C-index": float(c_index), "brier_score": brier_365}

def plot_risk_distribution(df, model_name):
    plt.hist(df['tte_cox_risk_score'], bins=50, edgecolor='black')
    
    plt.xlabel("Risk Score")
    plt.ylabel("Count")

    title = f"Risk Score Distribution: {model_name}"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    # figname = f"{modifier}_{title}.png"
    # figpath = os.path.join(out_path, figname)
    # plt.savefig(figpath)
    plt.show()

def plot_risk_vs_tte(df, model_name):
    plt.scatter(
        df['tte_cox_risk_score'],
        df['tte_cox_true_time'],
        c=df['tte_cox_true_event'],
        cmap='coolwarm',
        alpha=0.6
    )
    
    plt.xlabel("Predicted Risk Score")
    plt.ylabel("Time to Event")
    plt.colorbar(label="Event (1) vs Censored (0)")

    title = f"Risk vs Time-to-Event: {model_name}"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    # figname = f"{modifier}_{title}.png"
    # figpath = os.path.join(out_path, figname)
    # plt.savefig(figpath)
    plt.show()

from concurrent.futures import ProcessPoolExecutor

def bootstrap_once(seed, df, eval_day):
    """
    Single bootstrap iteration: resample with replacement and compute metrics.
    """
    np.random.seed(seed)
    boot_sample = df.sample(n=len(df), replace=True, random_state=seed)
    
    try:
        metrics = evaluate_model_fullsample(boot_sample, eval_day)
        return metrics
    except Exception as e:
        # Return NaN if bootstrap sample fails (e.g., no events)
        return {"AUC@365": np.nan, "C-index": np.nan, "brier_score": np.nan}


def bootstrap_metrics(df, eval_day, n_iterations=1000, n_workers=8):

    print(f"Bootstrapping with {n_iterations} iterations using {n_workers} workers...")
    
    # Generate random seeds for reproducibility
    seeds = np.random.randint(0, 100000, size=n_iterations)
    
    # Initialize storage for bootstrap results
    boot_metrics = {
        'C-index': [],
        f'AUC@{eval_day}': [],
        f'Brier@{eval_day}': []
    }
    
    # Parallel bootstrap
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(bootstrap_once, s, df, eval_day) for s in seeds]
        
        for i, f in enumerate(futures):
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_iterations} iterations...")
            
            result = f.result()
            boot_metrics['C-index'].append(result['C-index'])
            boot_metrics[f'AUC@{eval_day}'].append(result[f'AUC@{eval_day}'])
            boot_metrics[f'Brier@{eval_day}'].append(result['brier_score'])
    
    print("Bootstrapping completed.")
    
    # Compute statistics (remove NaN values before computing percentiles)
    # summary = {}
    # for metric_name, values in boot_metrics.items():
    #     clean_values = [v for v in values if not np.isnan(v)]
    #     if len(clean_values) > 0:
    #         summary[metric_name] = {
    #             'mean': np.mean(clean_values),
    #             'ci_lower': np.percentile(clean_values, 2.5),
    #             'ci_upper': np.percentile(clean_values, 97.5),
    #             'std': np.std(clean_values),
    #             'n_valid': len(clean_values)
    #         }
    #     else:
    #         summary[metric_name] = {
    #             'mean': np.nan,
    #             'ci_lower': np.nan,
    #             'ci_upper': np.nan,
    #             'std': np.nan,
    #             'n_valid': 0
    #         }

    summary = {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)) 
    for k, v in boot_metrics.items()}
    
    return summary

def evaluate_models_pt3(filepaths, EVAL_DAY=365, n_boot=1000, n_workers=8, 
                        N_PROGRESSOR_SAMPLES=5, RANDOM_SEED=42, verbose=False):
    
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Creates a shared test set so all models are evaluated on the same patient-timepoint pairs
    base_df = pd.read_csv(filepaths[0])
    test_df = sample_shared_test_set(base_df, EVAL_DAY, N_PROGRESSOR_SAMPLES, RANDOM_SEED)

    p_count = (test_df['tte_cox_true_event'] == 1).sum()
    tot = len(test_df)
    prev = 100 * p_count / tot if tot > 0 else 0
    print(f"Shared-test prevalence: {prev:.1f}% ({p_count}/{tot})\n")

    results = {}
    all_boot_metrics = {}
    df_list = []
    
    for filepath in filepaths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\nProcessing file: {name}")

        df_full = pd.read_csv(filepath)
        
        # print()
        # print("Full Data shape:", df_full.shape)
        # # *** MERGE WITH SHARED TEST SET ***
        # # This ensures we use the same timepoints across all models
        # merged = (
        #     test_df[['PatientID', 'tte_cox_true_time', 'tte_cox_true_event']]
        #     .merge(
        #         df_full[['PatientID', 'tte_cox_true_time', 'tte_cox_risk_score']],
        #         on=['PatientID', 'tte_cox_true_time'],
        #         how='inner'
        #     )
        # )
        # print("Data shape:", merged.shape)
        # print(merged.head())
        
        print(f"Computing metrics at evaluation day {EVAL_DAY}...")
        metrics = evaluate_model_fullsample(df_full, EVAL_DAY)

        
        print("Metrics: ")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        boot_metrics = bootstrap_metrics(df_full, EVAL_DAY, n_iterations=n_boot, n_workers=n_workers)
        
        if verbose:
            print("Bootstrapped Metrics with 95% Confidence Intervals:")
            for k, (mean, low, high) in boot_metrics.items():
                print(f"  {k}: {mean:.4f} [{low:.4f}, {high:.4f}]")
        
        # Build output row
        row = {"Model": name, "Eval_Day": EVAL_DAY}
        
        for k, v in metrics.items():
            if isinstance(v, dict):
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
            "merged_data": df_full,
            "eval_day": EVAL_DAY,
            "metrics": metrics
        }
        all_boot_metrics[name] = boot_metrics
        
        # plot_risk_distribution(df_full, name)
        # plot_risk_vs_tte(merged, name)
    
    metrics_df = pd.DataFrame(df_list)
    return metrics_df#, results, all_boot_metrics



# def evaluate_models_pt3_v0(filepaths, EVAL_DAY = 365, N_BOOTSTRAP_RUNS = 100, N_PROGRESSOR_SAMPLES = 5, RANDOM_SEED = 42, verbose=False):
#     # --- Configuration ---
#     # EVAL_DAY = 365
#     # N_BOOTSTRAP_RUNS = 100
#     # # change
#     # # +- 5-10
#     # # if patient has positive label, find window, if none, extract random 20 window sequence of events
#     # # what is the goal of the project ,of predicting the exact time of progression, or potential progression
#     # N_PROGRESSOR_SAMPLES = 5
#     # RANDOM_SEED = 42

#     np.random.seed(RANDOM_SEED)
#     random.seed(RANDOM_SEED)
#     # --- Build a shared test set from the first model's outputs ---
#     base_df = pd.read_csv(filepaths[0])
#     test_df = sample_shared_test_set(base_df, EVAL_DAY)

#     # Report prevalence in the shared sample
#     p_count = (test_df['tte_cox_true_event'] == 1).sum()
#     tot = len(test_df)
#     prev = 100 * p_count / tot if tot > 0 else 0
#     print(f"Shared-test prevalence: {prev:.1f}% ({p_count}/{tot})")

#     df_list = []
#     # --- Evaluate each model on that same sample ---
#     for path in filepaths:
#         model_name = os.path.splitext(os.path.basename(path))[0]
#         df_full = pd.read_csv(path)
#         merged = (
#             test_df[['PatientID', 'tte_cox_true_time', 'tte_cox_true_event']]
#             .merge(
#                 df_full[['PatientID', 'tte_cox_true_time', 'tte_cox_risk_score']],
#                 on=['PatientID', 'tte_cox_true_time'],
#                 how='inner'
#             )
#         )

#         # c_idx, auc365, bri365 = evaluate_model_fullsample(merged, EVAL_DAY)
#         # metrics = evaluate_model_fullsample(merged, EVAL_DAY)

#         bootstrap_results = []
#         for boot_run in range(N_BOOTSTRAP_RUNS):
#             boot_metrics = evaluate_model_fullsample(
#                 merged, 
#                 EVAL_DAY, 
#                 bootstrap=True, 
#                 random_state=RANDOM_SEED + boot_run
#             )
#             bootstrap_results.append(boot_metrics)

#         # Compute mean and CI from bootstrap samples
#         boot_df = pd.DataFrame(bootstrap_results)
#         metrics = {}
#         for metric_name in boot_df.columns:
#             values = boot_df[metric_name].dropna()
#             metrics[f"{metric_name}_boot_mean"] = values.mean()
#             metrics[f"{metric_name}_ci_lower"] = values.quantile(0.025)
#             metrics[f"{metric_name}_ci_upper"] = values.quantile(0.975)
#         #  - bootstrap metrics?

#         row = {"Model": model_name}
#         for k, v in metrics.items():
            
#             if isinstance(v, dict):
#                 for cm_key, cm_val in v.items():
#                     # Converts 'True Negative' to 'TN'
#                     row[cm_key] = cm_val
#             else:
#                 row[k] = v

#         df_list.append(row)
#         if verbose: 
#             print(f"\n--- {model_name} ---")
#             print(f"C-index            : {c_idx:.3f}")
#             print(f"AUC@{EVAL_DAY}      : {auc365:.3f}")
#             print(f"Brier@{EVAL_DAY}    : {bri365:.3f}")

#         plot_risk_distribution(merged, model_name)
#         plot_risk_vs_tte(merged, model_name)
    
#     metrics_df = pd.DataFrame(df_list)
#     return metrics_df

# %%
from contextlib import redirect_stdout, redirect_stderr

with open(log_path, 'w') as f:
    with redirect_stdout(f), redirect_stderr(f):
        
        metrics_1 = evaluate_models_pt1(filepaths, verbose=False)
        metrics_2 = evaluate_models_pt2(filepaths, verbose=False)
        metrics_3 = evaluate_models_pt3(filepaths, EVAL_DAY = 365, n_boot= 1000, n_workers=16
        , N_PROGRESSOR_SAMPLES= 5, RANDOM_SEED = 42, verbose=False)


# log_file.close()

# %%
def metrics_to_csv(path, metrics):
    print(path)
    try:
        # Attempt Polars syntax first
        metrics.write_csv(path)
    except AttributeError:
        # Fallback to Pandas syntax
        metrics.to_csv(path, index=False)

    return

# %%
metrics_to_csv(metric_path_1, metrics_1)

# %%
metrics_to_csv(metric_path_2, metrics_2)

# %%
metrics_to_csv(metric_path_3, metrics_3)

# %%
metrics_1

# %%
metrics_2.head()

# %%
metrics_3

# %%


# %%



