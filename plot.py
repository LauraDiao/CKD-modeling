import matplotlib.pyplot as plt
import pandas as pd
import re
config ="""
16:47:50 INFO: Running with the following configuration:
16:47:50 INFO: embedding_root: ./ckd_embeddings_100
16:47:50 INFO: window_size: 10
16:47:50 INFO: embed_dim: 768
16:47:50 INFO: epochs: 50
16:47:50 INFO: batch_size: 64
16:47:50 INFO: lr: 0.005
16:47:50 INFO: patience: 5
16:47:50 INFO: scheduler_patience: 2
16:47:50 INFO: metadata_file: patient_embedding_metadata.csv
16:47:50 INFO: random_seed: 42
16:47:50 INFO: hidden_dim: 128
16:47:50 INFO: num_layers: 2
16:47:50 INFO: rnn_dropout: 0.2
16:47:50 INFO: rnn_bidir: False
16:47:50 INFO: transformer_nhead: 4
16:47:50 INFO: transformer_dim_feedforward: 256
16:47:50 INFO: transformer_dropout: 0.2
16:47:50 INFO: max_patients: None
16:47:50 INFO: output_model_prefix: best_model
"""

log_data = """
17:06:58 INFO: Model=ImprovedRNN Accuracy=0.9260 F1=0.0000 Precision=0.0000 Recall=0.0000 AUROC=0.6143 AUPRC=0.0565
17:06:58 INFO: Model=ImprovedLSTM Accuracy=0.9435 F1=0.0000 Precision=0.0000 Recall=0.0000 AUROC=0.5918 AUPRC=0.0609
17:06:58 INFO: Model=ImprovedTransformer Accuracy=0.9537 F1=0.0000 Precision=0.0000 Recall=0.0000 AUROC=0.5357 AUPRC=0.0549
17:06:58 INFO: Model=MLP Accuracy=0.9347 F1=0.0337 Precision=0.0536 Recall=0.0246 AUROC=0.5925 AUPRC=0.0546
17:06:58 INFO: Model=TCN Accuracy=0.9541 F1=0.0163 Precision=1.0000 Recall=0.0082 AUROC=0.7490 AUPRC=0.1718
17:06:58 INFO: Model=NeuralODE Accuracy=0.9165 F1=0.1200 Precision=0.1172 Recall=0.1230 AUROC=0.5851 AUPRC=0.0795
"""

# Parse log data using regex
pattern = r"Model=(\w+) Accuracy=(\d\.\d+) F1=(\d\.\d+) Precision=(\d\.\d+) Recall=(\d\.\d+) AUROC=(\d\.\d+) AUPRC=(\d\.\d+)"
matches = re.findall(pattern, log_data)

# Create DataFrame
columns = ["Model", "Accuracy", "F1", "Precision", "Recall", "AUROC", "AUPRC"]
df = pd.DataFrame(matches, columns=columns)
df[columns[1:]] = df[columns[1:]].astype(float)

# Plot each metric in a separate figure as individual bar plots
for metric in ["Accuracy", "F1", "AUROC", "AUPRC"]:
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df[metric])
    plt.title(f"{metric} by Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

