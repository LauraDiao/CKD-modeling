Here’s a GitHub-ready `README.md` version of your writeup. It’s polished, explanatory, and appropriate for both research and implementation audiences:

---

# CKD Progression Prediction

This repository contains a PyTorch-based framework for modeling longitudinal progression of **Chronic Kidney Disease (CKD)** using visit-level embedding sequences. The primary task is to forecast whether a patient will progress to **stage 4 or higher** CKD at the *next clinical encounter*, based on a fixed-size window of prior visits.

The current implementation uses a gated recurrent unit (GRU) architecture trained on time-ordered sequences of pretrained patient visit embeddings.

---

### 🧠 Model Overview

The core model is a GRU-based sequence classifier. Given a sequence of embedding vectors \( \mathbf{x}_1, \dots, \mathbf{x}_T \in \mathbb{R}^{768} \) representing patient visits, the GRU updates a hidden state \( \mathbf{h}_t \in \mathbb{R}^{128} \) over time via the standard update and reset gate mechanism:

\[
\begin{aligned}
\mathbf{z}_t &= \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) \\
\mathbf{r}_t &= \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) \\
\tilde{\mathbf{h}}_t &= \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
\end{aligned}
\]

The final hidden state \( \mathbf{h}_T \) is then passed through a linear classifier to produce binary logits, which are interpreted as the probability of CKD progression at the next timepoint.

---

### ✅ Prediction Setup

This framework implements a realistic and clinically sound longitudinal prediction strategy:

- **Supervision Target**: CKD labels are binarized (`stage ≥ 4 → 1`) and shifted *forward in time* to define the next-visit prediction target.
  
- **Temporal Supervision**: For each patient, we construct sequences of embeddings using a sliding window approach. Each input sequence spans up to `T = 5` prior visits, and the label is the CKD state at the next visit.

- **Padding**: Sequences with fewer than `T` historical visits are left-padded with zero vectors, maintaining consistent input dimensionality while preserving visit chronology.

- **Patient-level Split**: To prevent information leakage, train/val/test splits are performed at the **patient level**, not the visit level.

- **Pretrained Visit Embeddings**: The model operates directly on fixed-size visit-level embeddings (e.g., obtained via BERT or other foundation models). This allows us to focus on modeling **temporal dynamics** rather than language understanding.

---

### 📈 What the GRU Learns

The model approximates a latent dynamical system that encodes the evolution of disease state. If we treat the GRU as learning a transition function \( f_\theta \), we get:

\[
\mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, \mathbf{x}_t)
\]

which implicitly captures temporal dependencies in clinical trajectory, including trends, abrupt changes, and the momentum of decline. This is especially powerful in the CKD setting, where disease evolution is nonlinear and temporally irregular.

From a systems perspective, the RNN is approximating a partially observable Markov decision process (POMDP), where visits are observations and hidden health status is propagated over time through its learned hidden states.

---

### 🚀 Getting Started

Clone the repo and run the training script with:

```bash
python train_ckd_model.py \
    --embedding-root ./ckd_embeddings_100 \
    --metadata-file patient_embedding_metadata.csv
```

You can modify model hyperparameters, data paths, and training settings via the CLI.

---

### 📂 Files

- `train_ckd_model.py`: End-to-end training and evaluation script.
- `CKDSequenceDataset`: Custom `torch.utils.data.Dataset` for loading padded sequences.
- `LongitudinalRNN`: GRU-based PyTorch model for sequence classification.

---

### 🔬 Citation

If you use this code for your research, please consider citing the repository or dropping a star to support the project.

---

Let me know if you'd like the README to include multiple model variants (e.g., Transformer, MLP, TCN) or a benchmarking table comparing them.


# eGFR-TFT
Can we forecast eGFR to predict future trajectories of patients and also model interventions into it

---
### Environment
Ensure you have Python (>=3.8) and the required packages installed.
```
pip install pandas numpy scikit-learn pytorch-lightning pytorch-forecasting torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To **run the enhanced code**, follow these steps:

##  **Prepare Data**
### **Create a `data/` directory and put your MIMIC-like CSV**
You need a CSV file (e.g., `mimic_demo.csv`) in the following format:

| patient_id | timestamp  | creatinine | age | gender | race  | eGFR | intervention_X | intervention_Y |
|------------|------------|------------|----|------|------|------|---------------|---------------|
| 1          | 2024-01-01 | 1.2        | 65 | male | white | 75   | 0             | 1             |
| 1          | 2024-01-10 | 1.3        | 65 | male | white | 72   | 1             | 0             |
| 2          | 2024-01-02 | 0.8        | 45 | female | black | 90 | 0             | 0             |

If you **don't have access real data**, we will supply a sample dataset.

---
Navigate to the project folder and run:

```bash
python train.py
```

This:
- Loads & preprocesses the **`mimic_demo.csv`** dataset.
- Simulates new interventions (if needed).
- Trains a **Temporal Fusion Transformer (TFT)** model.
- Saves the model to `tft_model.ckpt`.

If you have **GPU available**, it will automatically be used.

Once training is complete, you can use the trained model to **predict eGFR trajectories** for new patients.

First, ensure you have a **new test dataset** (`mimic_demo_new.csv`) formatted like your training data. Then, run:

```bash
python predict.py
```

This will:
- Load the trained **TFT model**.
- Process `mimic_demo_new.csv` for inference.
- Generate **future eGFR predictions**.
- Print example predictions.

## Simulate "What If" Scenarios**
To explore the **effect of new interventions**, run:

```bash
python scenario_runner.py
```

This script:
- Loads the trained model.
- Modifies a **specific patient's data** (e.g., applying **med_B** at time `t=30`).
- Runs **forecasting with and without the intervention**.
- Outputs the predicted **eGFR trajectory under different conditions**.

This is useful for **digital twin simulations**—predicting patient outcomes **with vs. without** certain interventions.

## **Customize Interventions**
If you want to add new medications, diagnostics, or procedures:
1. **Update the intervention list in `train.py`**
   ```python
   possible_interventions = ["med_A", "med_B", "diag_X", "new_med"]
   ```
2. **Ensure your dataset contains columns for them** (or they will be simulated).
3. Run training and predictions again.

### **Key Notes**
✅ **Runs on CPU or GPU** (automatically detects).  
✅ **Modular**—you can modify interventions, add new predictors, or change model architecture.  
✅ **Saves trained models** so you can reuse them without retraining.

