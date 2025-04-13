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

