# CKD Progression Prediction

This script sets up a valid and coherent framework for longitudinal classification using a GRU-based recurrent neural network. The task appears to be predicting progression of **Chronic Kidney Disease (CKD)**, specifically forecasting whether a patient will reach stage 4 or above (binary label) at the *next time point* given a window of prior visits encoded as embedding vectors. Here's what's going on in terms of modeling, sequence setup, and what the RNN is actually learning.

---

### **What the RNN is doing**

The core model is a `GRU`, which is a gated recurrent unit. This means that for each timestep \( t \), the GRU takes in an embedding vector \( \mathbf{x}_t \in \mathbb{R}^{768} \) and updates a hidden state \( \mathbf{h}_t \in \mathbb{R}^{128} \) according to gating functions that balance how much new input is incorporated and how much old state is retained. Formally, each timestep executes something like:

\[
\begin{align*}
\mathbf{z}_t &= \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z) \\
\mathbf{r}_t &= \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r) \\
\tilde{\mathbf{h}}_t &= \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h) \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
\end{align*}
\]

where \( \mathbf{z}_t \) and \( \mathbf{r}_t \) are the update and reset gates. This mechanism enables the GRU to propagate useful memory over variable-length sequences while dampening irrelevant history, which is ideal for longitudinal health data where visit frequency and severity vary by patient.

After consuming the full sequence of embeddings \( [\mathbf{x}_1, \dots, \mathbf{x}_T] \), the final hidden state \( \mathbf{h}_T \) (here accessed as `h_n[-1]`) summarizes the patient's state over the window. The classifier linearly maps \( \mathbf{h}_T \) to logits for binary prediction:

\[
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{W}_{\text{clf}} \mathbf{h}_T + \mathbf{b}_{\text{clf}})
\]

---

### **Is this a valid prediction setup?**

Yes, and actually it's a *very* reasonable way to handle patient trajectory prediction. Here’s why it works:

- **Supervised learning target**: You define `label` as whether a visit corresponds to CKD stage ≥4. Then, crucially, you create `next_label` by shifting that label column *forward in time* within each patient. This turns the problem into a next-visit prediction task.
  
- **Temporal sequence construction**: `build_sequences()` iterates over patients and builds sliding windows of embeddings of size \( T \leq 5 \). The input to the model is a sequence of embeddings up to time \( t \), and the target is the label at time \( t+1 \). This is appropriate temporal supervision.

- **Zero-padding**: The `pad_sequence()` function prepends zeros if the patient doesn't have enough prior visits. This lets you train with a consistent tensor shape while still preserving relative chronology (more recent events at the end of the sequence).

- **Patient-level split**: Importantly, train/val/test splits are done **by patient ID**, not visit. This avoids label leakage across time and ensures that the model is evaluated on unseen patients, which is crucial in clinical ML.

- **Use of pretrained embeddings**: The model is operating in a feature space that is already semantically enriched (e.g., maybe from ClinicalBERT or some custom embedder). This avoids retraining an entire transformer stack and focuses the RNN on modeling *temporal dynamics*, not language structure.

---

### **What is learned?**

This GRU learns a *representation of longitudinal disease progression* from fixed-size latent visit vectors. If the embeddings are informative about comorbidities, labs, and medications, then the GRU captures how sequences of such latent visits map onto future risk. It is likely learning transitions in health state, temporal correlation patterns, and maybe even the “acceleration” of decline (i.e., increasing rate of stage progression).

In a latent dynamical systems view, this is akin to inferring a state transition function \( f_\theta \) where:

\[
\mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, \mathbf{x}_t)
\]

and you're using this hidden state to predict future disease progression. In that sense, you're approximating a partially observable Markov decision process (POMDP) where your observations are sparse but sequentially dependent.

---

Let me know if you want to rework this into a Transformer-based temporal model or bring in structured missingness. There's plenty to riff on here.

---
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

