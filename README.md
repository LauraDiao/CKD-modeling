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

