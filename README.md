# CKD Progression Modeling

This repository provides a full pipeline for simulating longitudinal patient-note data, extracting contextual embeddings from clinical notes, and training various deep learning architectures to predict chronic kidney disease (CKD) progression. The pipeline consists of two main scripts:

---

### `embedding.py`: Pseudonote (tabular to text) Generation + Embedding Extraction

This script reads tabular event data and ICD mappings to generate synthetic, per-patient, per-day clinical notes. These notes are constructed by integrating demographic attributes, diagnoses (ICD-10 codes), medications, and procedures. The script maps GFR readings to CKD stages (1--5) using forward-filled values while enforcing monotonic progression. The resulting notes are encoded into dense representations (CLS embeddings) using a pretrained transformer model (e.g., ClinicalBERT).

Each embedding is stored in a per-patient directory with corresponding metadata (GFR, CKD stage, text). Metadata is saved to a CSV file for downstream modeling.

**Usage:**
```bash
bash run_embedding.sh
```

---

### `ckd_modeling.py`: Longitudinal CKD Classification Models

This script reads the embeddings and associated metadata and builds sequences of patient-note embeddings. The task is to predict whether the patient will progress to stage 4 or higher CKD at the next time step. Labels are derived from cleaned CKD stages, converted into a binary label indicating whether stage 4+ is reached.

It supports training and evaluation of the following sequence models:
- RNN and LSTM with optional bidirectionality
- Transformer encoder with positional encoding
- MLP over flattened embedding windows
- Temporal Convolutional Network (TCN)
- Neural ODE (if `torchdiffeq` is installed)

Each model is trained using early stopping and learning rate scheduling. Test performance includes Accuracy, F1, Precision, Recall, AUROC, and AUPRC. An optional label-switch analysis tracks whether models anticipate CKD progression earlier than the ground truth.

**Usage:**
```bash
bash run_modeling.sh
```

### Metadata Format and Embedding Structure

The script `embedding.py` produces a metadata CSV file named `patient_embedding_metadata.csv`. Each row in the file corresponds to a single patient-day pair and contains the following fields:

```markdown
| PatientID | EventDate  | GFR  | CKD_stage | text                                   | embedding_file            |
|-----------|------------|------|------------|----------------------------------------|---------------------------|
| Z1062220  | 2022-07-14 | 89.0 | 2          | ions: Ophthalmics - Misc.              | Z1062220_20220714.npz     |
| Z1062220  | 2022-07-15 | 89.0 | 2          | Central Muscle Relaxants               | Z1062220_20220715.npz     |
| Z1062220  | 2022-07-19 | 89.0 | 2          | uptake Inhibitors (SNRIs)              | Z1062220_20220719.npz     |
| Z1062220  | 2022-07-20 | 89.0 | 2          | ions: Antiperistaltic Agents           | Z1062220_20220720.npz     |
| Z1062220  | 2022-07-21 | 89.0 | 2          | ons: Antiperistaltic Agents            | Z1062220_20220721.npz     |
| Z1062220  | 2022-07-25 | 89.0 | 2          | cations: Analgesics Other              | Z1062220_20220725.npz     |
| Z1062220  | 2022-07-27 | 89.0 | 2          | counter: APPOINTMENT                   | Z1062220_20220727.npz     |
| Z1062220  | 2022-08-01 | 89.0 | 2          | s: Penicillin Combinations             | Z1062220_20220801.npz     |
| Z1062220  | 2022-08-03 | 89.0 | 2          | e - Labs: WBC  - Labs: K               | Z1062220_20220803.npz     |
| Z1062220  | 2022-08-04 | 89.0 | 2          | 110.1T: Unknown condition              | Z1062220_20220804.npz     |
| Z1062220  | 2022-08-10 | 89.0 | 2          | order, recurrent, moderate             | Z1062220_20220810.npz     |
| Z1062220  | 2022-08-15 | 89.0 | 2          | ions: Oil Soluble Vitamins             | Z1062220_20220815.npz     |
| Z1062220  | 2022-08-16 | 89.0 | 2          | ions: Antibiotics - Topical            | Z1062220_20220816.npz     |
| Z1062220  | 2022-08-17 | 89.0 | 2          | ions: Ophthalmics - Misc.              | Z1062220_20220817.npz     |
```

Each embedding is stored in a compressed `.npz` file and follows the folder structure:
```
{output_dir}/{PatientID}/{PatientID}_{EventDate}.npz
```
For instance, embeddings for patient `Z1062220` would be located at:
```
ckd_embeddings_100/Z1062220/Z1062220_20220714.npz
ckd_embeddings_100/Z1062220/Z1062220_20220715.npz
...
```

This design allows efficient access of per-patient longitudinal embeddings for modeling.

---

### Shell Scripts

**`run_embedding.sh`**
```bash
#!/bin/bash
python embedding.py \
  --csv patients_subset_100.csv \
  --icd icd_mapping.csv \
  --output_dir ckd_embeddings_100 \
  --model_name /home2/simlee/share/slee/GeneratEHR/clinicalBERT-emily \
  --embed_dim 768 \
  --batch_size 128
```

**`run_modeling.sh`**
```bash
#!/bin/bash
python ckd_modeling.py \
  --embedding-root ckd_embeddings_100 \
  --window-size 10 \
  --embed-dim 768 \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-3 \
  --patience 5 \
  --scheduler-patience 2 \
  --metadata-file patient_embedding_metadata.csv \
  --hidden-dim 128 \
  --num-layers 2 \
  --rnn-dropout 0.2 \
  --rnn-bidir \
  --transformer-nhead 4 \
  --transformer-dim-feedforward 256 \
  --transformer-dropout 0.2
```

Make sure to `chmod +x run_embedding.sh run_modeling.sh` before executing the scripts.

#### Acknowledgments

This work utilized resources provided by the [UCLA Department of Computational Medicine](https://compmed.ucla.edu/).

---

For further inquiries or collaboration, please contact:
- Simon A. Lee: [simonlee711@g.ucla.edu](mailto:simonlee711@g.ucla.edu)
- Jeffrey N. Chiang: [njchiang@g.ucla.edu](mailto:njchiang@g.ucla.edu)

