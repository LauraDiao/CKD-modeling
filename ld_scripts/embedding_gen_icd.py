# embedding generation script
# modified
#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datetime import datetime

# ==================================
# Boolean Configuration
# ==================================
custom_separator = True # << Controls file reading format ($ or ,)
filter_ckd_stage = True # << Controls filtering to patients who reached CKD stage >= 3
# this script uses icd > gfr by default
# ==================================

output_version_suffix = "_v2" # << for versioning output folders

# cuda
cuda_num = 3
print(f"cuda:{str(cuda_num)}")

# batch size
batch_size_ = 2048 # 1024

# change variables 
icd_file = "/opt/data/commonfilesharePHI/ldiao/ckd_project/icd_mapping.csv"
output_fname = 'patient_embedding_metadata.csv'

# custom_separator = True # << REMOVED: Moved to Boolean Configuration block
if not custom_separator: 
    subset_size = "25"  # 10, 100, all/full # <<
    output_dir = f"/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_embedding_{subset_size}"
    # event_file =  f"/opt/data/commonfilesharePHI/slee/ckd-optum/patients_subset_{subset_size}.csv"
    event_file =  f"/opt/data/workingdir/ldiao/ckd_project/patient_subsets/patients_subset_{subset_size}.csv"
if custom_separator:
    output_dir = "/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_embedding_full"
    event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt"

# icd script
output_dir += "_icd" # <<

# filter ckd stage
# filter_ckd_stage = True # << REMOVED: Moved to Boolean Configuration block
if filter_ckd_stage: 
    output_dir  += "_stage_filter" 

output_dir  += output_version_suffix 
print(output_dir)

try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient-day notes, map ICD to CKD stages, and generate embeddings using a transformer model."
    )
    parser.add_argument("--csv", type=str, default=event_file,
                        help="Path to the main event CSV file.")
    parser.add_argument("--icd", type=str, default=icd_file,
                        help="Path to the ICD mapping CSV file.")
    parser.add_argument("--output_dir", type=str, default=output_dir,
                        help="Directory in which to save the generated embeddings and metadata.")
    parser.add_argument("--model_name", type=str, default="/opt/data/commonfilesharePHI/slee/MEME/clinicalBERT-emily",
                        help="Pretrained transformer model to use for embeddings.")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="Dimension to which the model embedding should be truncated or padded.")
    parser.add_argument("--batch_size", type=int, default=batch_size_,
                        help="Batch size for encoding the synthetic notes.")
    return parser.parse_args()

def load_data(csv_path, icd_path):
    print(f"[INFO] Loading patient events from: {csv_path}")
    # Set low_memory to False to suppress dtype warnings for mixed types.
    if custom_separator: 
        df = pd.read_csv(csv_path, sep='$', low_memory=False).drop_duplicates()
    else: 
        df = pd.read_csv(csv_path, low_memory=False).drop_duplicates()
    df = df.drop_duplicates()
    icd_df = pd.read_csv(icd_path)

    df['DataCategory'] = df['DataCategory'].fillna('None')
    df['DataNumeric'] = df['DataNumeric'].fillna('None')
    df['EventTimeStamp'] = pd.to_datetime(df['EventTimeStamp'], errors='coerce')
    df['EventDate'] = df['EventTimeStamp'].dt.date
    df['is_icd'] = df['DataCategory'].str.upper().str.contains("(?i)^N18\..*", na=False)

    icd_df["icd_code"] = icd_df["icd_code"].astype(str).str.replace(".", "", regex=False)
    icd_map = dict(zip(icd_df["icd_code"], icd_df["long_title"]))
    return df, icd_map

def format_demographics(row):
    # When grouping by PatientID without resetting index, PatientID is in row.name.
    pid = row.name
    race_ethnicity = str(row["DataCategory"]).replace("//", " ").replace("/", " ")
    if "Unknown Not Reported" in race_ethnicity:
        race_ethnicity = race_ethnicity.replace("Unknown Not Reported", "").strip()
    if "Do not identify with Race" in race_ethnicity:
        race_ethnicity = race_ethnicity.replace("Do not identify with Race", "unknown race").strip()
    return f"Patient {pid} is a {race_ethnicity} patient."

def build_demographic_map(df):
    demographics = df[df["DataType"] == "Demographics"].dropna(subset=["DataCategory"])
    demographic_map = (
        demographics.groupby("PatientID")
        .first()
        .apply(format_demographics, axis=1)
        .to_dict()
    )
    return demographic_map

def generate_synthetic_notes(df, demographic_map, icd_map):
    events = df[df["DataType"] != "Demographics"].copy()
    grouped = events.groupby(['PatientID', 'EventDate'])
    records = []

    for (pid, date), group in tqdm(grouped, desc="Formatting synthetic notes"):
        note_lines = []
        icd = None

        if pid in demographic_map:
            note_lines.append(demographic_map[pid])
        else:
            note_lines.append(f"Patient {pid} demographics information not available.")

        date_str = datetime.strftime(pd.Timestamp(date), "%Y-%m-%d")
        note_lines.append(f"On {date_str}, the patient had the following records:")

        for _, row in group.iterrows():
            dt, cat, num = row['DataType'], row['DataCategory'], row['DataNumeric']
            if dt == "Diagnosis":
                icd_code = str(cat).replace(".", "")
                icd_title = icd_map.get(icd_code, "Unknown condition")
                note_lines.append(f"  - ICD-10 code {cat}: {icd_title}")
            elif dt == "Medication":
                note_lines.append(f"  - Medication administered: {cat}")
            elif dt == "Procedure":
                note_lines.append(f"  - Procedure performed: {cat}")
            else:
                note_lines.append(f"  - {dt}: {cat}")

            if row['is_icd']:
                try:
                    icd_candidate = str(cat)
                    if icd is None:
                        icd = icd_candidate
                except Exception:
                    continue

        full_note = "\n".join(note_lines)
        records.append({'PatientID': pid, 'EventDate': date, 'text': full_note, 'ICD': icd})

    summary_df = pd.DataFrame(records)
    print(f"[INFO] Generated {len(summary_df)} synthetic patient-day notes.")
    return summary_df

def icd_to_stage(icd):
    """
        'N18.1%': 1,
        'N18.2%': 2, 
        'N18.3%': 3, 
        'N18.4%': 4, 
        'N18.5%': 5, 
        'N18.6%': 'ESRD', 
        'N18.9%': 'CKD'
    """
    if pd.isna(icd):
        return None, 0
    if icd in 'N18.1%':
        return "1", 1
    if icd in 'N18.2%':
        return "2", 2
    if icd in 'N18.3%':
        return "3", 3
    if icd in 'N18.4%':
        return "4", 4
    return "5", 5

def forward_fill_ckd_stage(summary_df):
    """
    For each patient, forward-fill the ICD values (sorted by date), and map them to CKD stages
    based on conversions in icd_to_stage
    The stage is forced to be non-decreasing (i.e. if a new reading would lead to an improvement,
    the previous worse stage is retained).
    """
    summary_df = summary_df.sort_values(by=["PatientID", "EventDate"]).copy()
    # Convert ICD to numeric (if not already) and forward fill per patient.
    # summary_df["ICD"] = pd.to_numeric(summary_df["ICD"], errors="coerce")
    summary_df["ICD"] = summary_df.groupby("PatientID")["ICD"].ffill()
    
    # For each patient, enforce non-decreasing (progressive) stage.
    new_stages = {}
    for pid, group in summary_df.groupby("PatientID"):
        group = group.sort_values("EventDate")
        max_stage_rank = 0
        for idx, row in group.iterrows():
            computed_stage, rank = icd_to_stage(row["ICD"])
            # If the computed stage is less severe than the worst seen so far, retain the worst.
            if rank < max_stage_rank:
                final_stage = new_stages.get(prev_idx, computed_stage)
            else:
                final_stage = computed_stage
                max_stage_rank = rank
            new_stages[idx] = final_stage
            prev_idx = idx
    summary_df["CKD_stage"] = summary_df.index.map(new_stages)
    return summary_df

# testing
def clean_ckd_stage(value):
    try:
        return int(value)
    except:
        if isinstance(value, str) and value[0].isdigit():
            return int(value[0])
        else:
            return np.nan
            
def filter_patients_by_ckd_stage(df, ckd_stage_col, patient_id_col='PatientID'):
    initial_patients = df[patient_id_col].nunique()
    # Filter for visits where CKD stage is 3 or higher
    df_at_or_above_stage_3 = df[df[ckd_stage_col] >= 3]
    # Get unique PatientIDs from this filtered DataFrame
    patient_ids_to_keep = set(df_at_or_above_stage_3[patient_id_col].unique())
    
    patients_removed = initial_patients - len(patient_ids_to_keep)

    return patient_ids_to_keep    

def process_ckd_stage(df, filtering_stage= filter_ckd_stage):
    # print(df)
    df['CKD_stage_clean'] = df['CKD_stage'].apply(clean_ckd_stage)
    df = df.sort_values(by=['PatientID', 'EventDate'])
    df['CKD_stage_clean'] = df.groupby('PatientID')['CKD_stage_clean'].bfill().ffill()
    df = df.dropna(subset=['CKD_stage_clean'])
    df['CKD_stage_clean'] = df['CKD_stage_clean'].astype(int)
    df['label'] = df['CKD_stage_clean'].apply(lambda x: 1 if x >= 4 else 0)

    if filtering_stage:
        df_patients = filter_patients_by_ckd_stage(df, 'CKD_stage_clean')
        df = df[df["PatientID"].isin(df_patients)].copy()

    return df

def load_embedding_model(model_name, device):
    print(f"[INFO] Loading model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def get_cls_embeddings(texts, tokenizer, model, device, embed_dim):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    if cls_emb.size(1) > embed_dim:
        cls_emb = cls_emb[:, :embed_dim]
    else:
        pad = embed_dim - cls_emb.size(1)
        cls_emb = torch.nn.functional.pad(cls_emb, (0, pad), value=0)
    return cls_emb.cpu().numpy()

def generate_and_save_embeddings(summary_df, tokenizer, model, device, embed_dim, batch_size, output_dir):
    meta = []
    texts = summary_df['text'].tolist()
    ids = list(zip(summary_df['PatientID'], summary_df['EventDate']))
    icds = summary_df['ICD'].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding notes in batches"):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_icds = icds[i:i+batch_size]

        emb = get_cls_embeddings(batch_texts, tokenizer, model, device, embed_dim)

        for (pid, date), icd_val, vec in zip(batch_ids, batch_icds, emb):
            # Create a folder for the patient if it doesn't exist.
            patient_folder = os.path.join(output_dir, str(pid))
            os.makedirs(patient_folder, exist_ok=True)

            date_str = pd.to_datetime(date).strftime('%Y%m%d')
            fname = f"{pid}_{date_str}.npz"
            fpath = os.path.join(patient_folder, fname)
            np.savez_compressed(fpath, cls_embedding=vec)
            # Look up the CKD stage from the summary dataframe.
            stage_val = summary_df[(summary_df['PatientID'] == pid) & (summary_df['EventDate'] == date)]['CKD_stage'].values[0]
            meta.append({
                'PatientID': pid,
                'EventDate': date,
                'ICD': icd_val,
                'CKD_stage': stage_val,
                'text': summary_df[(summary_df['PatientID'] == pid) & (summary_df['EventDate'] == date)]['text'].values[0],
                'embedding_file': os.path.join(str(pid), fname)
            })

    meta_df = pd.DataFrame(meta)
    meta_csv_path = os.path.join(output_dir, output_fname)
    print(meta_df.shape)
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"[DONE] Metadata saved to: {meta_csv_path}")

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    df, icd_map = load_data(args.csv, args.icd)
    demographic_map = build_demographic_map(df)
    print("Demographic mapping:")
    print(demographic_map)

    summary_df = generate_synthetic_notes(df, demographic_map, icd_map)
    # Forward-fill ICD values and compute CKD stage per patient
    summary_df = forward_fill_ckd_stage(summary_df)
    print(summary_df.head())
    # cuda 1, 0
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_embedding_model(args.model_name, device)
    generate_and_save_embeddings(summary_df, tokenizer, model, device,
                                 args.embed_dim, args.batch_size, args.output_dir)


    print("End of Embedding Generation")

if __name__ == '__main__':
    main()