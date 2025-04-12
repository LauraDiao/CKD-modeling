#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient-day notes and embeddings using a transformer model."
    )
    parser.add_argument("--csv", type=str, default="patients_subset_10.csv",
                        help="Path to the main event CSV file.")
    parser.add_argument("--icd", type=str, default="icd_mapping.csv",
                        help="Path to the ICD mapping CSV file.")
    parser.add_argument("--output_dir", type=str, default="merged_daily_embeddings",
                        help="Directory in which to save the generated embeddings and metadata.")
    parser.add_argument("--model_name", type=str, default="/home2/simlee/share/slee/GeneratEHR/clinicalBERT-emily",
                        help="Pretrained transformer model to use for embeddings.")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Dimension to which the model embedding should be truncated or padded.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for encoding the synthetic notes.")
    return parser.parse_args()

def load_data(csv_path, icd_path):
    print(f"[INFO] Loading patient events from: {csv_path}")
    # low_memory=False suppresses mixed-type warnings; duplicate rows are removed.
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.drop_duplicates()
    icd_df = pd.read_csv(icd_path)

    df['DataCategory'] = df['DataCategory'].fillna('None')
    df['DataNumeric'] = df['DataNumeric'].fillna('None')
    df['EventTimeStamp'] = pd.to_datetime(df['EventTimeStamp'], errors='coerce')
    df['EventDate'] = df['EventTimeStamp'].dt.date
    df['is_gfr'] = df['DataCategory'].str.upper().str.contains('GFR|GFREST', na=False)

    icd_df["icd_code"] = icd_df["icd_code"].astype(str).str.replace(".", "", regex=False)
    icd_map = dict(zip(icd_df["icd_code"], icd_df["long_title"]))
    return df, icd_map

def format_demographics(row):
    # When grouping by PatientID without resetting index the PatientID is in row.name.
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
        gfr = None

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

            if row['is_gfr']:
                try:
                    gfr_candidate = float(num)
                    if gfr is None:
                        gfr = gfr_candidate
                except Exception:
                    continue

        full_note = "\n".join(note_lines)
        records.append({'PatientID': pid, 'EventDate': date, 'text': full_note, 'GFR': gfr})

    summary_df = pd.DataFrame(records)
    print(f"[INFO] Generated {len(summary_df)} synthetic patient-day notes.")
    return summary_df

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
    gfrs = summary_df['GFR'].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding notes in batches"):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_gfrs = gfrs[i:i+batch_size]

        emb = get_cls_embeddings(batch_texts, tokenizer, model, device, embed_dim)

        for (pid, date), gfr_val, vec in zip(batch_ids, batch_gfrs, emb):
            # Create a dedicated folder for the patient.
            patient_folder = os.path.join(output_dir, str(pid))
            os.makedirs(patient_folder, exist_ok=True)

            date_str = pd.to_datetime(date).strftime('%Y%m%d')
            fname = f"{pid}_{date_str}.npz"
            fpath = os.path.join(patient_folder, fname)
            np.savez_compressed(fpath, cls_embedding=vec)
            meta.append({
                'PatientID': pid,
                'EventDate': date,
                'GFR': gfr_val,
                'text': summary_df[(summary_df['PatientID'] == pid) & (summary_df['EventDate'] == date)]['text'].values[0],
                'embedding_file': os.path.join(str(pid), fname)
            })

    meta_df = pd.DataFrame(meta)
    meta_csv_path = os.path.join(output_dir, 'patient_embedding_metadata.csv')
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
    print(summary_df.head())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_embedding_model(args.model_name, device)
    generate_and_save_embeddings(summary_df, tokenizer, model, device,
                                 args.embed_dim, args.batch_size, args.output_dir)

if __name__ == '__main__':
    main()
