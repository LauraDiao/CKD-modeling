#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datetime import datetime


# change variables 
icd_file = "/opt/data/commonfilesharePHI/ldiao/ckd_project/icd_mapping.csv"
output_fname = 'patient_embedding_metadata.csv'

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient-day notes, map GFR to CKD stages, and generate embeddings using a transformer model."
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the main event CSV file for this job.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory in which to save the generated embeddings and metadata.")
    parser.add_argument("--model_name", type=str, default="/opt/data/commonfilesharePHI/slee/MEME/clinicalBERT-emily",
                        help="Pretrained transformer model to use for embeddings.")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="Dimension to which the model embedding should be truncated or padded.")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for encoding the synthetic notes.")
    parser.add_argument("--cpu_id", type=int, required=True,
                        help="CPU core to which this process should be locked.")
    parser.add_argument("--custom_separator", action='store_true',
                        help="Use a '$' separator for the input CSV file.")
    return parser.parse_args()

def load_data(csv_path, icd_path, custom_separator):
    print(f"[INFO] Loading patient events from: {csv_path}")
    separator = '$' if custom_separator else ','
    df = pd.read_csv(csv_path, sep=separator, low_memory=False).drop_duplicates()
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

def gfr_to_stage(gfr):
    if pd.isna(gfr):
        return None, 0
    if gfr >= 90:
        return "1", 1
    elif gfr >= 60:
        return "2", 2
    elif gfr >= 45:
        return "3a", 3.1
    elif gfr >= 30:
        return "3b", 3.2
    elif gfr >= 15:
        return "4", 4
    else:
        return "5", 5

def forward_fill_ckd_stage(summary_df):
    summary_df = summary_df.sort_values(by=["PatientID", "EventDate"]).copy()
    summary_df["GFR"] = pd.to_numeric(summary_df["GFR"], errors="coerce")
    summary_df["GFR"] = summary_df.groupby("PatientID")["GFR"].ffill()
    
    new_stages = {}
    for pid, group in summary_df.groupby("PatientID"):
        group = group.sort_values("EventDate")
        max_stage_rank = 0
        for idx, row in group.iterrows():
            computed_stage, rank = gfr_to_stage(row["GFR"])
            if rank < max_stage_rank:
                final_stage = new_stages.get(prev_idx, computed_stage)
            else:
                final_stage = computed_stage
                max_stage_rank = rank
            new_stages[idx] = final_stage
            prev_idx = idx
    summary_df["CKD_stage"] = summary_df.index.map(new_stages)
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
    
    metadata_dir = os.path.join(output_dir, "metadata_chunks")
    os.makedirs(metadata_dir, exist_ok=True)

    chunk_size = 1000
    chunk_count = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding notes in batches"):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_gfrs = gfrs[i:i+batch_size]

        emb = get_cls_embeddings(batch_texts, tokenizer, model, device, embed_dim)

        for (pid, date), gfr_val, vec in zip(batch_ids, batch_gfrs, emb):
            patient_folder = os.path.join(output_dir, str(pid))
            os.makedirs(patient_folder, exist_ok=True)

            date_str = pd.to_datetime(date).strftime('%Y%m%d')
            fname = f"{pid}_{date_str}.npz"
            fpath = os.path.join(patient_folder, fname)
            np.savez_compressed(fpath, cls_embedding=vec)
            
            stage_val = summary_df[(summary_df['PatientID'] == pid) & (summary_df['EventDate'] == date)]['CKD_stage'].values[0]
            
            meta.append({
                'PatientID': pid,
                'EventDate': date,
                'GFR': gfr_val,
                'CKD_stage': stage_val,
                'text': summary_df[(summary_df['PatientID'] == pid) & (summary_df['EventDate'] == date)]['text'].values[0],
                'embedding_file': os.path.join(str(pid), fname)
            })
            
            if len(meta) == chunk_size:
                start_index = chunk_count * chunk_size
                end_index = start_index + chunk_size -1
                meta_df = pd.DataFrame(meta)
                chunk_fname = f"patient_embedding_metadata_{start_index}_{end_index}.csv"
                meta_csv_path = os.path.join(metadata_dir, chunk_fname)
                meta_df.to_csv(meta_csv_path, index=False)
                print(f"[INFO] Saved metadata chunk to: {meta_csv_path}")
                meta = []
                chunk_count += 1
    
    if meta:
        start_index = chunk_count * chunk_size
        end_index = start_index + len(meta) - 1
        meta_df = pd.DataFrame(meta)
        chunk_fname = f"patient_embedding_metadata_{start_index}_{end_index}.csv"
        meta_csv_path = os.path.join(metadata_dir, chunk_fname)
        meta_df.to_csv(meta_csv_path, index=False)
        print(f"[INFO] Saved final metadata chunk to: {meta_csv_path}")

    print(f"[DONE] All embeddings and metadata chunks have been processed and saved.")

def main():
    args = parse_arguments()
    
    # Set CPU affinity
    os.sched_setaffinity(0, {args.cpu_id})
    print(f"Process locked to CPU core {args.cpu_id}")
    
    # Map CPU ID to CUDA device
    device_map = {
        0: "cuda:0",
        1: "cuda:1",
        2: "cuda:2",
        3: "cuda:3"
    }
    device = torch.device(device_map[args.cpu_id] if torch.cuda.is_available() and args.cpu_id in device_map else "cpu")
    print(f"Process using device: {device}")

    df, icd_map = load_data(args.input_csv, icd_file, args.custom_separator)
    demographic_map = build_demographic_map(df)
    
    summary_df = generate_synthetic_notes(df, demographic_map, icd_map)
    summary_df = forward_fill_ckd_stage(summary_df)
    print(summary_df.head())
    
    tokenizer, model = load_embedding_model(args.model_name, device)
    
    job_output_dir = os.path.join(args.output_dir, f"job_{args.cpu_id}")
    os.makedirs(job_output_dir, exist_ok=True)
    
    generate_and_save_embeddings(summary_df, tokenizer, model, device,
                                 args.embed_dim, args.batch_size, job_output_dir)

    print("End of Embedding Generation")

if __name__ == '__main__':
    main()