# %%
# embedding generation script
# modified
#!/usr/bin/env python
import os
import pandas as pd
import polars as pl 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# %%
# cuda
cuda_num = 3
print(f"cuda:{str(cuda_num)}")
# %%
# batch size
batch_size_ = 2048 # 1024

# change variables 
icd_file = "/opt/data/commonfilesharePHI/ldiao/ckd_project/icd_mapping.csv"
output_fname = 'patient_embedding_metadata.csv'

# generating subset to test embedding surv
custom_separator = False # <<
if not custom_separator: 
    subset_size = "25"  # 10, 100, all/full # <<
    output_dir = f"/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_embedding_{subset_size}"
    # output_dir = f"/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/ckd_embedding_{subset_size}"
    # event_file =  f"/opt/data/commonfilesharePHI/slee/ckd-optum/patients_subset_{subset_size}.csv"
    event_file =  f"/opt/data/workingdir/ldiao/ckd_project/patient_subsets/patients_subset_{subset_size}.csv"
if custom_separator:
    # output_dir = "/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_embedding_full"
    output_dir = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/ckd_embedding_full"
    event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt.parquet"

# icd script
output_dir += "_icd" # <<

# filter ckd stage
filter_ckd_stage = True # <<
if filter_ckd_stage: 
    output_dir  += "_stage_filter" 

print(output_dir)
# %%
try:
    os.mkdir(output_dir)
except FileExistsError:
    pass
# %% DEBUG
csv = event_file 
icd = icd_file 
output_dir = output_dir 
model_name= "/opt/data/commonfilesharePHI/slee/MEME/clinicalBERT-emily"
embed_dim = 768
batch_size=batch_size_  

os.makedirs(output_dir, exist_ok=True)

# %%
icd_df = pd.read_csv(icd)
icd_df["icd_code"] = icd_df["icd_code"].astype(str).str.replace(".", "", regex=False)
icd_map = dict(zip(icd_df["icd_code"], icd_df["long_title"]))
# %%
df = pl.read_parquet(csv)
# embedding_subset
# if not custom_separator: 
#     df = pl.read_csv(
#         event_file,
#         infer_schema_length=None,
#         null_values="null",
#     ).unique()
# %%
custom_map = {
    1: 1, # "1",
    2: 2, # "2",
    3: 3, # "3",
    4: 4, # "4",
    5: 5, # "5",
    6: 6, # "ESRD",
    9: 0, # "CKD"
}
print("Building Filter")
ckd_icd_df = (
    df.filter(pl.col("DataCategory").str.contains("N18"))
    .with_columns(
      pl.col("DataCategory")
        .str.extract(r"N18\.([1-9])", 1)
        .cast(pl.Int64)
        .replace(custom_map, default=None)
        .alias("CKD_stage_numeric")
    )
    .select([pl.col("PatientID"), pl.col("EventTimeStamp"), pl.col("META_1"), pl.col("CKD_stage_numeric")])
    .with_columns(
        pl.col("CKD_stage_numeric")
            .max()
            .over("PatientID")
            .alias("max_stage")
    )
    .filter(pl.col("max_stage") >= 3)
    .unique()
) 
print("Filtering and converting")
df = (
    df
    .join(
        ckd_icd_df.select("PatientID").unique(), 
        on="PatientID", 
        how='inner')
    .drop_nulls(subset=["DataNumeric"])
    .with_columns([
        pl.col("EventTimeStamp").str.strptime(pl.Datetime("us"))
        , pl.col("DataCategory").fill_null(pl.col("META_2"))
    ])
    .with_columns(
        pl.col("EventTimeStamp").dt.date().alias("EventDate")
    )

)

# %% demographics and encounter

demographic_df = (
    df.filter(pl.col("DataType") == "Demographics")
    .drop_nulls(subset="DataCategory")
    .group_by("PatientID")
    .first()
    .with_columns(
        pl.format(
            "Patient {} is a {} patient."
            , pl.col("PatientID")
            , pl.col("DataCategory")
                .str.replace("//", " ")
                .str.replace("/", " or ")
                .str.replace("Do not identify with Race", "unknown race")
                .str.replace("Unknown Not Reported", "")
                .str.strip_chars()
        ).alias("sentence")
    )
)

demographic_map = dict(zip(demographic_df["PatientID"], demographic_df["sentence"]))

# %% Constructing the clauses to append to sentences
sentences_df = (
    df.filter(~pl.col("DataType").is_in(["Demographics", "Encounter"]))
    .with_columns(
        pl.when(pl.col("DataType") == "Diagnosis").then(
            pl.col("DataCategory")
                .str.replace(".", "", literal=True)
                .replace(icd_map, default="Unknown")
        ).otherwise(pl.format("NA"))
        .alias("long_title")
        , pl.when((pl.col("DataType") == "Encounter") & 
                (pl.col("DataCategory").str.contains("//"))).then(
            pl.col("DataCategory").str.replace("//", ": ")
        ).otherwise(pl.format("NA"))
        .alias("EncounterType")
    )
    .filter(pl.col("long_title") != "Unknown") # mostly ICD-9
    .with_columns(
        pl.when(pl.col("DataType") == "Diagnosis").then(
            # pl.format("  - ICD-10 code {}", pl.col("DataCategory"))
            pl.format(" - ICD-10 code {}: {}", pl.col("DataCategory"), pl.col("long_title"))
        )
        .when(pl.col("DataType") == "Medications").then(
            pl.format(" - Medication administered: {}", pl.col("DataCategory"))
        )
        .when(pl.col("DataType") == "Procedure").then(
            pl.format(" - Procedure performed: {}", pl.col("DataCategory"))
        )
        .when(pl.col("DataType") == "Labs").then(
            pl.format(" - {}: {}", pl.col("DataCategory"), pl.col("DataNumeric"))
        )
        .when(pl.col("DataType") == "Encounter").then(
            pl.when(pl.col("EncounterType") != "NA").then(
                pl.format(" - {}", pl.col("DataCategory"))
            ).otherwise(
                pl.format(" - {}: {}", pl.col("DataCategory"), pl.col("DataNumeric"))
            )
        )
        .otherwise(pl.format("Not parsed."))
        .alias("sentence")
    )
    .drop(pl.col("META_2"))
    .unique()
)

# %%
day_grouped_df = (
    sentences_df.group_by("PatientID", "META_1", "EventDate")
      .agg([
        pl.col("sentence").str.concat(" ").alias("day_summary")
        , pl.col("EventDate").first().alias("date")
      ])
      .with_columns(
        pl.format("On {}, the patient had the following records: {}", pl.col("date"), pl.col("day_summary"))
        .alias("day_summary")
      )
)
# %%%
enc_grouped_df = (
    day_grouped_df.group_by("PatientID", "META_1")
        .agg([
            pl.col("day_summary").str.concat("\n").alias("enc_summary")
            , pl.col("EventDate").min().alias("date")
            , pl.col("META_1").first().alias("enc_id")
            , pl.col("PatientID").first().replace(demographic_map, default="Demographics not available.").alias("demographic_string")
        ])
        .with_columns(
            pl.format("Encounter {} starting {}. {} {}", pl.col("enc_id"), pl.col("date"), pl.col("demographic_string"), pl.col("enc_summary")).alias("enc_summary")
        )
    .drop("demographic_string")
    .sort("PatientID", "date")
    .with_columns(pl.int_range(pl.len()).over("PatientID").alias("emb_id"))
)

enc_grouped_df.head() # this is the final dataframe which we can treat as the meta. 

# %% generating embeddings
print(f"[INFO] Loading model from: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = f"cuda:{cuda_num}"
model.to(device)
model.eval()
# %%
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


# %%
# just do a for loop through all of them... should be fine... 
enc_grouped_df = enc_grouped_df.join(
    ckd_icd_df.select(["PatientID", "META_1", "CKD_stage_numeric", "max_stage"])
    , on=["PatientID", "META_1"], how='left'
)

# changes
enc_grouped_df.write_csv(os.path.join(output_dir, "meta_v2.csv"))

# %%

print("Saved meta df")
patient_grouped_df = (
    enc_grouped_df.sort(["PatientID", "emb_id"])
        .group_by("PatientID")
        .agg(pl.col("enc_summary").alias("pseudonotes"))
)
grouped_dict = dict(zip(patient_grouped_df["PatientID"], patient_grouped_df["pseudonotes"]))
print("Generating embeddings")
for pid, texts in tqdm(grouped_dict.items()):
    patient_folder = os.path.join(output_dir, str(pid))
    os.makedirs(patient_folder, exist_ok=True)

    embeddings = [get_cls_embeddings(t, tokenizer, model, device, embed_dim) for t in texts]
    fname = f"{pid}"
    fpath = os.path.join(patient_folder, fname)
    np.save(fpath, np.array(embeddings).squeeze())



# %%
# df = pd.read_csv("/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt", sep="$")
# df.iloc[:-3].to_parquet(f"{csv}.parquet", engine="pyarrow", compression="zstd", index=False)

# %%

# %%
