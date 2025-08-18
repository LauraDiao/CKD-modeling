import polars as pl
from tqdm import tqdm
import os
import re
import logging

# -----------------------------
# Config
# -----------------------------
output_path = "./../../../commonfilesharePHI/ldiao/ckd_project/"
output_dir_m = output_path + "ckd_tab_m_v1_10"  # 10, 100, full
event_file =  "./../../../commonfilesharePHI/slee/ckd-optum/patients_subset_10.csv" # 10, 100, all - path to the main event CSV file
# event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt"  # path to all data
output_fname = "ckd_processed_tab.csv"

try:
    os.makedirs(output_dir_m, exist_ok=True)
except Exception as e:
    raise RuntimeError(f"Could not create output directory: {e}")

# -----------------------------
# Logging setup
# -----------------------------
log_path = os.path.join(output_dir_m, "tab_gen_m_polars.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()  # remove if only file logging desired
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Load and preprocess
# -----------------------------
df = (
    pl.scan_csv(
        event_file,
        separator="$",
        null_values="null",
        schema_overrides={"DataNumeric": pl.Float64}
    )
    .unique()
    .with_columns([
        pl.col("EventTimeStamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("EventTimeStamp"),
    ])
    .with_columns([
        pl.col("EventTimeStamp").dt.date().alias("EventDate"),
        pl.col("DataCategory").fill_null("None"),
        pl.col("DataNumeric").cast(pl.Float64)
    ])
)

# -----------------------------
# Base: full patient-day index
# -----------------------------
df = df.with_columns([
    pl.col("DataCategory").str.to_uppercase().str.contains("GFR|GFREST").alias("is_gfr")
])

all_days = (
    df.select(["PatientID", "EventDate"])
    .unique()
    .drop_nulls()
    .sort(["PatientID", "EventDate"])
)

# -----------------------------
# Extract and forward-fill GFR
# -----------------------------
gfr_daywise = (
    df.filter(pl.col("is_gfr") & pl.col("DataNumeric").is_not_null())
      .group_by(["PatientID", "EventDate"])
      .agg(pl.col("DataNumeric").first().alias("GFR_combined"))
)

base_df = (
    all_days.join(gfr_daywise, on=["PatientID", "EventDate"], how="left")
    .sort(["PatientID", "EventDate"])
    .with_columns([
        pl.col("GFR_combined").forward_fill().over("PatientID")
    ])
)

# -----------------------------
# CKD stage mapping
# -----------------------------
def gfr_to_stage(gfr):
    if gfr is None:
        return None
    if gfr >= 90: return "1"
    if gfr >= 60: return "2"
    if gfr >= 45: return "3a"
    if gfr >= 30: return "3b"
    if gfr >= 15: return "4"
    return "5"

pdf = base_df.collect().to_pandas()
new_stages = {}
for pid, group in pdf.groupby("PatientID"):
    group = group.sort_values("EventDate")
    max_rank = 0
    prev_idx = None
    for idx, row in group.iterrows():
        stage = gfr_to_stage(row["GFR_combined"])
        rank_map = {"1":1,"2":2,"3a":3.1,"3b":3.2,"4":4,"5":5,None:0}
        rank = rank_map.get(stage, 0)
        if rank < max_rank:
            stage = new_stages.get(prev_idx, stage)
        else:
            max_rank = rank
        new_stages[idx] = stage
        prev_idx = idx

pdf["CKD_stage"] = pdf.index.map(new_stages)
base_df = pl.from_pandas(pdf)

# -----------------------------
# Diagnosis one-hot encoding
# -----------------------------
def truncate_icd(code):
    code = str(code).strip().replace(" ", "")
    if '.' in code:
        prefix, suffix = code.split('.', 1)
        return f"{prefix}.{suffix[0]}" if suffix else prefix
    return code

diag_df = (
    df.filter(pl.col("DataType") == "Diagnosis")
      .with_columns(pl.col("DataCategory").map_elements(truncate_icd).alias("ICD_clean"))
      .group_by(["PatientID", "EventDate"])
      .agg(pl.col("ICD_clean").list())
      .collect()
)

all_icds = sorted({icd for sublist in diag_df["ICD_clean"] for icd in sublist})
for icd in all_icds:
    diag_df = diag_df.with_columns(
        pl.col("ICD_clean").list.contains(icd).cast(pl.Int8).alias(f"diag_{icd}")
    )
diag_df = diag_df.drop("ICD_clean")

base_df = base_df.join(diag_df, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Medications one-hot
# -----------------------------
med_df = (
    df.filter(pl.col("DataType") == "Medications")
      .with_columns(
          pl.col("DataCategory").cast(pl.Utf8).str.to_uppercase().str.replace(" ", "_").alias("med_clean")
      )
      .group_by(["PatientID", "EventDate"])
      .agg(pl.col("med_clean").list())
      .collect()
)

all_meds = sorted({m for sublist in med_df["med_clean"] for m in sublist})
for m in all_meds:
    med_df = med_df.with_columns(
        pl.col("med_clean").list.contains(m).cast(pl.Int8).alias(f"med_{m}")
    )
med_df = med_df.drop("med_clean")

base_df = base_df.join(med_df, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Labs pivot
# -----------------------------
lab_df = (
    df.filter((pl.col("DataType") == "Labs") & pl.col("DataNumeric").is_not_null())
      .with_columns(pl.col("DataCategory").cast(pl.Utf8).str.to_uppercase())
      .group_by(["PatientID", "EventDate", "DataCategory"])
      .agg(pl.col("DataNumeric").first())
      .pivot(values="DataNumeric", index=["PatientID", "EventDate"], columns="DataCategory")
      .collect()
)
lab_df = lab_df.rename({col: f"lab_{col}" for col in lab_df.columns if col not in ["PatientID", "EventDate"]})
base_df = base_df.join(lab_df, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Final report and save
# -----------------------------
logger.info(f"[INFO] Final tabular shape: {base_df.shape}")
logger.info(f"[INFO] Sample features:\n{base_df.head().to_pandas()}")
logger.info(f"[INFO] CKD stage counts:\n{base_df.select('CKD_stage').to_pandas().value_counts(dropna=False)}")

base_df.write_csv(os.path.join(output_dir_m, output_fname))
logger.info("End of Tabular Generation")
