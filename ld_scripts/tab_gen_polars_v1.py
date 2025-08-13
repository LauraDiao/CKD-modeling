import polars as pl
from tqdm import tqdm
import re
import os
import logging

# Define a function to apply to a column
def truncate_icd(code):
    code = str(code).strip().replace(" ", "")
    if '.' in code:
        prefix, suffix = code.split('.', 1)
        return f"{prefix}.{suffix[0]}" if suffix else prefix
    return code

# Setup logging
output_path = "./../../../commonfilesharePHI/ldiao/ckd_project/"
output_dir_m = output_path + "ckd_tab_m_10" # 10, 100, full; change
output_fname = "ckd_processed_tab.csv"
log_fname = "tab_gen_log.log"
event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt"

try:
    os.makedirs(output_dir_m, exist_ok=True)
except Exception as e:
    print(f"Error creating output directory: {e}")
    exit()

log_file_path = os.path.join(output_dir_m, log_fname)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Load and preprocess
# -----------------------------
df = pl.read_csv(event_file, separator='$', infer_schema_length=10000).unique()
logger.info(f"Initial DataFrame shape: {df.shape}")

df = (
    df.with_columns(
        pl.col('EventTimeStamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S.%f', strict=False),
    )
    .with_columns(
        pl.col('EventTimeStamp').dt.date().alias('EventDate'),
        pl.col('DataCategory').fill_null('None'),
        pl.col('DataNumeric').cast(pl.Float64, strict=False),
    )
)

# -----------------------------
# Base: full patient-day index
# -----------------------------
df = df.with_columns(
    pl.col('DataCategory').str.to_uppercase().str.contains("GFR|GFREST").alias("is_gfr")
)

all_days = (
    df.select(['PatientID', 'EventDate'])
    .unique()
    .sort(['PatientID', 'EventDate'])
    .drop_nulls()
)

# -----------------------------
# Extract and forward-fill GFR
# -----------------------------
gfr_df = df.filter(pl.col('is_gfr') & pl.col('DataNumeric').is_not_null())

gfr_daywise = (
    gfr_df.group_by(['PatientID', 'EventDate'])
    .agg(pl.col('DataNumeric').first().alias('GFR_combined'))
)

base_df = all_days.join(gfr_daywise, on=['PatientID', 'EventDate'], how='left')

base_df = base_df.sort(['PatientID', 'EventDate'])
base_df = base_df.with_columns(
    pl.col('GFR_combined').forward_fill().over('PatientID')
)

def gfr_to_stage(gfr):
    if gfr is None:
        return (None, 0)
    if gfr >= 90:
        return ("1", 1)
    if gfr >= 60:
        return ("2", 2)
    if gfr >= 45:
        return ("3a", 3.1)
    if gfr >= 30:
        return ("3b", 3.2)
    if gfr >= 15:
        return ("4", 4)
    return ("5", 5)

# Enforce monotonic CKD staging
base_df = base_df.with_columns(
    pl.struct(['GFR_combined'])
    .map_elements(lambda s: gfr_to_stage(s['GFR_combined']), return_dtype=pl.Struct([pl.Field("CKD_stage", pl.Utf8), pl.Field("CKD_rank", pl.Float64)]))
    .alias('stage_tuple')
)
base_df = base_df.with_columns(
    pl.col('stage_tuple').struct.field('CKD_stage').alias('CKD_stage'),
    pl.col('stage_tuple').struct.field('CKD_rank').alias('CKD_rank')
)

base_df = base_df.with_columns(
    pl.col('CKD_rank').cum_max().over('PatientID').alias('max_rank')
)

base_df = base_df.with_columns(
    pl.when(pl.col('CKD_rank') < pl.col('max_rank'))
    .then(pl.col('CKD_stage').backward_fill().over('PatientID'))
    .otherwise(pl.col('CKD_stage'))
    .alias('CKD_stage')
)

base_df = base_df.drop(['stage_tuple', 'CKD_rank', 'max_rank'])


# -----------------------------
# One-hot encode diagnoses (truncated ICD codes)
# -----------------------------
diag_df = df.filter(pl.col("DataType") == "Diagnosis")

diag_df = diag_df.with_columns(
    pl.col("DataCategory").map_elements(truncate_icd).alias("ICD_clean")
)

diagnosis_map = (
    diag_df.group_by(["PatientID", "EventDate"])
    .agg(pl.col("ICD_clean").list())
)

all_icd_codes = diagnosis_map.select(pl.col("ICD_clean").list.unique().flatten()).to_series().to_list()
all_icd_codes = [f"diag_{code}" for code in all_icd_codes]

diag_df_onehot = (
    diagnosis_map.with_columns(
        pl.lit(1).alias("val"),
        pl.col("ICD_clean").map_elements(lambda x: [f"diag_{c}" for c in x], return_dtype=pl.List(pl.Utf8)).alias("diag_names")
    )
    .pivot(values="val", columns="diag_names", index=["PatientID", "EventDate"], aggregate_function="first")
    .fill_null(0)
)
base_df = base_df.join(diag_df_onehot, on=["PatientID", "EventDate"], how="left").fill_null(0)

# -----------------------------
# One-hot encode medications
# -----------------------------
med_df = df.filter(pl.col("DataType") == "Medications")
med_df = med_df.with_columns(
    pl.col("DataCategory").str.to_uppercase().str.replace(" ", "_").alias("med_clean")
)

medication_map = (
    med_df.group_by(["PatientID", "EventDate"])
    .agg(pl.col("med_clean").list())
)

all_meds = medication_map.select(pl.col("med_clean").list.unique().flatten()).to_series().to_list()
all_meds = [f"med_{c}" for c in all_meds]

med_df_onehot = (
    medication_map.with_columns(
        pl.lit(1).alias("val"),
        pl.col("med_clean").map_elements(lambda x: [f"med_{c}" for c in x], return_dtype=pl.List(pl.Utf8)).alias("med_names")
    )
    .pivot(values="val", columns="med_names", index=["PatientID", "EventDate"], aggregate_function="first")
    .fill_null(0)
)
base_df = base_df.join(med_df_onehot, on=["PatientID", "EventDate"], how="left").fill_null(0)

# -----------------------------
# Pivot-style lab expansion
# -----------------------------
lab_df = df.filter((pl.col("DataType") == "Labs") & pl.col("DataNumeric").is_not_null())
lab_df = lab_df.with_columns(pl.col("DataCategory").str.to_uppercase())

lab_pivot = (
    lab_df.group_by(["PatientID", "EventDate", "DataCategory"])
    .agg(pl.col("DataNumeric").first())
    .pivot(values="DataNumeric", columns="DataCategory", index=["PatientID", "EventDate"], aggregate_function="first")
    .pipe(lambda df_p: df_p.rename({col: f"lab_{col}" for col in df_p.columns if col not in ["PatientID", "EventDate"]}))
)

base_df = base_df.join(lab_pivot, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Optional: One-hot encode demographics
# -----------------------------
def format_demographics(s):
    s = str(s).replace("//", " ").replace("/", " ")
    s = re.sub(r"Unknown Not Reported", "", s).strip()
    s = re.sub(r"Do not identify with Race", "unknown race", s).strip()
    return s

demo_df = df.filter(pl.col("DataType") == "Demographics").drop_nulls(subset=["DataCategory"])

if not demo_df.is_empty():
    demo_df = (
        demo_df.group_by("PatientID")
        .first()
        .with_columns(pl.col("DataCategory").map_elements(format_demographics).alias("demo_string"))
    )
    
    unique_demos = demo_df.select(pl.col("demo_string").unique()).to_series().to_list()
    
    demo_df = demo_df.with_columns(
        pl.lit(1).alias("val"),
        pl.col("demo_string").map_elements(lambda x: f"demo_{x}", return_dtype=pl.Utf8).alias("demo_names")
    )
    
    demo_onehot = demo_df.pivot(values="val", columns="demo_names", index=["PatientID"], aggregate_function="first").fill_null(0)
    
    base_df = base_df.join(demo_onehot, on="PatientID", how="left").fill_null(0)

logger.info("Testing stage completed")
# -----------------------------
# Final report
# -----------------------------
logger.info(f"Final tabular shape: {base_df.shape}")
logger.info(f"Sample features:\n{base_df.head()}")
logger.info(f"CKD stage counts:\n{base_df['CKD_stage'].value_counts(sort=True, drop_nulls=False)}")
base_df_path = os.path.join(output_dir_m, output_fname)
logger.info(f"Final DataFrame shape: {base_df.shape}")
base_df.write_csv(base_df_path, separator=',', has_header=True)

logger.info("End of Tabular Generation")