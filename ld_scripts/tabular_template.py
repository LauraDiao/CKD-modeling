# %%
#  tab_gen_polars_read_ftsplit.py
# speed optimizations by converting the custom Python staging functions (map_elements) 
# to vectorized Polars expressions (when/then/otherwise) and by 
# using Polars' efficient iter_slices for chunking the one-hot encoding process.
# modified fit_transform process (icd codes, etc)
# polars script with datatype toggles
# read_csv

"""
Plan:
- one-hot encode tabular data by encounter
- RNN: on progression

"""

import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import os
import re
import logging
from tqdm import tqdm
import numpy as np # Need to import numpy for clean_ckd_stage's use of np.nan
# %%
# variables
output_fname = "ckd_processed_tab.csv"
subset_size = "10"  # 10, 100, full # <<
output_version_suffix = "_v7" # << for versioning output folders

# --- TOGGLES ---
custom_separator = True     # << If True, uses '$' separator and full path, else uses default separator and subset path.
use_float64 = False         # << True to use Float64, False for Float32
use_int64 = False           # << True to use Int64, False for Int16
use_gfr = False             # << If True, uses GFR for CKD staging, else uses ICD codes.
filter_ckd_stage = True     # << If True, filters patients who never reach CKD stage 3 or higher.

# --- Path and Data Type Initialization ---
if not custom_separator: 
    output_dir = f"/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_tab_{subset_size}"
    # event_file = f"/opt/data/commonfilesharePHI/slee/ckd-optum/patients_subset_{subset_size}.csv"
    event_file = f"/opt/data/workingdir/ldiao/ckd_project/patient_subsets/patients_subset_{subset_size}.csv"
if custom_separator:
    output_dir = "/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_tab_full"
    # event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt"
    event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt.parquet"

# %%
df = pl.read_parquet(event_file)

# %%
dx_df = (
    df
    .filter((pl.col("DataType") == "Diagnosis") 
        & (pl.col("DataCategory").str.contains(r"^[A-Za-z]")))
    .with_columns(
        pl.col("DataCategory").str.head(5).alias("TruncatedICD")
        , pl.lit(1).alias("Value")
    )
    .drop(["META_2", "EventTimeStamp"])
    .unique()
    .group_by(["PatientID", "META_1"])
    .agg(pl.col("TruncatedICD").alias("icd_codes"))
)
# %% too large
# pdx_df = dx_df.pivot(
#     on='TruncatedICD'
#     , index=['PatientID', 'META_1']
#     , values='Value'
#     , aggregate_function='max')
# %%
meds_df = (
    df
    .filter((pl.col("DataType") == "Medications"))
    .drop(["EventTimeStamp"])
    .unique()
    .group_by(["PatientID", "META_1"])
    .agg(pl.col("DataCategory").alias("med_list"))
)
# %%
demographics_df = (
    df
    .filter(pl.col("DataType") == "Demographics")
    .unique()
)
# %%
encounters_df = (
    df
    .filter(pl.col("DataType") == "Encounter")
    .unique()
)

# %%
labs_df = (
    df
    .filter(
        (pl.col("DataType") == "Labs")
        & pl.col("DataNumeric").is_not_null()
    )
    .with_columns(
        pl.col("DataCategory").fill_null("Sodium") # sodium is NA... 
    )
    .sort(["PatientID", "META_1", "DataCategory", "EventTimeStamp"])
    .group_by(["PatientID", "META_1", "DataCategory"])
    .agg(
        pl.col("DataNumeric")
        .cast(pl.Float64, strict=False)
        .drop_nulls()
        .last()
        .alias("ResultValue")
    )
    .pivot(on='DataCategory', index=["PatientID", "META_1"], values="ResultValue", aggregate_function='mean')
)
# %%




# %%
# --- data types based on toggles ---
data_numeric_dtype = pl.Float64 if use_float64 else pl.Float32
data_integer_dtype = pl.Int64 if use_int64 else pl.Int16

# --- build output directory name ---
output_dir  += f"_{'f64' if use_float64 else 'f32'}"
output_dir += f"_{'i64' if use_int64 else 'i16'}"
output_dir  += "_read" # << scan vs read csv

# icd vs gfr to ckd stage
if use_gfr:
    output_dir += "_gfr"
if not use_gfr:
    output_dir += "_icd"

# filter ckd stage
if filter_ckd_stage: 
    output_dir  += "_stage_filter" 

output_dir  += output_version_suffix 

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
except FileExistsError:
    print(f"Output directory already exists: {output_dir}")

print(f"Processing started. Output directory: {output_dir}")
print(f"Using DataNumeric data type: {'Float64' if use_float64 else 'Float32'}")
print(f"Using DataInteger data type: {'Int64' if use_int64 else 'Int16'}")

# Setup logging
log_file_path = os.path.join(output_dir, "tab_gen_m.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Processing started. Output directory: {output_dir}")

# -----------------------------
# Load and preprocess with Polars
# -----------------------------
# Using pl.read_csv to load the entire file into a DataFrame
# Add a toggle to switch between separators

if custom_separator:
    df = pl.read_csv(
        event_file,
        separator='$',
        infer_schema_length=None,
        null_values="null",
    ).unique()
else:
    df = pl.read_csv(
        event_file,
        infer_schema_length=None,
        null_values="null",
    ).unique()

logger.info(f"Initial DataFrame schema: {df.schema}")

df = df.with_columns(
    pl.col("PatientID").cast(pl.Utf8, strict=False),
    pl.col("EventTimeStamp").cast(pl.Utf8, strict=False),
    pl.col("DataCategory").cast(pl.Utf8, strict=False),
    pl.col("DataNumeric").cast(data_numeric_dtype, strict=False),
    pl.col("DataType").cast(pl.Utf8, strict=False),
)

# Clean and prepare initial columns
df = df.with_columns(
    pl.col("EventTimeStamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False).alias("EventTimeStamp"),
    pl.col("DataCategory").fill_null("None"),
).with_columns(
    pl.col("EventTimeStamp").dt.date().alias("EventDate")
)

# -----------------------------
# Base: full patient-day index
# -----------------------------
all_days = df.select(["PatientID", "EventDate"]).unique().sort(["PatientID", "EventDate"])
all_days = all_days.drop_nulls("EventDate")

# -----------------------------
# Extract and forward-fill GFR
# -----------------------------
if use_gfr: 
    gfr_df = df.filter(
        (pl.col("DataCategory").str.contains("(?i)GFR|GFREST")) &
        (pl.col("DataNumeric").is_not_null())
    ).select("PatientID", "EventDate", "DataNumeric")

    gfr_daywise = gfr_df.group_by("PatientID", "EventDate").first().rename({"DataNumeric": "GFR_combined"})

    base_df = all_days.join(gfr_daywise, on=["PatientID", "EventDate"], how="left").sort(["PatientID", "EventDate"])


    # Forward-fill GFR
    base_df = base_df.with_columns(
        pl.col("GFR_combined").forward_fill().over("PatientID")
    )

    # Replaced Python functions with vectorized Polars expressions for speed
    base_df = base_df.with_columns(
        pl.when(pl.col("GFR_combined") >= 90).then(pl.lit("1"))
        .when(pl.col("GFR_combined") >= 60).then(pl.lit("2"))
        .when(pl.col("GFR_combined") >= 45).then(pl.lit("3a"))
        .when(pl.col("GFR_combined") >= 30).then(pl.lit("3b"))
        .when(pl.col("GFR_combined") >= 15).then(pl.lit("4"))
        .when(pl.col("GFR_combined").is_null()).then(pl.lit(None).cast(pl.Utf8))
        .otherwise(pl.lit("5")).alias("CKD_stage"),
        
        pl.when(pl.col("GFR_combined") >= 90).then(1)
        .when(pl.col("GFR_combined") >= 60).then(2)
        .when(pl.col("GFR_combined") >= 45).then(3.1)
        .when(pl.col("GFR_combined") >= 30).then(3.2)
        .when(pl.col("GFR_combined") >= 15).then(4)
        .when(pl.col("GFR_combined").is_null()).then(0)
        .otherwise(5).alias("CKD_rank")
    ).with_columns(
        pl.col("CKD_rank").cast(data_numeric_dtype)
    )

    # Enforce monotonic CKD staging using a cumulative maximum, a faster and more robust method
    base_df = base_df.with_columns(
        pl.col("CKD_rank").fill_null(0).cum_max().over("PatientID").alias("CKD_rank_monotonic")
    ).with_columns(
        pl.when(pl.col("CKD_rank_monotonic") == 1).then(pl.lit("1"))
        .when(pl.col("CKD_rank_monotonic") == 2).then(pl.lit("2"))
        .when(pl.col("CKD_rank_monotonic") == 3.1).then(pl.lit("3a"))
        .when(pl.col("CKD_rank_monotonic") == 3.2).then(pl.lit("3b"))
        .when(pl.col("CKD_rank_monotonic") == 4).then(pl.lit("4"))
        .when(pl.col("CKD_rank_monotonic") == 5).then(pl.lit("5"))
        .otherwise(pl.lit(None)).alias("CKD_stage")
    ).drop("CKD_rank", "CKD_rank_monotonic")

# -----------------------------
# Extract and forward-fill ICD
# -----------------------------
if not use_gfr: 
    icd_filter = "(?i)^N18\..*" 
    # all icd codes: (?i)\."
    icd_df = df.filter(
        (pl.col("DataCategory").str.contains(icd_filter)) &
        (pl.col("DataNumeric").is_not_null())
    ).select("PatientID", "EventDate", "DataCategory")

    icd_daywise = icd_df.group_by("PatientID", "EventDate").first().rename({"DataCategory": "ICD_combined"})
    base_df = all_days.join(icd_daywise, on=["PatientID", "EventDate"], how="left").sort(["PatientID", "EventDate"])
    # base_df.head()

    # Forward-fill ICD
    base_df = base_df.with_columns(
        pl.col("ICD_combined").forward_fill().over("PatientID")
    )
    # base_df.head()

    # Replaced Python functions with vectorized Polars expressions for speed
    base_df = base_df.with_columns(
        pl.when(pl.col("ICD_combined").str.contains('N18.1')).then(pl.lit("1"))
        .when(pl.col("ICD_combined").str.contains('N18.2')).then(pl.lit("2"))
        .when(pl.col("ICD_combined").str.contains('N18.3')).then(pl.lit("3"))
        .when(pl.col("ICD_combined").str.contains('N18.4')).then(pl.lit("4"))
        .when(pl.col("ICD_combined").is_null()).then(pl.lit(None).cast(pl.Utf8))
        .otherwise(pl.lit("5")).alias("CKD_stage"),

        pl.when(pl.col("ICD_combined").str.contains('N18.1')).then(1)
        .when(pl.col("ICD_combined").str.contains('N18.2')).then(2)
        .when(pl.col("ICD_combined").str.contains('N18.3')).then(3)
        .when(pl.col("ICD_combined").str.contains('N18.4')).then(4)
        .when(pl.col("ICD_combined").is_null()).then(0)
        .otherwise(5).alias("CKD_rank")
    ).with_columns(
        pl.col("CKD_rank").cast(data_numeric_dtype)
    )

    # Enforce monotonic CKD staging using a cumulative maximum, a faster and more robust method
    base_df = base_df.with_columns(
        pl.col("CKD_rank").fill_null(0).cum_max().over("PatientID").alias("CKD_rank_monotonic")
    ).with_columns(
        pl.when(pl.col("CKD_rank_monotonic") == 1).then(pl.lit("1"))
        .when(pl.col("CKD_rank_monotonic") == 2).then(pl.lit("2"))
        .when(pl.col("CKD_rank_monotonic") == 3.1).then(pl.lit("3a"))
        .when(pl.col("CKD_rank_monotonic") == 3.2).then(pl.lit("3b"))
        .when(pl.col("CKD_rank_monotonic") == 4).then(pl.lit("4"))
        .when(pl.col("CKD_rank_monotonic") == 5).then(pl.lit("5"))
        .otherwise(pl.lit(None)).alias("CKD_stage")
    ).drop("CKD_rank", "CKD_rank_monotonic")

# -----------------------------
# clean and filter ckd stage
# -----------------------------
# Python function 'clean_ckd_stage' replaced with vectorized Polars expressions below

def filter_patients_by_ckd_stage(df, ckd_stage_col, patient_id_col='PatientID'):
    initial_patients = df[patient_id_col].nunique()
    # Filter for visits where CKD stage is 3 or higher
    df_at_or_above_stage_3 = df[df[ckd_stage_col] >= 3]
    # Get unique PatientIDs from this filtered DataFrame
    patient_ids_to_keep = set(df_at_or_above_stage_3[patient_id_col].unique())
    
    patients_removed = initial_patients - len(patient_ids_to_keep)
    logger.info(f"Identified {len(patient_ids_to_keep)} patients with at least one visit at or above CKD stage 3.")
    logger.info(f"Filtered out approximately {patients_removed} patients who are always below CKD stage 3.")
    
    return patient_ids_to_keep

def process_ckd_stage(df: pl.DataFrame, ckd_column: str, patient_id_col: str = 'PatientID', filtering_stage= filter_ckd_stage):
    if ckd_column not in df.columns:
        logger.error(f"'{ckd_column}' column not found in tabular data. Cannot proceed with label generation.")
        return None

    clean_col_name = f'{ckd_column}_clean'

    # Vectorized cleaning logic to replace the Python function:
    base_df = df.with_columns(
        # 1. Try to parse as Int: if successful, keep the value (e.g., '1', '2')
        pl.col(ckd_column).cast(pl.Int64, strict=False).alias(clean_col_name)
    ).with_columns(
        # 2. Handle '3a', '3b' which failed casting to Int64: set them to 3
        pl.when(pl.col(clean_col_name).is_null() & pl.col(ckd_column).str.contains('3a|3b')).then(pl.lit(3).cast(pl.Int64))
        # 3. For other non-numeric strings, try taking the first digit (e.g. from '1st')
        .when(pl.col(clean_col_name).is_null() & pl.col(ckd_column).str.slice(0, 1).str.contains(r"^\d$")).then(pl.col(ckd_column).str.slice(0, 1).cast(pl.Int64))
        # 4. Keep existing values
        .otherwise(pl.col(clean_col_name))
        .alias(clean_col_name)
    )

    # Use Polars' window functions to backfill and forward fill nulls within each patient group.
    base_df = base_df.with_columns(
        pl.col(clean_col_name).forward_fill().over(patient_id_col).alias(clean_col_name) # FF first for more typical usage
    ).with_columns(
        pl.col(clean_col_name).backward_fill().over(patient_id_col).alias(clean_col_name) # Then BF
    )
    
    # Remove patients with no stage info after the fill operations.
    base_df = base_df.drop_nulls(subset=[clean_col_name])
    # note: icd vs gfr - this will impact patient counts
    
    if filtering_stage:
        patients_to_keep = base_df.group_by(patient_id_col).agg(
            pl.col(clean_col_name).max().ge(3).alias("keep_patient")
        ).filter(pl.col("keep_patient")).select(patient_id_col)
        
        # Use an inner join to keep only the rows for the filtered patients.
        base_df = base_df.join(patients_to_keep, on=patient_id_col, how="inner")
        
        logger.info(f"Shape of base_df after filtering for patients at or above stage 3: {base_df.shape}")
    
    return base_df

ckd_column = "CKD_stage" # "CKD_stage"
base_df = process_ckd_stage(base_df, ckd_column)
base_df.head()

# -----------------------------
# One-hot encode diagnoses (truncated ICD codes)
# -----------------------------
def truncate_icd(code):
    code = str(code).strip().replace(" ", "")
    if '.' in code:
        prefix, suffix = code.split('.', 1)
        return f"{prefix}.{suffix[0]}" if suffix else prefix
    return code

diag_df = df.filter(pl.col("DataType") == "Diagnosis").with_columns(
    pl.col("DataCategory").map_elements(truncate_icd, return_dtype=pl.Utf8).alias("ICD_clean")
).group_by("PatientID", "EventDate").agg(pl.col("ICD_clean").unique().sort().alias("ICD_list"))


# --- Memory Optimization: Separate Fit and Chunked Transform for ICD ---
mlb_diag = MultiLabelBinarizer()
# 1. Fit: Use Polars to efficiently gather the complete, global vocabulary of ATOMIC labels
all_icd_codes = diag_df["ICD_list"].explode().unique().drop_nulls().to_list()
# Crucial: Wrap the list of codes in an outer list so MLB treats it as ONE sample with ALL possible labels
mlb_diag.fit([all_icd_codes]) 
logger.info(f"ICD MultiLabelBinarizer fitted with {len(mlb_diag.classes_)} unique classes.")

# 2. Transform: Process in chunks
chunk_size = 50000  # Define a manageable chunk size
diag_onehot_chunks = []

# MODIFIED: Use iter_slices for better chunk iteration with tqdm (Added Progress Bar)
for chunk_df in tqdm(diag_df.iter_slices(chunk_size), total=(diag_df.shape[0] + chunk_size - 1) // chunk_size, desc="One-Hot Encoding ICD Codes"):
    # Convert chunked ICD_list column to a Python list for sklearn
    chunk_icd_list = chunk_df["ICD_list"].to_list()

    # Transform the chunk (no warnings because the full vocabulary is known)
    chunk_features = mlb_diag.transform(chunk_icd_list)

    # Convert to Polars DataFrame, cast to the integer type
    chunk_onehot_pl = pl.DataFrame(
        chunk_features, 
        schema=[f"diag_{c}" for c in mlb_diag.classes_]
    ).cast(data_integer_dtype)

    diag_onehot_chunks.append(
        pl.concat([chunk_df.select("PatientID", "EventDate"), chunk_onehot_pl], how="horizontal")
    )

diag_df_onehot = pl.concat(diag_onehot_chunks)
# ---------------------------------------------------------------------

base_df = base_df.join(diag_df_onehot, on=["PatientID", "EventDate"], how="left")


# -----------------------------
# One-hot encode medications
# -----------------------------
med_df = df.filter(pl.col("DataType") == "Medications").with_columns(
    pl.col("DataCategory").str.to_uppercase().str.replace(" ", "_").alias("med_clean")
).group_by("PatientID", "EventDate").agg(pl.col("med_clean").unique().sort().alias("med_list"))

# --- Memory Optimization: Separate Fit and Chunked Transform for Meds ---
mlb_med = MultiLabelBinarizer()
# 1. Fit: Use Polars to efficiently gather the complete, global vocabulary of ATOMIC labels
all_med_codes = med_df["med_list"].explode().unique().drop_nulls().to_list()
# Crucial: Wrap the list of codes in an outer list so MLB treats it as ONE sample with ALL possible labels
mlb_med.fit([all_med_codes])
logger.info(f"Medication MultiLabelBinarizer fitted with {len(mlb_med.classes_)} unique classes.")

# 2. Transform: Process in chunks
# chunk_size is already defined
med_onehot_chunks = []

# MODIFIED: Use iter_slices for better chunk iteration with tqdm (Added Progress Bar)
for chunk_df in tqdm(med_df.iter_slices(chunk_size), total=(med_df.shape[0] + chunk_size - 1) // chunk_size, desc="One-Hot Encoding Medications"):
    # Convert chunked med_list column to a Python list for sklearn
    chunk_med_list = chunk_df["med_list"].to_list()

    # Transform the chunk (no warnings because the full vocabulary is known)
    chunk_features = mlb_med.transform(chunk_med_list)

    # Convert to Polars DataFrame, cast to the integer type
    chunk_onehot_pl = pl.DataFrame(
        chunk_features, 
        schema=[f"med_{c}" for c in mlb_med.classes_]
    ).cast(data_integer_dtype)
    
    med_onehot_chunks.append(
        pl.concat([chunk_df.select("PatientID", "EventDate"), chunk_onehot_pl], how="horizontal")
    )

med_df_onehot = pl.concat(med_onehot_chunks)
# ---------------------------------------------------------------------

base_df = base_df.join(med_df_onehot, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Pivot-style lab expansion
# -----------------------------
lab_df = df.filter(
    (pl.col("DataType") == "Labs") & (pl.col("DataNumeric").is_not_null())
).with_columns(
    pl.col("DataCategory").cast(pl.Utf8, strict=False).str.to_uppercase().alias("LabCategory")
).group_by("PatientID", "EventDate", "LabCategory").agg(pl.col("DataNumeric").first())

lab_pivot = lab_df.pivot(
    index=["PatientID", "EventDate"],
    on="LabCategory",
    values="DataNumeric",
    aggregate_function="first",
)

# Dynamically generate a dictionary for renaming the pivoted columns
rename_dict = {c: f"lab_{c}" for c in lab_pivot.columns[2:]}
lab_pivot = lab_pivot.rename(rename_dict)

base_df = base_df.join(lab_pivot, on=["PatientID", "EventDate"], how="left")


# -----------------------------
# Optional: One-hot encode demographics
# -----------------------------
def format_demographics(demo_string):
    demo_string = str(demo_string).replace("//", " ").replace("/", " ")
    if "Unknown Not Reported" in demo_string:
        demo_string = demo_string.replace("Unknown Not Reported", "").strip()
    if "Do not identify with Race" in demo_string:
        demo_string = demo_string.replace("Do not identify with Race", "unknown race").strip()
    return demo_string

# Filter and group the data
demo_df = df.filter((pl.col("DataType") == "Demographics") & pl.col("DataCategory").is_not_null())\
            .group_by("PatientID").first().select(["PatientID", "DataCategory"])


if not demo_df.is_empty():
    demo_df = demo_df.with_columns(
        pl.col("DataCategory").map_elements(format_demographics).alias("demo_string")
    )
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    demo_encoded = enc.fit_transform(demo_df.select("demo_string").to_numpy())
    demo_onehot = pl.DataFrame(demo_encoded, schema=[f"demo_{c}" for c in enc.categories_[0]]).cast(data_integer_dtype)
    demo_df = pl.concat([demo_df.select("PatientID"), demo_onehot], how="horizontal")
else:
    # Handle case with no demographics data by creating a dummy dataframe with the correct schema
    all_demo_categories = df.filter(pl.col("DataType") == "Demographics" & pl.col("DataCategory").is_not_null())\
                            .select(pl.col("DataCategory").map_elements(format_demographics).unique()).to_series().to_list()
    if not all_demo_categories:
        all_demo_categories = [""] # Ensure there is at least one category to fit the encoder
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(pl.Series(all_demo_categories).to_numpy().reshape(-1, 1))
    empty_demo_onehot = pl.DataFrame(enc.transform([[""]]), schema=[f"demo_{c}" for c in enc.categories_[0]]).cast(data_integer_dtype)
    demo_df = pl.DataFrame({"PatientID": [], "demo_string": []}).with_columns(
        pl.col("PatientID").cast(pl.Utf8)
    )
    demo_df = demo_df.hstack(empty_demo_onehot)

base_df = base_df.join(demo_df, on="PatientID", how="left")

# -----------------------------
# Final report
# -----------------------------
logger.info(f"[INFO] Final tabular shape: {base_df.shape}")
logger.info(f"[INFO] Sample features:\n{base_df.head()}")
logger.info(f"[INFO] CKD stage counts:\n{base_df['CKD_stage'].value_counts(sort=True)}")
base_df_path = os.path.join(output_dir, output_fname)
logger.info(f"Writing final DataFrame of shape {base_df.shape} to {base_df_path}")
base_df.write_csv(base_df_path)

logger.info("End of Tabular Generation")

# check csv
# Construct the full file path
final_file_path = os.path.join(output_dir, output_fname)

# Read the processed CSV file
try:
    final_df = pl.read_csv(
        final_file_path,
        schema_overrides={"CKD_stage": pl.Utf8}
    )
    print("File read successfully.")
    print(final_df.head())
except Exception as e:
    print(f"An error occurred while reading the file: {e}")