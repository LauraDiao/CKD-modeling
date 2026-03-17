# %%
# polars script with datatype toggles
# read_csv
import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import os
import re
import logging
from tqdm import tqdm
 

# %%
# variables
output_fname = "processed_tab_eskd.csv"
subset = False # <<

if subset: 
    subset_size = "100"  # 10, 100, full # <<
    output_dir = f"/opt/data/workingdir/ldiao/ckd_project/tabular_subset_{subset_size}"
    # output_dir = f"/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_tab_subset_{subset_size}"
    event_file =  f"/opt/data/workingdir/ldiao/ckd_project/tabular_subset_{subset_size}/unprocessed_tab_subset_{subset_size}.csv"
if not subset:
    # output_dir = "/opt/data/commonfilesharePHI/ldiao/ckd_project/ckd_tab_full"
    output_dir = f"/opt/data/workingdir/ldiao/ckd_project/tabular_full"
    event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt"


# %%
event_file

# %%

# --- data type toggles ---
use_float64 = False # True to use Float64, False for other type
use_int64 = False # True to use Int64, False for other type
# --- data types based on toggles ---
data_numeric_dtype = pl.Float64 if use_float64 else pl.Float32
data_integer_dtype = pl.Int64 if use_int64 else pl.Int16
# --- add type suffixes to output directory ---
# output_dir  += f"_{'f64' if use_float64 else 'f32'}"
# output_dir += f"_{'i64' if use_int64 else 'i16'}"

# scan vs read csv
# output_dir  += "_read" # <<

# filter ckd stage
filter_ckd_stage = True # <<
# if filter_ckd_stage: 
#     output_dir  += "_stage_filter" 

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
except FileExistsError:
    print(f"Output directory already exists: {output_dir}")

print(f"Processing started. Output directory: {output_dir}")
print(f"Using DataNumeric data type: {'Float64' if use_float64 else 'Float32'}")
print(f"Using DataInteger data type: {'Int64' if use_int64 else 'Int16'}")


# %%
print(output_dir)

# %%
# Setup logging
log_file_path = os.path.join(output_dir, "tab_gen_m.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Processing started. Output directory: {output_dir}")


# %%
# -----------------------------
# Load and preprocess with Polars
# -----------------------------
# Using pl.read_csv to load the entire file into a DataFrame
# Add a toggle to switch between separators
print(event_file)
df = pl.read_csv(
    event_file,
    separator='$',
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
print(df.shape)
# Clean and prepare initial columns
df = df.with_columns(
    pl.col("EventTimeStamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False).alias("EventTimeStamp"),
    pl.col("DataCategory").fill_null("None"),
).with_columns(
    pl.col("EventTimeStamp").dt.date().alias("EventDate")
)


# %%
df.shape

# %%
# -----------------------------
# Base: full patient-day index
# -----------------------------
all_days = df.select(["PatientID", "EventDate"]).unique().sort(["PatientID", "EventDate"])
all_days = all_days.drop_nulls("EventDate")

# -----------------------------
# Extract and forward-fill ICD
# -----------------------------

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
    if icd is None:
        return None
    if icd in 'N18.1%':
        return "1"
    if icd in 'N18.2%':
        return "2"
    if icd in 'N18.3%':
        return "3"
    if icd in 'N18.4%':
        return "4"
    return "5"

def icd_to_rank(icd):
    """
        'N18.1%': 1,
        'N18.2%': 2, 
        'N18.3%': 3, 
        'N18.4%': 4, 
        'N18.5%': 5, 
        'N18.6%': 'ESRD', 
        'N18.9%': 'CKD'
    """
    if icd is None:
        return 0
    if icd in 'N18.1%':
        return 1
    if icd in 'N18.2%':
        return 2
    if icd in 'N18.3%':
        return 3
    if icd in 'N18.4%':
        return 4
    return 5

base_df = base_df.with_columns(
    pl.col("ICD_combined").map_elements(icd_to_stage, return_dtype=pl.Utf8).alias("CKD_stage"),
    pl.col("ICD_combined").map_elements(icd_to_rank, return_dtype=data_numeric_dtype).alias("CKD_rank")
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


# %%
base_df

# %%

# -----------------------------
# clean and filter ckd stage
# -----------------------------
def clean_ckd_stage(value):
    try:
        # Handle cases like '3.1' or '3.2' if they are strings from CSV
        val_float = float(value)
        return int(val_float) # Truncate to integer stage
    except ValueError:
        if isinstance(value, str):
            if value.lower() == '3a': return 3
            if value.lower() == '3b': return 3 # Often grouped as stage 3
            if value[0].isdigit():
                return int(value[0])
        return np.nan
    except TypeError: # Handles if value is already NaN or None
        return np.nan

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

# %%

def process_ckd_stage(df: pl.DataFrame, ckd_column: str, patient_id_col: str = 'PatientID', filtering_stage= filter_ckd_stage):
    if ckd_column not in df.columns:
        logger.error(f"'{ckd_column}' column not found in tabular data. Cannot proceed with label generation.")
        return None

    # Define the name of the new cleaned column
    clean_col_name = f'{ckd_column}_clean'

    # Apply the element-wise cleaning function.
    # Note: map_elements can be less performant than vectorized operations,
    # but it is the closest equivalent to pandas.apply for a custom Python function.
    base_df = df.with_columns(
        pl.col(ckd_column).map_elements(clean_ckd_stage, return_dtype=pl.Int64).alias(clean_col_name)
    )

    # Use Polars' window functions to backfill and forward fill nulls within each patient group.
    base_df = base_df.with_columns(
        pl.col(clean_col_name).fill_null(strategy='backward').over(patient_id_col).alias(clean_col_name)
    ).with_columns(
        pl.col(clean_col_name).fill_null(strategy='forward').over(patient_id_col).alias(clean_col_name)
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

# %%
# -----------------------------
# One-hot encode diagnoses (truncated ICD codes)
# v1
# -----------------------------
def truncate_icd(code):
    code = str(code).strip().replace(" ", "")
    if '.' in code:
        prefix, suffix = code.split('.', 1)
        return f"{prefix}.{suffix[0]}" if suffix else prefix
    return code

aki_icd_codes = ["N17.0", "N17.1", "N17.2", "N17.8", "N17.9"]

# diag_df = df.filter((pl.col("DataType") == "Diagnosis") & 
#     (                pl.col("DataCategory").is_in(aki_icd_codes))
# ).with_columns(
#     pl.col("DataCategory").map_elements(truncate_icd, return_dtype=pl.Utf8).alias("ICD_clean")
# ).group_by("PatientID", "EventDate").agg(pl.col("ICD_clean").unique().sort().alias("ICD_list"))

# mlb_diag = MultiLabelBinarizer()
# diag_features = mlb_diag.fit_transform(diag_df["ICD_list"])
# diag_onehot = pl.DataFrame(diag_features, schema=[f"diag_{c}" for c in mlb_diag.classes_]).cast(data_integer_dtype)
# diag_df_onehot = pl.concat([diag_df.select("PatientID", "EventDate"), diag_onehot], how="horizontal")

# base_df = base_df.join(diag_df_onehot, on=["PatientID", "EventDate"], how="left")



# %%
aki_events = df.filter(
    (pl.col("DataType") == "Diagnosis") & 
    (pl.col("DataCategory").is_in(aki_icd_codes))
)

# 3. Sum (count) the occurrences into one column per Patient/Date
aki_count = aki_events.group_by(["PatientID", "EventDate"]).agg(
    pl.len().alias("AKI_ICD_Total")
)

# 4. Join this single column back to your base_df
base_df = base_df.join(aki_count, on=["PatientID", "EventDate"], how="left").with_columns(
    pl.col("AKI_ICD_Total").fill_null(0) # Ensure days with no AKI are 0, not null
)

# %%
base_df.head()

# %%
# -----------------------------
# One-hot encode diagnoses (truncated ICD codes)
# v2
# -----------------------------
#  Identify AKI events ---
# aki_events = df.filter(
#     pl.col("DataCategory").str.contains("(?i)AKI|ACUTE KIDNEY INJURY")
# ).group_by("PatientID", "EventDate").agg(
#     pl.len().alias("AKI_count")
# )

# base_df = base_df.join(aki_events, on=["PatientID", "EventDate"], how="left").with_columns(
#     pl.col("AKI_count").fill_null(0)
# )

# %%
# aki_events

# %%
# top lab features from the paper ---
top_lab_features = [
    "CREATININE", "GFR", "GFREST", "ALBUMIN/CREATININE RATIO", 
    "PROTEIN/CREATININE RATIO", "BUN", "PTH"
]

lab_df = df.filter(
    (pl.col("DataType") == "Labs") & 
    (pl.col("DataNumeric").is_not_null()) &
    (pl.col("DataCategory").cast(pl.Utf8).str.to_uppercase().str.contains("|".join(top_lab_features)))
).with_columns(
    pl.col("DataCategory").cast(pl.Utf8).str.to_uppercase().alias("LabCategory")
).group_by("PatientID", "EventDate", "LabCategory").agg(pl.col("DataNumeric").first())

# %%
lab_df

# %%
lab_pivot = lab_df.pivot(
    index=["PatientID", "EventDate"],
    on="LabCategory",
    values="DataNumeric",
    aggregate_function="first",
)

# %%
lab_pivot 

# %%
# Dynamically generate a dictionary for renaming the pivoted columns
rename_dict = {c: f"lab_{c}" for c in lab_pivot.columns[2:]}
lab_pivot = lab_pivot.rename(rename_dict)

base_df = base_df.join(lab_pivot, on=["PatientID", "EventDate"], how="left")



# %%
base_df 

# %%
# exclude demographics - not mentioned in eskd paper

# %%
print(base_df.columns)

# %%
base_df.head()

# %%

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
    final_df = pl.read_csv(final_file_path)
    print("File read successfully.")
    print(final_df.head())
except Exception as e:
    print(f"An error occurred while reading the file: {e}")

# %%



