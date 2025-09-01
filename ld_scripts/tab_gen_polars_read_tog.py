# polars script with datatype toggles
# read_csv
import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import os
import re
import logging
from tqdm import tqdm

# variables
output_path = "./../../../commonfilesharePHI/ldiao/ckd_project/"
custom_separator = False # <<
if not custom_separator: 
    subset_size = "10"  # 10, 100, full # <<
    output_dir = output_path + f"ckd_tab_{subset_size}"
    event_file =  f"./../../../commonfilesharePHI/slee/ckd-optum/patients_subset_{subset_size}.csv"
if custom_separator:
    output_dir = output_path + "ckd_tab_full"
    event_file = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/CKD-Pull_v2.rpt"
output_fname = "ckd_processed_tab.csv"

# --- data type toggles ---
use_float64 = False # True to use Float64, False for other type
use_int64 = False # True to use Int64, False for other type
# --- data types based on toggles ---
data_numeric_dtype = pl.Float64 if use_float64 else pl.Float32
data_integer_dtype = pl.Int64 if use_int64 else pl.Int16
# --- add type suffixes to output directory ---
output_dir  += f"_{'f64' if use_float64 else 'f32'}"
output_dir += f"_{'i64' if use_int64 else 'i16'}"

# scan vs read csv
output_dir  += "_read" # <<

# filter ckd stage
filter_ckd_stage = False
if filter_ckd_stage: 
    output_dir  += "stage_filter" # ""

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

def gfr_to_stage(gfr):
    if gfr >= 90:
        return "1"
    elif gfr >= 60:
        return "2"
    elif gfr >= 45:
        return "3a"
    elif gfr >= 30:
        return "3b"
    elif gfr >= 15:
        return "4"
    elif gfr is None:
        return None
    else:
        return "5"

def gfr_to_rank(gfr):
    if gfr >= 90:
        return 1
    elif gfr >= 60:
        return 2
    elif gfr >= 45:
        return 3.1
    elif gfr >= 30:
        return 3.2
    elif gfr >= 15:
        return 4
    elif gfr is None:
        return 0
    else:
        return 5

base_df = base_df.with_columns(
    pl.col("GFR_combined").map_elements(gfr_to_stage, return_dtype=pl.Utf8).alias("CKD_stage"),
    pl.col("GFR_combined").map_elements(gfr_to_rank, return_dtype=data_numeric_dtype).alias("CKD_rank")
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

# clean and filter ckd stage
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

if 'CKD_stage' in base_df.columns:
    base_df['CKD_stage_clean'] = base_df['CKD_stage'].apply(clean_ckd_stage)
    # Fill missing stages within a patient's record
    base_df['CKD_stage_clean'] = base_df.groupby('PatientID')['CKD_stage_clean'].bfill().ffill()
    base_df = base_df.dropna(subset=['CKD_stage_clean']) # Remove patients with no stage info
    base_df['CKD_stage_clean'] = base_df['CKD_stage_clean'].astype(int)
else:
    logger.error("'CKD_stage' column not found in tabular data. Cannot proceed with label generation.")

if filtering_stage: 
    base_df_patients = filter_patients_by_ckd_stage(base_df, 'CKD_stage_clean')
    base_df = base_df[base_df["PatientID"].isin(base_df_patients)].copy()
    logger.info(f"Shape of base_df after filtering for patients at or above stage 3: {base_df.shape}")

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

mlb_diag = MultiLabelBinarizer()
diag_features = mlb_diag.fit_transform(diag_df["ICD_list"])
diag_onehot = pl.DataFrame(diag_features.astype(data_integer_dtype), schema=[f"diag_{c}" for c in mlb_diag.classes_])
diag_df_onehot = pl.concat([diag_df.select("PatientID", "EventDate"), diag_onehot], how="horizontal")

base_df = base_df.join(diag_df_onehot, on=["PatientID", "EventDate"], how="left")


# -----------------------------
# One-hot encode medications
# -----------------------------
med_df = df.filter(pl.col("DataType") == "Medications").with_columns(
    pl.col("DataCategory").str.to_uppercase().str.replace(" ", "_").alias("med_clean")
).group_by("PatientID", "EventDate").agg(pl.col("med_clean").unique().sort().alias("med_list"))

mlb_med = MultiLabelBinarizer()
med_features = mlb_med.fit_transform(med_df["med_list"])
med_onehot = pl.DataFrame(med_features.astype(data_integer_dtype), schema=[f"med_{c}" for c in mlb_med.classes_])
med_df_onehot = pl.concat([med_df.select("PatientID", "EventDate"), med_onehot], how="horizontal")

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
    columns="LabCategory",
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
    demo_onehot = pl.DataFrame(demo_encoded.astype(data_integer_dtype), schema=[f"demo_{c}" for c in enc.categories_[0]])
    demo_df = pl.concat([demo_df.select("PatientID"), demo_onehot], how="horizontal")
else:
    # Handle case with no demographics data by creating a dummy dataframe with the correct schema
    all_demo_categories = df.filter(pl.col("DataType") == "Demographics" & pl.col("DataCategory").is_not_null())\
                            .select(pl.col("DataCategory").map_elements(format_demographics).unique()).to_series().to_list()
    if not all_demo_categories:
        all_demo_categories = [""] # Ensure there is at least one category to fit the encoder
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(pl.Series(all_demo_categories).to_numpy().reshape(-1, 1))
    empty_demo_onehot = pl.DataFrame(enc.transform([[""]]).astype(data_integer_dtype), schema=[f"demo_{c}" for c in enc.categories_[0]])
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
    final_df = pl.read_csv(final_file_path)
    print("File read successfully.")
    print(final_df.head())
except Exception as e:
    print(f"An error occurred while reading the file: {e}")