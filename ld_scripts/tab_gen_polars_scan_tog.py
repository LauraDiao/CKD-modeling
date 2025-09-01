# polars script with datatype toggles
# scan_csv
import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import os
import re
import logging
from tqdm import tqdm
from memory_profiler import memory_usage

# variables
output_path = "./../../../commonfilesharePHI/ldiao/ckd_project/"
custom_separator = True # <<
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
output_dir  += "_scan" # <<

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
except FileExistsError:
    print(f"Output directory already exists: {output_dir}")

print(f"Processing started. Output directory: {output_dir}")
print(f"Using DataNumeric data type: {'Float64' if use_float64 else 'Float32'}")
print(f"Using DataInteger data type: {'Int64' if use_int64 else 'Int16'}")

# -----------------------------
# Load and preprocess with Polars
# -----------------------------
print("--- Starting Data Loading and Preprocessing ---")
if custom_separator:
    df = pl.scan_csv(
        event_file,
        separator='$',
        infer_schema_length=None,
        null_values="null",
    ).unique()
else:
    df = pl.scan_csv(
        event_file,
        infer_schema_length=None,
        null_values="null",
    ).unique()

print(f"Initial DataFrame schema: {df.schema}")

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
print("--- Data Loading and Preprocessing Complete ---")

# -----------------------------
# Base: full patient-day index
# -----------------------------
print("--- Starting CKD Staging and Monotonic Rank Enforcement ---")
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

# Enforce monotonic CKD staging
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
print("--- CKD staging and monotonic enforcement complete ---")

# -----------------------------
# One-hot encode diagnoses, medications, and labs
# -----------------------------
print("--- Starting One-Hot Encoding and Lab Expansion ---")
def truncate_icd(code):
    code = str(code).strip().replace(" ", "")
    if '.' in code:
        prefix, suffix = code.split('.', 1)
        return f"{prefix}.{suffix[0]}" if suffix else prefix
    return code

diag_df = df.filter(pl.col("DataType") == "Diagnosis").with_columns(
    pl.col("DataCategory").map_elements(truncate_icd, return_dtype=pl.Utf8).alias("ICD_clean")
).group_by("PatientID", "EventDate").agg(pl.col("ICD_clean").unique().sort().alias("ICD_list"))

diag_df_collect = diag_df.collect()
mlb_diag = MultiLabelBinarizer()
diag_features = mlb_diag.fit_transform(diag_df_collect["ICD_list"])
diag_onehot = pl.DataFrame(diag_features, schema=[f"diag_{c}" for c in mlb_diag.classes_]).cast(data_integer_dtype)
diag_df_onehot = pl.concat([diag_df_collect.select("PatientID", "EventDate"), diag_onehot], how="horizontal")
base_df = base_df.join(diag_df_onehot.lazy(), on=["PatientID", "EventDate"], how="left")
print("Diagnosis one-hot encoding complete.")

med_df = df.filter(pl.col("DataType") == "Medications").with_columns(
    pl.col("DataCategory").str.to_uppercase().str.replace(" ", "_").alias("med_clean")
).group_by("PatientID", "EventDate").agg(pl.col("med_clean").unique().sort().alias("med_list"))

med_df_collect = med_df.collect()
mlb_med = MultiLabelBinarizer()
med_features = mlb_med.fit_transform(med_df_collect["med_list"])
med_onehot = pl.DataFrame(med_features, schema=[f"med_{c}" for c in mlb_med.classes_]).cast(data_integer_dtype)
med_df_onehot = pl.concat([med_df_collect.select("PatientID", "EventDate"), med_onehot], how="horizontal")
base_df = base_df.join(med_df_onehot.lazy(), on=["PatientID", "EventDate"], how="left")
print("Medication one-hot encoding complete.")

lab_df = df.filter(
    (pl.col("DataType") == "Labs") & (pl.col("DataNumeric").is_not_null())
).with_columns(
    pl.col("DataCategory").cast(pl.Utf8, strict=False).str.to_uppercase().alias("LabCategory")
).group_by("PatientID", "EventDate", "LabCategory").agg(pl.col("DataNumeric").first())

lab_df_eager = lab_df.collect()
lab_pivot = lab_df_eager.pivot(
    index=["PatientID", "EventDate"],
    columns="LabCategory",
    values="DataNumeric",
    aggregate_function="first",
)
rename_dict = {c: f"lab_{c}" for c in lab_pivot.columns[2:]}
lab_pivot_renamed = lab_pivot.rename(rename_dict).lazy()
base_df = base_df.join(lab_pivot_renamed, on=["PatientID", "EventDate"], how="left")
print("Lab pivot expansion complete.")
print("--- One-Hot Encoding and Lab Expansion Complete ---")

# -----------------------------
# One-hot encode demographics and final output
# -----------------------------
print("--- Starting Demographics Encoding and Final Output ---")
def format_demographics(demo_string):
    demo_string = str(demo_string).replace("//", " ").replace("/", " ")
    if "Unknown Not Reported" in demo_string:
        demo_string = demo_string.replace("Unknown Not Reported", "").strip()
    if "Do not identify with Race" in demo_string:
        demo_string = demo_string.replace("Do not identify with Race", "unknown race").strip()
    return demo_string

demo_df = df.filter((pl.col("DataType") == "Demographics") & pl.col("DataCategory").is_not_null())\
            .group_by("PatientID").first().select(["PatientID", "DataCategory"])

demo_df_collect = demo_df.collect()

if not demo_df_collect.is_empty():
    demo_df_collect = demo_df_collect.with_columns(
        pl.col("DataCategory").map_elements(format_demographics).alias("demo_string")
    )
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    demo_encoded = enc.fit_transform(demo_df_collect.select("demo_string").to_numpy())
    demo_onehot = pl.DataFrame(demo_encoded, schema=[f"demo_{c}" for c in enc.categories_[0]]).cast(data_integer_dtype)
    demo_df = pl.concat([demo_df_collect.select("PatientID"), demo_onehot], how="horizontal")
else:
    all_demo_categories = df.filter((pl.col("DataType") == "Demographics") & pl.col("DataCategory").is_not_null())\
                            .select(pl.col("DataCategory").map_elements(format_demographics).unique()).collect().to_series().to_list()
    if not all_demo_categories:
        all_demo_categories = ["unknown race"]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(pl.Series(all_demo_categories).to_numpy().reshape(-1, 1))
    empty_demo_onehot = pl.DataFrame(columns=[f"demo_{c}" for c in enc.categories_[0]]).cast(data_integer_dtype)
    demo_df = pl.DataFrame({"PatientID": []}).with_columns(pl.col("PatientID").cast(pl.Utf8)).hstack(empty_demo_onehot)

base_df = base_df.join(demo_df.lazy(), on="PatientID", how="left")
print("Demographics one-hot encoding complete.")

# -----------------------------
# Final report
# -----------------------------
base_df_final = base_df.collect()

print(f"[INFO] Final tabular shape: {base_df_final.shape}")
print(f"[INFO] Sample features:\n{base_df_final.head()}")
print(f"[INFO] CKD stage counts:\n{base_df_final['CKD_stage'].value_counts(sort=True)}")
base_df_path = os.path.join(output_dir, output_fname)
print(f"Writing final DataFrame of shape {base_df_final.shape} to {base_df_path}")
base_df_final.write_csv(base_df_path)
print("End of Tabular Generation")
print("--- Demographics Encoding and Final Output Complete ---")