# polars script with datatype toggles and batch processing
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
if custom_separator = False: 
    subset_size = "10"  # 10, 100, full # <<
    output_dir = output_path + f"ckd_tab_{subset_size}"
    event_file =  f"./../../../commonfilesharePHI/slee/ckd-optum/patients_subset_{subset_size}.csv"
if custom_separator:
    output_dir = output_path + "ckd_tab_full_batched"
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
output_dir  += "_scan_2" # <<
BATCH_SIZE = 100000  # Adjust based on your system's memory and performance

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
except FileExistsError:
    print(f"Output directory already exists: {output_dir}")

print(f"Processing started. Output directory: {output_dir}")
print(f"Using DataNumeric data type: {'Float64' if use_float64 else 'Float32'}")
print(f"Using PatientID data type: {'Int64' if use_int64 else 'i16'}")

# -----------------------------
# STEP 1: PRE-FIT ENCODERS ON UNIQUE CATEGORIES
# This is a crucial step to ensure consistent columns across all batches.
# We do a quick scan_csv to get all unique categories without loading the whole file.
# -----------------------------
print("--- Pre-fitting One-Hot Encoders ---")
lazy_df_full = pl.scan_csv(
    event_file,
    separator='$' if custom_separator else ',',
    infer_schema_length=None,
    null_values="null"
)
# Get unique diagnosis categories
all_diag_categories = lazy_df_full.filter(pl.col("DataType") == "Diagnosis").select(
    pl.col("DataCategory").str.replace_all(" ", "").str.split_exact(".", 2).struct.field("field_0")
    .str.replace_all(r"(\.)(\w)", "$1").alias("ICD_clean")
).unique().collect().to_series().to_list()

# Get unique medication categories
all_med_categories = lazy_df_full.filter(pl.col("DataType") == "Medications").select(
    pl.col("DataCategory").str.to_uppercase().str.replace(" ", "_").alias("med_clean")
).unique().collect().to_series().to_list()

# Get unique demographics categories
def format_demographics(demo_string):
    demo_string = str(demo_string).replace("//", " ").replace("/", " ")
    if "Unknown Not Reported" in demo_string:
        demo_string = demo_string.replace("Unknown Not Reported", "").strip()
    if "Do not identify with Race" in demo_string:
        demo_string = demo_string.replace("Do not identify with Race", "unknown race").strip()
    return demo_string

all_demo_categories_df = lazy_df_full.filter(pl.col("DataType") == "Demographics").select("DataCategory").unique().collect()
all_demo_categories = all_demo_categories_df.with_columns(
    pl.col("DataCategory").map_elements(format_demographics).alias("demo_string")
).unique().select("demo_string").to_series().to_list()

# Initialize and fit encoders with all unique categories
mlb_diag = MultiLabelBinarizer()
mlb_diag.fit([all_diag_categories])
mlb_med = MultiLabelBinarizer()
mlb_med.fit([all_med_categories])
ohe_demo = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_demo.fit(pl.Series(all_demo_categories).to_numpy().reshape(-1, 1))
print("--- Encoders pre-fitted on unique categories ---")

# -----------------------------
# STEP 2: BATCH PROCESSING LOOP
# -----------------------------
print("--- Starting batched data processing ---")
reader = pl.read_csv_batched(
    event_file,
    separator='$' if custom_separator else ',',
    infer_schema_length=None,
    null_values="null",
    batch_size=BATCH_SIZE
)

batch_counter = 0
final_output_path = os.path.join(output_dir, output_fname)
has_header_written = False

while True:
    try:
        batch_df = reader.next_batches(1)[0]
    except StopIteration:
        break

    batch_df = batch_df.unique().with_columns(
        pl.col("PatientID").cast(pl.Utf8, strict=False),
        pl.col("EventTimeStamp").cast(pl.Utf8, strict=False),
        pl.col("DataCategory").cast(pl.Utf8, strict=False),
        pl.col("DataNumeric").cast(data_numeric_dtype, strict=False),
        pl.col("DataType").cast(pl.Utf8, strict=False),
    )

    batch_df = batch_df.with_columns(
        pl.col("EventTimeStamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False).alias("EventTimeStamp"),
        pl.col("DataCategory").fill_null("None"),
    ).with_columns(
        pl.col("EventTimeStamp").dt.date().alias("EventDate")
    )

    # Base: full patient-day index for this batch
    all_days = batch_df.select(["PatientID", "EventDate"]).unique().sort(["PatientID", "EventDate"])
    all_days = all_days.drop_nulls("EventDate")

    # Extract and forward-fill GFR
    gfr_df = batch_df.filter(
        (pl.col("DataCategory").str.contains("(?i)GFR|GFREST")) &
        (pl.col("DataNumeric").is_not_null())
    ).select("PatientID", "EventDate", "DataNumeric")

    gfr_daywise = gfr_df.group_by("PatientID", "EventDate").first().rename({"DataNumeric": "GFR_combined"})
    base_df = all_days.join(gfr_daywise, on=["PatientID", "EventDate"], how="left").sort(["PatientID", "EventDate"])
    base_df = base_df.with_columns(pl.col("GFR_combined").forward_fill().over("PatientID"))

    # Optimized CKD staging
    base_df = base_df.with_columns(
        pl.when(pl.col("GFR_combined") >= 90).then(pl.lit("1"))
        .when(pl.col("GFR_combined") >= 60).then(pl.lit("2"))
        .when(pl.col("GFR_combined") >= 45).then(pl.lit("3a"))
        .when(pl.col("GFR_combined") >= 30).then(pl.lit("3b"))
        .when(pl.col("GFR_combined") >= 15).then(pl.lit("4"))
        .when(pl.col("GFR_combined").is_null()).then(pl.lit(None))
        .otherwise(pl.lit("5")).alias("CKD_stage"),
        pl.when(pl.col("GFR_combined") >= 90).then(1)
        .when(pl.col("GFR_combined") >= 60).then(2)
        .when(pl.col("GFR_combined") >= 45).then(3.1)
        .when(pl.col("GFR_combined") >= 30).then(3.2)
        .when(pl.col("GFR_combined") >= 15).then(4)
        .when(pl.col("GFR_combined").is_null()).then(0)
        .otherwise(5).alias("CKD_rank")
    )
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

    # One-hot encode diagnoses, medications, and labs
    diag_df_batch = batch_df.filter(pl.col("DataType") == "Diagnosis").with_columns(
        pl.col("DataCategory").str.replace_all(" ", "").str.split_exact(".", 2).struct.field("field_0")
        .str.replace_all(r"(\.)(\w)", "$1").alias("ICD_clean")
    ).group_by("PatientID", "EventDate").agg(pl.col("ICD_clean").unique().sort().alias("ICD_list"))
    diag_features = mlb_diag.transform(diag_df_batch["ICD_list"].to_list())
    diag_onehot = pl.DataFrame(diag_features, schema=[f"diag_{c}" for c in mlb_diag.classes_]).cast(data_integer_dtype)
    diag_df_onehot = pl.concat([diag_df_batch.select("PatientID", "EventDate"), diag_onehot], how="horizontal")
    base_df = base_df.join(diag_df_onehot, on=["PatientID", "EventDate"], how="left")

    med_df_batch = batch_df.filter(pl.col("DataType") == "Medications").with_columns(
        pl.col("DataCategory").str.to_uppercase().str.replace(" ", "_").alias("med_clean")
    ).group_by("PatientID", "EventDate").agg(pl.col("med_clean").unique().sort().alias("med_list"))
    med_features = mlb_med.transform(med_df_batch["med_list"].to_list())
    med_onehot = pl.DataFrame(med_features, schema=[f"med_{c}" for c in mlb_med.classes_]).cast(data_integer_dtype)
    med_df_onehot = pl.concat([med_df_batch.select("PatientID", "EventDate"), med_onehot], how="horizontal")
    base_df = base_df.join(med_df_onehot, on=["PatientID", "EventDate"], how="left")

    lab_df_batch = batch_df.filter(
        (pl.col("DataType") == "Labs") & (pl.col("DataNumeric").is_not_null())
    ).with_columns(
        pl.col("DataCategory").cast(pl.Utf8, strict=False).str.to_uppercase().alias("LabCategory")
    ).group_by("PatientID", "EventDate", "LabCategory").agg(pl.col("DataNumeric").first())
    lab_pivot = lab_df_batch.pivot(
        index=["PatientID", "EventDate"],
        columns="LabCategory",
        values="DataNumeric",
        aggregate_function="first",
    )
    rename_dict = {c: f"lab_{c}" for c in lab_pivot.columns[2:]}
    lab_pivot_renamed = lab_pivot.rename(rename_dict)
    base_df = base_df.join(lab_pivot_renamed, on=["PatientID", "EventDate"], how="left")

    # One-hot encode demographics
    demo_df_batch = batch_df.filter((pl.col("DataType") == "Demographics") & pl.col("DataCategory").is_not_null()) \
                            .group_by("PatientID").first().select(["PatientID", "DataCategory"])
    if not demo_df_batch.is_empty():
        demo_df_batch = demo_df_batch.with_columns(
            pl.col("DataCategory").map_elements(format_demographics).alias("demo_string")
        )
        demo_encoded = ohe_demo.transform(demo_df_batch.select("demo_string").to_numpy())
        demo_onehot = pl.DataFrame(demo_encoded, schema=[f"demo_{c}" for c in ohe_demo.categories_[0]]).cast(data_integer_dtype)
        demo_df_final = pl.concat([demo_df_batch.select("PatientID"), demo_onehot], how="horizontal")
        base_df = base_df.join(demo_df_final, on="PatientID", how="left")
    
    # Write batch to file
    base_df.write_csv(
        final_output_path,
        file_mode="a" if has_header_written else "w",
        include_header=not has_header_written
    )
    has_header_written = True
    batch_counter += 1
    print(f"Processed and wrote batch {batch_counter}. Shape: {base_df.shape}")

print("--- Batched processing complete. ---")