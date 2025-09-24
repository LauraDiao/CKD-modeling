# importing 
from helper_functions import*
# functions shared across notebooks/py scripts

#-------------------------------------------------------------------------------
# Functions: icd functions - polars
#-----------------------------------------------------------------------
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



#-------------------------------------------------------------------------------
# Functions: icd functions - pandas
#-------------------------------------------------------------------------------

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
    if pd.isna(icd):
        return None, 0
    if icd in 'N18.1%':
        return "1", 1
    if icd in 'N18.2%':
        return "2", 2
    if icd in 'N18.3%':
        return "3", 3
    if icd in 'N18.4%':
        return "4", 4
    return "5", 5







#-------------------------------------------------------------------------------
# Functions: gfr functions - polars
#-------------------------------------------------------------------------------

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


#-------------------------------------------------------------------------------
# Functions: gfr functions - pandas
#-------------------------------------------------------------------------------


df['is_gfr'] = df['DataCategory'].str.upper().str.contains("GFR|GFREST", na=False)
all_days = df[['PatientID', 'EventDate']].drop_duplicates().sort_values(['PatientID', 'EventDate'])
# change
all_days = all_days.dropna()
# -----------------------------
# Extract and forward-fill GFR
# -----------------------------
gfr_df = df[df['is_gfr'] & df['DataNumeric'].notna()]
gfr_daywise = (
    gfr_df.groupby(['PatientID', 'EventDate'])['DataNumeric']
    .first().reset_index().rename(columns={'DataNumeric': 'GFR_combined'})
)

base_df = pd.merge(all_days, gfr_daywise, on=['PatientID', 'EventDate'], how='left')
base_df = base_df.sort_values(['PatientID', 'EventDate'])
base_df["GFR_combined"] = base_df.groupby("PatientID")["GFR_combined"].ffill()

def gfr_to_stage(gfr):
    if pd.isna(gfr): 
        return None, 0
    if gfr >= 90: 
        return "1", 1
    if gfr >= 60: 
        return "2", 2
    if gfr >= 45: 
        return "3a", 3.1
    if gfr >= 30: 
        return "3b", 3.2
    if gfr >= 15: 
        return "4", 4
    return "5", 5

# Enforce monotonic CKD staging
new_stages = {}
for pid, group in base_df.groupby("PatientID"):  # tqdm can be re-enabled here
    group = group.sort_values("EventDate")
    max_rank = 0
    prev_idx = None
    for idx, row in group.iterrows():
        stage, rank = gfr_to_stage(row["GFR_combined"])
        if rank < max_rank:
            stage = new_stages.get(prev_idx, stage)
        else:
            max_rank = rank
        new_stages[idx] = stage
        prev_idx = idx

base_df["CKD_stage"] = base_df.index.map(new_stages)


#-------------------------------------------------------------------------------
# Functions: ckd functions - polars
#-------------------------------------------------------------------------------
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


#-------------------------------------------------------------------------------
# Functions: ckd functions - pandas
#-------------------------------------------------------------------------------

def clean_ckd_stage(value):
    try:
        return int(value)
    except ValueError:
        if isinstance(value, str) and value[0].isdigit():
            return int(value[0])
        else:
            return np.nan
            
def filter_patients_by_ckd_stage(df, ckd_stage_col, patient_id_col='PatientID'):
    initial_patients = df[patient_id_col].nunique()
    # Filter for visits where CKD stage is 3 or higher
    df_at_or_above_stage_3 = df[df[ckd_stage_col] >= 3]
    # Get unique PatientIDs from this filtered DataFrame
    patient_ids_to_keep = set(df_at_or_above_stage_3[patient_id_col].unique())
    
    patients_removed = initial_patients - len(patient_ids_to_keep)

    return patient_ids_to_keep        

def find_CKD_stage_progression(df):
    df_sorted = df.sort_values(by=['PatientID', 'EventDate_dt'])
    
    # difference in CKD_stage for each patient
    df_sorted['stage_diff'] = df_sorted.groupby('PatientID')['CKD_stage_clean'].diff()

    # filter where the stage difference is positive (i.e., increased)
    df_increased = df_sorted[df_sorted['stage_diff'] > 0].copy()

    # retreive previous CKD_stage for context
    df_increased['previous_CKD_stage'] = df_sorted.groupby('PatientID')['CKD_stage_clean'].shift(1)
    
    # rename relevant columns
    result = df_increased[['PatientID', 'EventDate_dt', 'previous_CKD_stage', 'CKD_stage_clean']]
    result.rename(columns={'CKD_stage_clean': 'new_CKD_stage'}, inplace=True)
    
    return result

def unique_patient_ckd_counts(df):
    # Select only the necessary columns and drop duplicate rows based on PatientID
    # to ensure each patient is counted only once for their CKD stage.
    unique_patients_ckd = df[['PatientID', 'CKD_stage_clean']].drop_duplicates(subset=['PatientID'])

    # Count the occurrences of each CKD stage among these unique patients
    ckd_stage_counts = unique_patients_ckd['CKD_stage_clean'].value_counts()

    return ckd_stage_counts.sort_index()


#-------------------------------------------------------------------------------
# Functions: embedding generation counting - pandas
#-------------------------------------------------------------------------------
def count_subdirectories(path):
    """Counts the number of subdirectories in a given path.
    (counting patients)"""
    return sum(1 for entry in os.scandir(path) if entry.is_dir())

def count_files(path):
    """
    Counts all files within a given directory and its subdirectories.
    
    Args:
        path (str): The starting directory path.
    
    Returns:
        int: The total count of files.
        (counting embeddings)
    """
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(path):
        total_files += len(filenames)
    return total_files

#-------------------------------------------------------------------------------
# Functions: 
#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
# Functions: 
#-------------------------------------------------------------------------------


