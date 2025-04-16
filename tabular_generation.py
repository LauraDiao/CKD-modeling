import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

# -----------------------------
# Load and preprocess
# -----------------------------
df = pd.read_csv("patients_subset_10.csv", low_memory=False).drop_duplicates()
df['EventTimeStamp'] = pd.to_datetime(df['EventTimeStamp'], errors='coerce')
df['EventDate'] = df['EventTimeStamp'].dt.date
df['DataCategory'] = df['DataCategory'].fillna('None')
df['DataNumeric'] = pd.to_numeric(df['DataNumeric'], errors='coerce')
df['is_gfr'] = df['DataCategory'].str.upper().str.contains("GFR|GFREST", na=False)

# -----------------------------
# Base: full patient-day index
# -----------------------------
all_days = df[['PatientID', 'EventDate']].drop_duplicates().sort_values(['PatientID', 'EventDate'])

# -----------------------------
# Extract and forward-fill GFR
# -----------------------------
gfr_df = df[df['is_gfr'] & df['DataNumeric'].notna()]
gfr_daywise = (
    gfr_df.groupby(['PatientID', 'EventDate'])['DataNumeric']
    .first()
    .reset_index()
    .rename(columns={'DataNumeric': 'GFR_combined'})
)

base_df = pd.merge(all_days, gfr_daywise, on=['PatientID', 'EventDate'], how='left')
base_df = base_df.sort_values(['PatientID', 'EventDate'])
base_df["GFR_combined"] = base_df.groupby("PatientID")["GFR_combined"].ffill()

def gfr_to_stage(gfr):
    if pd.isna(gfr): return None, 0
    if gfr >= 90: return "1", 1
    if gfr >= 60: return "2", 2
    if gfr >= 45: return "3a", 3.1
    if gfr >= 30: return "3b", 3.2
    if gfr >= 15: return "4", 4
    return "5", 5

# Enforce monotonic CKD staging
new_stages = {}
for pid, group in base_df.groupby("PatientID"):
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

# -----------------------------
# One-hot encode diagnoses
# -----------------------------
diag_df = df[df["DataType"] == "Diagnosis"].copy()
diag_df["ICD_clean"] = diag_df["DataCategory"].astype(str).str.replace(".", "", regex=False)

diagnosis_map = diag_df.groupby(["PatientID", "EventDate"])["ICD_clean"].apply(list)
mlb = MultiLabelBinarizer()
diag_features = mlb.fit_transform(diagnosis_map.values)

diag_df_onehot = pd.DataFrame(
    diag_features,
    columns=[f"diag_{c}" for c in mlb.classes_],
    index=diagnosis_map.index
).reset_index()

base_df = pd.merge(base_df, diag_df_onehot, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Manual lab expansion (no pivot)
# -----------------------------
lab_df = df[df["DataType"] == "Lab"].copy()
lab_df["DataCategory"] = lab_df["DataCategory"].astype(str).str.upper()

lab_expanded = base_df.copy()
for lab in lab_df["DataCategory"].unique():
    lab_slice = lab_df[lab_df["DataCategory"] == lab]
    lab_slice = lab_slice.rename(columns={"DataNumeric": lab}).drop(columns="DataCategory")
    lab_expanded = pd.merge(lab_expanded, lab_slice[['PatientID', 'EventDate', lab]],
                            on=["PatientID", "EventDate"], how="left")

base_df = lab_expanded

# -----------------------------
# One-hot encode demographics
# -----------------------------
def format_demographics(row):
    race_ethnicity = str(row["DataCategory"]).replace("//", " ").replace("/", " ")
    if "Unknown Not Reported" in race_ethnicity:
        race_ethnicity = race_ethnicity.replace("Unknown Not Reported", "").strip()
    if "Do not identify with Race" in race_ethnicity:
        race_ethnicity = race_ethnicity.replace("Do not identify with Race", "unknown race").strip()
    return race_ethnicity

def build_demographic_map(df):
    demo_df = df[df["DataType"] == "Demographics"].dropna(subset=["DataCategory"])
    return demo_df.groupby("PatientID").first().apply(format_demographics, axis=1).to_dict()

demo_map = build_demographic_map(df)
demo_df = pd.DataFrame(list(demo_map.items()), columns=["PatientID", "demo_string"])

if not demo_df.empty:
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    demo_encoded = enc.fit_transform(demo_df[["demo_string"]])
    demo_onehot = pd.DataFrame(demo_encoded, columns=[f"demo_{c}" for c in enc.categories_[0]])
    demo_df = pd.concat([demo_df[["PatientID"]], demo_onehot], axis=1)
else:
    demo_df = pd.DataFrame(columns=["PatientID"])

base_df = pd.merge(base_df, demo_df, on="PatientID", how="left")

# -----------------------------
# Final report
# -----------------------------
print("[INFO] Final tabular shape:", base_df.shape)
print("[INFO] Sample features:\n", base_df.head())
print("[INFO] CKD stage counts:\n", base_df["CKD_stage"].value_counts(dropna=False))
base_df.to_csv("ckd_processed_tab_data_10.csv")
