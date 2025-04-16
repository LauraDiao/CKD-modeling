import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

# -----------------------------
# Load and preprocess
# -----------------------------
df = pd.read_csv("patients_subset_100.csv", low_memory=False).drop_duplicates()
df['EventTimeStamp'] = pd.to_datetime(df['EventTimeStamp'], errors='coerce')
df['EventDate'] = df['EventTimeStamp'].dt.date
df['DataCategory'] = df['DataCategory'].fillna('None')
df['DataNumeric'] = pd.to_numeric(df['DataNumeric'], errors='coerce')

# -----------------------------
# Base: full patient-day index
# -----------------------------
df['is_gfr'] = df['DataCategory'].str.upper().str.contains("GFR|GFREST", na=False)
all_days = df[['PatientID', 'EventDate']].drop_duplicates().sort_values(['PatientID', 'EventDate'])

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
mlb_diag = MultiLabelBinarizer()
diag_features = mlb_diag.fit_transform(diagnosis_map.values)

diag_df_onehot = pd.DataFrame(
    diag_features,
    columns=[f"diag_{c}" for c in mlb_diag.classes_],
    index=diagnosis_map.index
).reset_index()

base_df = pd.merge(base_df, diag_df_onehot, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# One-hot encode medications
# -----------------------------
med_df = df[df["DataType"] == "Medications"].copy()
med_df["med_clean"] = med_df["DataCategory"].astype(str).str.upper().str.replace(" ", "_")

medication_map = med_df.groupby(["PatientID", "EventDate"])["med_clean"].apply(list)
mlb_med = MultiLabelBinarizer()
med_features = mlb_med.fit_transform(medication_map.values)

med_df_onehot = pd.DataFrame(
    med_features,
    columns=[f"med_{c}" for c in mlb_med.classes_],
    index=medication_map.index
).reset_index()

base_df = pd.merge(base_df, med_df_onehot, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Pivot-style lab expansion
# -----------------------------
lab_df = df[(df["DataType"] == "Labs") & df["DataNumeric"].notna()].copy()
lab_df["DataCategory"] = lab_df["DataCategory"].astype(str).str.upper()

lab_pivot = (
    lab_df.groupby(["PatientID", "EventDate", "DataCategory"])["DataNumeric"]
    .first().unstack("DataCategory").reset_index()
)

lab_pivot.columns = ["PatientID", "EventDate"] + [f"lab_{c}" for c in lab_pivot.columns[2:]]
base_df = pd.merge(base_df, lab_pivot, on=["PatientID", "EventDate"], how="left")

# -----------------------------
# Optional: One-hot encode demographics
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
base_df.to_csv("ckd_processed_tab_data_100.csv", index=False)
