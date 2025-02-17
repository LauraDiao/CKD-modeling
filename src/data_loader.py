import pandas as pd
import numpy as np

def load_mimic_data(file_path: str) -> pd.DataFrame:
    """
    Load the MIMIC-like demo dataset from a CSV file.
    
    This function:
      - Loads the CSV into a DataFrame.
      - Simulates missing demographics if needed (age, gender, race).
      - Simulates missing creatinine (uniform distribution).
      - Simulates eGFR if it is not provided, using a CKD-EPI–like formula.
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'])

    # 1. Simulate demographics if missing
    if 'age' not in df.columns:
        np.random.seed(42)
        df['age'] = np.random.randint(20, 90, size=len(df))
    if 'gender' not in df.columns:
        np.random.seed(42)
        df['gender'] = np.random.choice(['male', 'female'], size=len(df))
    if 'race' not in df.columns:
        np.random.seed(42)
        df['race'] = np.random.choice(['white', 'black', 'asian', 'other'], size=len(df))
    
    # 2. Simulate creatinine if missing
    if 'creatinine' not in df.columns:
        np.random.seed(42)
        df['creatinine'] = np.random.uniform(0.6, 1.5, size=len(df))
    
    # 3. Simulate eGFR if missing (simplified CKD-EPI–like formula)
    if 'eGFR' not in df.columns:
        def simulate_eGFR(row):
            creat = row['creatinine']
            age = row['age']
            gender = row['gender']
            if gender.lower() == 'female':
                kappa = 0.7
                alpha = -0.329
                gender_factor = 1.018
            else:
                kappa = 0.9
                alpha = -0.411
                gender_factor = 1.0
            ratio = creat / kappa
            eGFR_value = (
                141
                * (min(ratio, 1) ** alpha)
                * (max(ratio, 1) ** -1.209)
                * (0.993 ** age)
                * gender_factor
            )
            return eGFR_value
        df['eGFR'] = df.apply(simulate_eGFR, axis=1)
    
    return df
