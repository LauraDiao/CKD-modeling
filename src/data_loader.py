# data_loader.py
import pandas as pd
import numpy as np

def load_mimic_data(file_path: str) -> pd.DataFrame:
    """
    Load the MIMIC demo dataset from a CSV file.
    
    Expected columns:
      - patient_id: Unique patient identifier.
      - timestamp: Date/time of the observation.
      - creatinine: Lab value (if missing, it will be simulated).
      - age, gender, race: Demographic information (if missing, will be simulated).
      - (optionally) eGFR: If not present, we simulate it.
    
    Returns:
      A pandas DataFrame with the required columns.
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Simulate demographics if missing
    if 'age' not in df.columns:
        np.random.seed(42)
        df['age'] = np.random.randint(20, 90, size=len(df))
    if 'gender' not in df.columns:
        np.random.seed(42)
        df['gender'] = np.random.choice(['male', 'female'], size=len(df))
    if 'race' not in df.columns:
        np.random.seed(42)
        df['race'] = np.random.choice(['white', 'black', 'asian', 'other'], size=len(df))
    
    # Simulate creatinine if not present
    if 'creatinine' not in df.columns:
        np.random.seed(42)
        df['creatinine'] = np.random.uniform(0.6, 1.5, size=len(df))
    
    # If eGFR is not available, simulate it using a CKD-EPI–like formula.
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
            min_ratio = min(ratio, 1)
            max_ratio = max(ratio, 1)
            eGFR_value = 141 * (min_ratio ** alpha) * (max_ratio ** -1.209) * (0.993 ** age) * gender_factor
            return eGFR_value
        df['eGFR'] = df.apply(simulate_eGFR, axis=1)
        
    return df

