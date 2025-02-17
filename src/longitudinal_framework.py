# longitudinal_framework.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def create_longitudinal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw data into a patient-level longitudinal dataset.
    
    - Sorts by patient_id and timestamp.
    - Fills missing values within each patient (forward/backward fill).
    - Adds a time feature (time since the patient’s first observation, in days).
    
    Returns:
      A DataFrame with a new column 'time_since_first'.
    """
    # Sort data by patient and time
    df = df.sort_values(by=['patient_id', 'timestamp'])
    
    # Fill missing values per patient (if any)
    df = df.groupby('patient_id').apply(
        lambda group: group.fillna(method='ffill').fillna(method='bfill')
    ).reset_index(drop=True)
    
    # Create time feature (in days since first observation)
    df['time_since_first'] = df.groupby('patient_id')['timestamp'].transform(
        lambda x: (x - x.min()).dt.total_seconds() / (3600 * 24)
    )
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    One-hot encodes the specified categorical columns.
    
    Args:
      df: Input DataFrame.
      categorical_columns: List of column names to be one-hot encoded.
      
    Returns:
      DataFrame with one-hot encoded features replacing the originals.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_array, 
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index
    )
    df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    return df
