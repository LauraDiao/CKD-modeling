import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def create_longitudinal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw data into a patient-level longitudinal dataset.
    
    - Sort by patient_id and timestamp
    - Fill missing values within each patient
    - Add a time feature (days since first observation)
    """
    # Sort
    df = df.sort_values(by=['patient_id', 'timestamp'])
    df = df.groupby('patient_id').apply(
        lambda grp: grp.fillna(method='ffill').fillna(method='bfill')
    ).reset_index(drop=True)
    
    # Add 'time_since_first' in days
    df['time_since_first'] = df.groupby('patient_id')['timestamp'].transform(
        lambda x: (x - x.min()).dt.total_seconds() / (3600 * 24)
    )
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    One-hot encode all specified categorical columns.
    
    If an intervention is a binary 0/1, we can treat it as categorical for
    consistency (or keep it as a numeric if we prefer).
    """
    if not categorical_columns:
        return df
    
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_array, 
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index
    )
    
    df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    return df
