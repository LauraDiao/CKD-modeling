import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from data_loader import load_mimic_data
from longitudinal_framework import create_longitudinal_dataset, encode_categorical_features

def main():
    file_path = "mimic_demo_new.csv"  # New data file with similar structure.
    df_new = load_mimic_data(file_path)
    df_new = create_longitudinal_dataset(df_new)
    
    # Ensure the discrete concept exists in the new data (simulate if needed)
    if 'intervention' not in df_new.columns:
        import numpy as np
        np.random.seed(42)
        df_new['intervention'] = np.random.choice([0, 1], size=len(df_new), p=[0.9, 0.1])
    
    categorical_cols = ['gender', 'race', 'intervention']
    df_new = encode_categorical_features(df_new, categorical_cols)
    df_new = df_new.rename(columns={'time_since_first': 'time_idx'})
    
    tft = TemporalFusionTransformer.load_from_checkpoint("tft_model.pth")
    
    max_encoder_length = 20
    max_prediction_length = 10
    new_dataset = TimeSeriesDataSet(
        df_new,
        time_idx="time_idx",
        target="eGFR",
        group_ids=["patient_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["patient_id"],
        static_reals=["age"],
        time_varying_known_reals=["time_idx", "intervention"],
        time_varying_unknown_reals=["creatinine", "eGFR"],
        target_normalizer=tft.target_normalizer,  # use same normalization as training
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    new_dataloader = new_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    predictions = tft.predict(new_dataloader)
    print("Predicted eGFR trajectories:")
    print(predictions)

if __name__ == "__main__":
    main()
