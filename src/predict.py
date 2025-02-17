import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from data_loader import load_mimic_data
from longitudinal_framework import create_longitudinal_dataset, encode_categorical_features
from intervention_manager import ensure_interventions
# no need to simulate here unless you want to artificially create interventions

def main():
    file_path_new = "mimic_demo_new.csv"
    df_new = load_mimic_data(file_path_new)
    df_new = create_longitudinal_dataset(df_new)
    
    # Suppose we have the same interventions as training
    possible_interventions = ["med_A", "med_B", "diag_X"]
    df_new = ensure_interventions(df_new, possible_interventions)
    
    # (Optional) encode categorical features if needed
    # But if we used numeric 0/1 for interventions, we only encode 'gender' and 'race'.
    categorical_cols = ["gender", "race"]  
    df_new = encode_categorical_features(df_new, categorical_cols)
    
    df_new.rename(columns={"time_since_first": "time_idx"}, inplace=True)
    
    # Load trained model
    best_ckpt = "tft_model.ckpt"
    trained_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    
    # Build a dataset for predictions
    max_encoder_length = 20
    max_prediction_length = 10
    new_dataset = TimeSeriesDataSet(
        df_new,
        time_idx="time_idx",
        target="eGFR",
        group_ids=["patient_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        
        static_categoricals=["patient_id"],
        static_reals=["age"],
        time_varying_known_reals=["time_idx"] + possible_interventions,
        time_varying_unknown_reals=["creatinine", "eGFR"],
        
        target_normalizer=trained_model.target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    new_dataloader = new_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    # Generate predictions
    predictions = trained_model.predict(new_dataloader)
    print("Predictions shape:", predictions.shape)
    print("Example predictions:", predictions[:5])

if __name__ == "__main__":
    main()
