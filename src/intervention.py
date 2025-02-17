import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from data_loader import load_mimic_data
from longitudinal_framework import create_longitudinal_dataset, encode_categorical_features
from intervention_manager import ensure_interventions

def create_future_scenario(df, patient_id, start_time_idx, end_time_idx, intervention_name):
    """
    For the specified patient, set the given intervention to '1' for
    time indices between start_time_idx and end_time_idx.
    """
    scenario_df = df.copy()
    mask = (
        (scenario_df["patient_id"] == patient_id) &
        (scenario_df["time_idx"] >= start_time_idx) &
        (scenario_df["time_idx"] <= end_time_idx)
    )
    scenario_df.loc[mask, intervention_name] = 1
    return scenario_df

def main():
    file_path_new = "mimic_demo_new.csv"
    df_new = load_mimic_data(file_path_new)
    df_new = create_longitudinal_dataset(df_new)
    
    # Same interventions as training
    possible_interventions = ["med_A", "med_B", "diag_X"]
    df_new = ensure_interventions(df_new, possible_interventions)
    
    # rename for forecasting
    df_new.rename(columns={"time_since_first": "time_idx"}, inplace=True)
    
    # We'll skip the usual categorical encoding for brevity or do it if needed:
    cat_cols = ["gender", "race"]
    df_new = encode_categorical_features(df_new, cat_cols)
    
    # Load model
    best_ckpt = "tft_model.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    
    # Example scenario: For patient_id=1234, let's set 'med_B'=1 from t=30 to t=40
    scenario_df = create_future_scenario(
        df_new, patient_id=1234, start_time_idx=30, end_time_idx=40, intervention_name="med_B"
    )
    
    # Build dataset from scenario
    max_encoder_length = 20
    max_prediction_length = 10
    scenario_dataset = TimeSeriesDataSet(
        scenario_df,
        time_idx="time_idx",
        target="eGFR",
        group_ids=["patient_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["patient_id"],
        static_reals=["age"],
        time_varying_known_reals=["time_idx"] + possible_interventions,
        time_varying_unknown_reals=["creatinine", "eGFR"],
        target_normalizer=model.target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    scenario_loader = scenario_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    # Predict eGFR under this scenario
    scenario_preds = model.predict(scenario_loader)
    
    print("Scenario predictions with 'med_B' = 1 from t=30 to t=40 for patient 1234:")
    print(scenario_preds)

if __name__ == "__main__":
    main()
