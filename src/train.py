import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Trainer
from pytorch_forecasting.metrics import QuantileLoss

from data_loader import load_mimic_data
from longitudinal_framework import create_longitudinal_dataset, encode_categorical_features

def main():
    file_path = "mimic_demo.csv"  # Path to your MIMIC demo CSV file
    df = load_mimic_data(file_path)
    df = create_longitudinal_dataset(df)
    
    # For demonstration, we simulate an "intervention" column (e.g., medication change) randomly.
    if 'intervention' not in df.columns:
        np.random.seed(42)
        df['intervention'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    
    # Assume we want to encode 'gender' and 'race' and also the intervention (if desired).
    categorical_cols = ['gender', 'race', 'intervention']
    df = encode_categorical_features(df, categorical_cols)
    
    # Rename the computed time feature to "time_idx" for the forecasting API.
    df = df.rename(columns={'time_since_first': 'time_idx'})
    
    # Define forecasting horizons:
    max_prediction_length = 10  # forecast 10 time steps into the future
    max_encoder_length = 20     # use the previous 20 time steps as history
    training_cutoff = df['time_idx'].max() - max_prediction_length

    # Create the dataset.
    # In this demo:
    #   - Static features: patient_id and age.
    #   - Known time-varying features: time_idx and the intervention (discrete concept).
    #   - Unknown time-varying features: creatinine and eGFR.
    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="eGFR",
        group_ids=["patient_id"],
        min_encoder_length=max_encoder_length,  # allow encoder length to vary if desired
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["patient_id"],  # patient identifier as static (could be encoded further)
        static_reals=["age"],
        time_varying_known_reals=["time_idx", "intervention"],
        time_varying_unknown_reals=["creatinine", "eGFR"],
        target_normalizer=None,  # For demo purposes; consider standardization in practice.
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create dataloaders
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # predicting several quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    # Using PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=30,
        gpus=1 if torch.cuda.is_available() else 0,
    )
    
    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Save the trained model for future inference
    tft.save("tft_model.pth")
    print("Training complete. Model saved to tft_model.pth.")

if __name__ == "__main__":
    main()
