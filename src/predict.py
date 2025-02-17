# train.py
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, GroupNormalizer

from data_loader import load_mimic_data
from longitudinal_framework import create_longitudinal_dataset, encode_categorical_features
from intervention_manager import ensure_interventions, simulate_interventions
from model_tft import TFTModel

def main():
    seed_everything(42)
    
    # Load and preprocess data
    file_path = "mimic_demo.csv"
    df = load_mimic_data(file_path)
    df = create_longitudinal_dataset(df)
    
    # Manage interventions
    # Suppose we have multiple discrete interventions to track
    possible_interventions = ["med_A", "med_B", "diag_X"]
    
    # Ensure columns exist (if missing, create them)
    df = ensure_interventions(df, possible_interventions)
    # For demonstration, randomly simulate them (10% chance for each)
    df = simulate_interventions(df, possible_interventions, probability=0.1)
    
    # Encode categorical features
    # Here, we treat 'gender' and 'race' as categorical, plus each intervention as well.
    categorical_cols = ["gender", "race"] + possible_interventions
    df = encode_categorical_features(df, categorical_cols)
    
    # rename time_since_first to "time_idx"
    df.rename(columns={"time_since_first": "time_idx"}, inplace=True)
    
    max_prediction_length = 10
    max_encoder_length = 20
    training_cutoff = df["time_idx"].max() - max_prediction_length
    
    # Build the TimeSeriesDataSet
    # We'll add normalizers for target and continuous variables
    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="eGFR",
        group_ids=["patient_id"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        
        static_categoricals=["patient_id"],  # or we could treat patient_id as a label encoder
        static_reals=["age"],
        
        time_varying_known_reals=["time_idx"] + possible_interventions,  
        # if you prefer to treat interventions as known real (0/1) instead of categorical
        
        time_varying_unknown_reals=["creatinine", "eGFR"],
        
        # Normalization for the target
        target_normalizer=GroupNormalizer(
            groups=["patient_id"], 
            transformation="softplus"
            # or "standard" / "robust" / "log" depending on your preference
        ),
        
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create dataloaders
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = training.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # train models
    model = TFTModel(
        training_dataset=training,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=16,
        output_size=7,  # for quantile forecasts
        learning_rate=1e-3
    )
    
    trainer = Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        gradient_clip_val=0.1,
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Save model
    trainer.save_checkpoint("tft_model.ckpt")
    print("Training complete. Model saved to 'tft_model.ckpt'.")

if __name__ == "__main__":
    main()
