import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

# We assume the pytorch_forecasting library is available.
# (If not, see: https://pytorch-forecasting.readthedocs.io/)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

class TFTModel(pl.LightningModule):
    """
    A PyTorch Lightning module that wraps a Temporal Fusion Transformer model.
    
    This class is provided for illustration. In our training script below we 
    directly instantiate a TFT via PyTorch Forecasting's API.
    """
    def __init__(self, training_dataset: TimeSeriesDataSet, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # e.g., predicting several quantiles
            loss=nn.L1Loss()
        )
        
    def forward(self, x):
        return self.tft(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
