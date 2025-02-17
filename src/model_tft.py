import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

class TFTModel(pl.LightningModule):
    """
    Wraps PyTorch Forecasting's TemporalFusionTransformer with customized hyperparams.
    """
    def __init__(
        self, 
        training_dataset: TimeSeriesDataSet,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.2,
        hidden_continuous_size: int = 16,
        output_size: int = 7,  # number of quantiles or a single numeric output
        loss=None,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        
        if loss is None:
            # Default to quantile loss. You can also use MSE, MAE, etc.
            loss = QuantileLoss()
        
        self.save_hyperparameters()  # saves hyperparameters to checkpoint
        self.tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=output_size,
            loss=loss,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
    
    def forward(self, x):
        return self.tft(x)
    
    def configure_optimizers(self):
        return self.tft.configure_optimizers()
    
    def training_step(self, batch, batch_idx):
        return self.tft.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.tft.validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.tft.test_step(batch, batch_idx)
