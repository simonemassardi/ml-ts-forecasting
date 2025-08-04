"""
LSTM model for traffic time series forecasting using PyTorch Lightning.
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import mlflow
import mlflow.pytorch


class LSTMTrafficModel(L.LightningModule):
    """LSTM-based traffic forecasting model."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 1,
        learning_rate: float = 0.001,
        log_to_mlflow: bool = True,
    ):
        """
        Args:
            input_size: Number of input features (1 for univariate time series)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            prediction_horizon: Number of time steps to predict
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        self.log_to_mlflow = log_to_mlflow
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, prediction_horizon)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, prediction_horizon)
        """
        # Reshape input to (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from the sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Generate predictions
        predictions = self.fc(last_output)  # (batch_size, prediction_horizon)
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        
        # Handle different target shapes
        if y.dim() > 1 and y.size(1) == 1:
            y = y.squeeze(1)
        
        loss = F.mse_loss(y_hat.squeeze(), y.squeeze())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log to MLflow
        if self.log_to_mlflow and mlflow.active_run():
            mlflow.log_metric('train_loss', loss.item(), step=self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        
        # Handle different target shapes
        if y.dim() > 1 and y.size(1) == 1:
            y = y.squeeze(1)
        
        loss = F.mse_loss(y_hat.squeeze(), y.squeeze())
        mae = F.l1_loss(y_hat.squeeze(), y.squeeze())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log to MLflow
        if self.log_to_mlflow and mlflow.active_run():
            mlflow.log_metric('val_loss', loss.item(), step=self.current_epoch)
            mlflow.log_metric('val_mae', mae.item(), step=self.current_epoch)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self(x)
        
        # Handle different target shapes
        if y.dim() > 1 and y.size(1) == 1:
            y = y.squeeze(1)
        
        loss = F.mse_loss(y_hat.squeeze(), y.squeeze())
        mae = F.l1_loss(y_hat.squeeze(), y.squeeze())
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        
        # Log to MLflow
        if self.log_to_mlflow and mlflow.active_run():
            mlflow.log_metric('test_loss', loss.item())
            mlflow.log_metric('test_mae', mae.item())
        
        return {'test_loss': loss, 'test_mae': mae}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'reduce_on_plateau': True,
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        x, _ = batch
        return self(x)