"""
PyTorch Lightning DataModule for traffic time series data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.preprocessing import StandardScaler


class TrafficDataset(Dataset):
    """Dataset for traffic time series data."""
    
    def __init__(
        self, 
        data: np.ndarray, 
        sequence_length: int, 
        prediction_horizon: int = 1
    ):
        """
        Args:
            data: Normalized traffic data array
            sequence_length: Number of past time steps to use
            prediction_horizon: Number of time steps to predict
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.sequence_length]
        # Target (next value(s) to predict)
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


class TrafficDataModule(L.LightningDataModule):
    """Lightning DataModule for traffic data."""
    
    def __init__(
        self,
        file_path: str,
        sequence_length: int = 24,
        prediction_horizon: int = 1,
        batch_size: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        
        self.scaler = StandardScaler() if normalize else None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        
    def prepare_data(self):
        """Download and prepare data (called once per node)."""
        # Load the CSV file
        df = pd.read_csv(self.file_path)
        
        # Extract the target variable (X column)
        self.traffic_data = df['X'].values

        # Exclude missing values
        self.traffic_data = self.traffic_data[~np.isnan(self.traffic_data)]
        
        # Store datetime for later reference
        self.datetime_index = pd.to_datetime(df['DateTime'])
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            # Split the data
            n_total = len(self.traffic_data)
            n_train = int(n_total * self.train_split)
            n_val = int(n_total * self.val_split)
            
            train_data = self.traffic_data[:n_train]
            val_data = self.traffic_data[n_train:n_train + n_val]
            test_data = self.traffic_data[n_train + n_val:]
            
            # Normalize the data
            if self.normalize:
                train_data = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
                val_data = self.scaler.transform(val_data.reshape(-1, 1)).flatten()
                test_data = self.scaler.transform(test_data.reshape(-1, 1)).flatten()
            
            # Create datasets
            self.data_train = TrafficDataset(
                train_data, self.sequence_length, self.prediction_horizon
            )
            self.data_val = TrafficDataset(
                val_data, self.sequence_length, self.prediction_horizon
            )
            
        if stage == "test" or stage is None:
            n_total = len(self.traffic_data)
            n_train = int(n_total * self.train_split)
            n_val = int(n_total * self.val_split)
            test_data = self.traffic_data[n_train + n_val:]
            
            if self.normalize and self.scaler is not None:
                test_data = self.scaler.transform(test_data.reshape(-1, 1)).flatten()
            
            self.data_test = TrafficDataset(
                test_data, self.sequence_length, self.prediction_horizon
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val, 
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True
        )
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale."""
        if self.normalize and self.scaler is not None:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data