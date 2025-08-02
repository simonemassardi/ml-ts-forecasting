"""
MLflow utilities for experiment tracking.
"""

import mlflow
import mlflow.pytorch
import tempfile
import os
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class MLflowLogger:
    """MLflow experiment tracking utility."""
    
    def __init__(
        self, 
        experiment_name: str,
        tracking_uri: str = "mlruns",
        run_name: Optional[str] = None
    ):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (local or remote)
            run_name: Optional run name
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set MLflow experiment: {e}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Start MLflow run."""
        effective_run_name = run_name or self.run_name
        mlflow.start_run(run_name=effective_run_name)
        return mlflow.active_run()
    
    def end_run(self):
        """End MLflow run."""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            print(f"Warning: Could not log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log PyTorch model to MLflow."""
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            print(f"Warning: Could not log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to MLflow."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"Warning: Could not log artifact: {e}")
    
    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib figure to MLflow."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, artifact_file)
                figure.savefig(temp_path, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(temp_path)
        except Exception as e:
            print(f"Warning: Could not log figure: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration as parameters, handling nested dictionaries."""
        try:
            flattened = self._flatten_dict(config)
            self.log_params(flattened)
        except Exception as e:
            print(f"Warning: Could not log config: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def create_prediction_plot(
    predictions: np.ndarray, 
    actual_values: np.ndarray,
    title: str = "Traffic Forecasting Results",
    n_points: int = 500
) -> plt.Figure:
    """
    Create prediction plot for MLflow logging.
    
    Args:
        predictions: Predicted values
        actual_values: Actual values
        title: Plot title
        n_points: Number of points to plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot a subset for better visualization
    n_points = min(n_points, len(predictions))
    x_axis = range(n_points)
    
    ax.plot(x_axis, actual_values[:n_points], label='Actual', alpha=0.8, linewidth=1)
    ax.plot(x_axis, predictions[:n_points], label='Predicted', alpha=0.8, linewidth=1)
    
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Traffic Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_loss_plot(train_losses: list, val_losses: list) -> plt.Figure:
    """
    Create training/validation loss plot for MLflow logging.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def log_data_info(mlflow_logger: MLflowLogger, data_module):
    """Log dataset information to MLflow."""
    try:
        # Prepare data to get info
        data_module.prepare_data()
        data_module.setup('fit')
        
        info = {
            'dataset_size': len(data_module.traffic_data),
            'train_size': len(data_module.data_train),
            'val_size': len(data_module.data_val),
            'sequence_length': data_module.sequence_length,
            'prediction_horizon': data_module.prediction_horizon,
            'batch_size': data_module.batch_size,
            'normalized': data_module.normalize
        }
        
        mlflow_logger.log_params(info)
        
    except Exception as e:
        print(f"Warning: Could not log data info: {e}")


def get_run_metrics_summary(run_id: str) -> Dict[str, float]:
    """
    Get summary of metrics from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Dictionary of final metric values
    """
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        print(f"Warning: Could not get run metrics: {e}")
        return {}