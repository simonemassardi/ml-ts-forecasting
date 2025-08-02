"""
Prediction script for LSTM traffic forecasting model.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import mlflow
import mlflow.pytorch

from src.data.data_module import TrafficDataModule
from src.models.lstm_model import LSTMTrafficModel
from src.utils.preprocessing import load_config
from src.utils.mlflow_utils import create_prediction_plot, MLflowLogger


def load_model_from_checkpoint(checkpoint_path: str, config: dict) -> LSTMTrafficModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config: Configuration dictionary
        
    Returns:
        Loaded LSTM model
    """
    model = LSTMTrafficModel.load_from_checkpoint(
        checkpoint_path,
        input_size=1,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        prediction_horizon=config['data']['prediction_horizon'],
        learning_rate=config['model']['learning_rate'],
        log_to_mlflow=False  # Disable MLflow logging during inference
    )
    model.eval()
    return model


def load_model_from_mlflow(run_id: str, model_name: str = "model") -> LSTMTrafficModel:
    """
    Load model from MLflow.
    
    Args:
        run_id: MLflow run ID
        model_name: Name of the model artifact in MLflow
        
    Returns:
        Loaded LSTM model
    """
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model


def make_predictions(
    model: LSTMTrafficModel, 
    data_module: TrafficDataModule,
    num_predictions: int = 100
) -> tuple:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained LSTM model
        data_module: Data module for data loading
        num_predictions: Number of predictions to make
        
    Returns:
        Tuple of (predictions, actual_values, timestamps)
    """
    model.eval()
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= num_predictions // data_module.batch_size:
                break
                
            pred = model(x)
            predictions.append(pred.cpu().numpy())
            actual_values.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actual_values = np.concatenate(actual_values)
    
    # Inverse transform if data was normalized
    if data_module.normalize:
        predictions = data_module.inverse_transform(predictions)
        actual_values = data_module.inverse_transform(actual_values)
    
    return predictions, actual_values


def plot_predictions(
    predictions: np.ndarray, 
    actual_values: np.ndarray,
    title: str = "Traffic Forecasting Results"
):
    """
    Plot prediction results.
    
    Args:
        predictions: Predicted values
        actual_values: Actual values
        title: Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Plot a subset for better visualization
    n_points = min(500, len(predictions))
    x_axis = range(n_points)
    
    plt.plot(x_axis, actual_values[:n_points], label='Actual', alpha=0.8, linewidth=1)
    plt.plot(x_axis, predictions[:n_points], label='Predicted', alpha=0.8, linewidth=1)
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_metrics(predictions: np.ndarray, actual_values: np.ndarray) -> dict:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted values
        actual_values: Actual values
        
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predictions - actual_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual_values))
    mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def main():
    """Main function for prediction."""
    parser = argparse.ArgumentParser(description='Make predictions with LSTM Traffic Forecasting Model')
    
    # Model loading options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--checkpoint', 
        type=str,
        help='Path to model checkpoint'
    )
    model_group.add_argument(
        '--mlflow_run_id',
        type=str,
        help='MLflow run ID to load model from'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num_predictions', 
        type=int, 
        default=100,
        help='Number of predictions to make'
    )
    parser.add_argument(
        '--log_to_mlflow',
        action='store_true',
        help='Log prediction results to MLflow'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up data module
    data_module = TrafficDataModule(
        file_path=config['data']['file_path'],
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        normalize=config['data']['normalize']
    )
    
    # Load model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}...")
        model = load_model_from_checkpoint(args.checkpoint, config)
    elif args.mlflow_run_id:
        print(f"Loading model from MLflow run: {args.mlflow_run_id}...")
        model = load_model_from_mlflow(args.mlflow_run_id)
    else:
        raise ValueError("Either --checkpoint or --mlflow_run_id must be provided")
    
    # Make predictions
    print("Making predictions...")
    predictions, actual_values = make_predictions(
        model, data_module, args.num_predictions
    )
    
    # Calculate metrics
    metrics = calculate_metrics(predictions.flatten(), actual_values.flatten())
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    plot_predictions(predictions.flatten(), actual_values.flatten())
    
    # Log to MLflow if requested
    if args.log_to_mlflow:
        # Initialize MLflow
        mlflow_config = config.get('mlflow', {})
        mlflow_logger = MLflowLogger(
            experiment_name=mlflow_config.get('experiment_name', 'lstm_traffic_forecasting'),
            tracking_uri=mlflow_config.get('tracking_uri', 'mlruns')
        )
        
        # Start prediction run
        run_name = f"prediction_{args.mlflow_run_id if args.mlflow_run_id else 'checkpoint'}"
        mlflow_run = mlflow_logger.start_run(run_name)
        
        try:
            # Log prediction metrics
            mlflow_logger.log_metrics(metrics)
            
            # Log model source
            if args.checkpoint:
                mlflow_logger.log_params({'model_source': 'checkpoint', 'checkpoint_path': args.checkpoint})
            elif args.mlflow_run_id:
                mlflow_logger.log_params({'model_source': 'mlflow', 'source_run_id': args.mlflow_run_id})
            
            # Create and log prediction plot
            pred_fig = create_prediction_plot(predictions.flatten(), actual_values.flatten())
            mlflow_logger.log_figure(pred_fig, "prediction_results.png")
            plt.close(pred_fig)  # Close to save memory
            
            print(f"\nResults logged to MLflow run: {mlflow_run.info.run_id}")
            
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")
        finally:
            mlflow_logger.end_run()


if __name__ == "__main__":
    main()