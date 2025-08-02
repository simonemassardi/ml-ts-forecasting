"""
Training script for LSTM traffic forecasting model.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger
import mlflow
import mlflow.pytorch

from src.data.data_module import TrafficDataModule
from src.models.lstm_model import LSTMTrafficModel
from src.utils.preprocessing import load_config
from src.utils.mlflow_utils import MLflowLogger, log_data_info


def train_model(config_path: str = "configs/config.yaml", run_name: str = None):
    """
    Train the LSTM traffic forecasting model.
    
    Args:
        config_path: Path to the configuration file
        run_name: Optional MLflow run name
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize MLflow
    mlflow_config = config.get('mlflow', {})
    mlflow_logger = MLflowLogger(
        experiment_name=mlflow_config.get('experiment_name', 'lstm_traffic_forecasting'),
        tracking_uri=mlflow_config.get('tracking_uri', 'mlruns'),
        run_name=run_name or mlflow_config.get('run_name')
    )
    
    # Generate run name if not provided
    if not run_name and not mlflow_config.get('run_name'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"lstm_run_{timestamp}"
    
    # Start MLflow run
    mlflow_run = mlflow_logger.start_run(run_name)
    
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
    
    try:
        # Log configuration to MLflow
        mlflow_logger.log_config(config)
        
        # Set up model
        model = LSTMTrafficModel(
            input_size=1,  # Univariate time series
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            prediction_horizon=config['data']['prediction_horizon'],
            learning_rate=config['model']['learning_rate'],
            log_to_mlflow=True
        )
        
        # Log model parameters
        model_params = {
            'input_size': 1,
            'hidden_size': config['model']['hidden_size'],
            'num_layers': config['model']['num_layers'],
            'dropout': config['model']['dropout'],
            'prediction_horizon': config['data']['prediction_horizon'],
            'learning_rate': config['model']['learning_rate']
        }
        mlflow_logger.log_params(model_params)
    
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config['training']['patience'],
                mode='min',
                verbose=True
            ),
            ModelCheckpoint(
                dirpath='checkpoints/',
                filename='lstm-traffic-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=config['logging']['save_top_k'],
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Set up logger
        logger = TensorBoardLogger(
            save_dir='logs/',
            name='lstm_traffic_forecasting'
        )
        
        # Set up trainer
        trainer = L.Trainer(
            max_epochs=config['training']['max_epochs'],
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=config['training']['gradient_clip_val'],
            log_every_n_steps=config['logging']['log_every_n_steps'],
            deterministic=True,
            enable_checkpointing=True
        )
    
        # Log data information
        log_data_info(mlflow_logger, data_module)
        
        # Train the model
        print("Starting training...")
        trainer.fit(model, data_module)
        
        # Test the model
        print("Testing the model...")
        test_results = trainer.test(model, data_module)
        
        # Log final test metrics
        if test_results:
            final_metrics = test_results[0]
            mlflow_logger.log_metrics({
                'final_test_loss': final_metrics.get('test_loss', 0.0),
                'final_test_mae': final_metrics.get('test_mae', 0.0)
            })
        
        # Log model to MLflow if enabled
        if mlflow_config.get('log_model', True):
            print("Logging model to MLflow...")
            mlflow.pytorch.log_model(
                model, 
                "model",
                pip_requirements=[
                    "torch>=2.0.0",
                    "lightning>=2.5.2",
                    "numpy>=1.24.0"
                ]
            )
        
        # Log best checkpoint path
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path and mlflow_config.get('log_artifacts', True):
            mlflow_logger.log_artifact(best_model_path, "checkpoints")
        
        print(f"Training completed! Best model saved to: {best_model_path}")
        print(f"MLflow run ID: {mlflow_run.info.run_id}")
        print(f"MLflow run URL: {mlflow.get_tracking_uri()}/#/experiments/{mlflow_run.info.experiment_id}/runs/{mlflow_run.info.run_id}")
        
        return trainer, model, data_module
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        # End MLflow run
        mlflow_logger.end_run()
    return trainer, model, data_module


def main():
    """Main function for training."""
    parser = argparse.ArgumentParser(description='Train LSTM Traffic Forecasting Model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='MLflow run name (auto-generated if not provided)'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Train the model
    trainer, model, data_module = train_model(args.config, args.run_name)


if __name__ == "__main__":
    main()