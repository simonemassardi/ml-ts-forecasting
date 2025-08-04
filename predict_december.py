"""
Predict December 2016 traffic values using trained LSTM model.
This script fills in the missing December 2016 data using iterative forecasting.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.data_module import TrafficDataModule
from src.models.lstm_model import LSTMTrafficModel
from src.utils.preprocessing import load_config
from src.utils.mlflow_utils import MLflowLogger, create_prediction_plot


def load_and_prepare_data(file_path: str):
    """
    Load traffic data and identify December 2016 missing period.
    
    Returns:
        df: Full dataframe
        historical_data: Data up to November 2016
        december_dates: DateTime index for December 2016
        last_valid_index: Index of last valid data point
    """
    print("Loading and analyzing dataset...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')
    
    # Find where December 2016 starts (first missing value)
    december_start_idx = df[df['DateTime'].dt.strftime('%Y-%m') == '2016-12'].index[0]
    
    # Historical data (everything before December 2016)
    historical_df = df.iloc[:december_start_idx].copy()
    historical_data = historical_df['X'].values
    
    # Remove any NaN values from historical data
    historical_data = historical_data[~np.isnan(historical_data)]
    
    # December 2016 datetime index
    december_df = df.iloc[december_start_idx:].copy()
    december_dates = december_df['DateTime'].values
    
    print(f"Historical data points: {len(historical_data)}")
    print(f"December 2016 missing values: {len(december_dates)}")
    print(f"Historical data range: {historical_df['DateTime'].min()} to {historical_df['DateTime'].max()}")
    print(f"December range to predict: {december_dates[0]} to {december_dates[-1]}")
    
    return df, historical_data, december_dates, december_start_idx


def iterative_forecast(model: LSTMTrafficModel, data_module: TrafficDataModule, historical_data, n_predictions, sequence_length):
    """
    Generate iterative forecasts for December 2016.
    
    Args:
        model: Trained LSTM model
        data_module: Data module for scaling
        historical_data: Historical traffic data
        n_predictions: Number of predictions to make
        sequence_length: Model sequence length
        
    Returns:
        predictions: Array of predicted values
    """
    model.eval()
    
    # Normalize historical data using the same scaler
    if data_module.normalize and data_module.scaler is not None:
        data_module.prepare_data()  # This loads the traffic data
        data_module.setup('fit')    # This fits the scaler
        historical_normalized = data_module.scaler.transform(historical_data.reshape(-1, 1)).flatten()
    else:
        historical_normalized = historical_data
    
    # Start with the last sequence_length points as context
    current_sequence = historical_normalized[-sequence_length:].copy()
    predictions = []
    
    print(f"Starting iterative prediction for {n_predictions} steps...")
    
    with torch.no_grad():
        for step in range(n_predictions):
            if step % 24 == 0:  # Print progress every day
                print(f"Predicting day {step//24 + 1} of {n_predictions//24 + 1}...")
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
            
            # Make prediction
            pred = model(input_tensor)
            pred_value = pred.cpu().numpy().flatten()[0]
            
            # Store prediction
            predictions.append(pred_value)
            
            # Update sequence for next prediction (rolling window)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_value
    
    predictions = np.array(predictions)
    
    # Denormalize predictions
    if data_module.normalize and data_module.scaler is not None:
        predictions = data_module.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return predictions


def create_december_visualization(predictions, december_dates, historical_data, historical_dates):
    """
    Create comprehensive visualization of December predictions.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Convert dates to pandas datetime for better plotting
    december_dates = pd.to_datetime(december_dates)
    historical_dates = pd.to_datetime(historical_dates)
    
    # Plot 1: Full timeline with predictions
    # Show last 2 months of historical + December predictions
    recent_mask = historical_dates >= '2016-10-01'
    
    axes[0].plot(historical_dates[recent_mask], historical_data[recent_mask], 
                 'b-', label='Historical (Oct-Nov 2016)', linewidth=1, alpha=0.8)
    axes[0].plot(december_dates, predictions, 
                 'r-', label='Predicted (Dec 2016)', linewidth=1.5)
    axes[0].axvline(x=december_dates[0], color='gray', linestyle='--', alpha=0.5, label='Prediction Start')
    axes[0].set_title('Traffic Prediction: December 2016 Forecast')
    axes[0].set_ylabel('Traffic Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: December predictions only (daily pattern)
    december_df = pd.DataFrame({
        'DateTime': december_dates,
        'Traffic': predictions
    })
    december_df['Hour'] = december_df['DateTime'].dt.hour
    december_df['Day'] = december_df['DateTime'].dt.day
    
    # Show first week in detail
    first_week = december_df[december_df['Day'] <= 7]
    axes[1].plot(first_week['DateTime'], first_week['Traffic'], 'g-', linewidth=1.5)
    axes[1].set_title('December 2016 Predictions - First Week Detail')
    axes[1].set_ylabel('Traffic Value')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Daily patterns analysis
    daily_avg = december_df.groupby('Hour')['Traffic'].mean()
    axes[2].plot(daily_avg.index, daily_avg.values, 'purple', linewidth=2, marker='o')
    axes[2].set_title('December 2016 - Average Daily Traffic Pattern')
    axes[2].set_xlabel('Hour of Day')
    axes[2].set_ylabel('Average Traffic Value')
    axes[2].set_xticks(range(0, 24, 2))
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_predictions_to_csv(predictions, december_dates, output_path='december_2016_predictions.csv'):
    """Save predictions to CSV file."""
    df = pd.DataFrame({
        'DateTime': december_dates,
        'Date': pd.to_datetime(december_dates).date,
        'Hour': pd.to_datetime(december_dates).hour,
        'X': predictions
    })
    
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def main():
    """Main function for December 2016 prediction."""
    parser = argparse.ArgumentParser(description='Predict December 2016 Traffic Values')
    
    # Model loading options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    model_group.add_argument('--mlflow_run_id', type=str, help='MLflow run ID')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    parser.add_argument('--data_path', type=str, default='data/traffic.csv', help='Traffic data path')
    parser.add_argument('--output_csv', type=str, default='december_2016_predictions.csv', help='Output CSV path')
    parser.add_argument('--log_to_mlflow', action='store_true', help='Log results to MLflow')
    parser.add_argument('--show_plots', action='store_true', help='Display plots')
    
    args = parser.parse_args()
    
    print("🚀 Starting December 2016 Traffic Prediction...")
    
    # Load configuration and data
    config = load_config(args.config)
    df, historical_data, december_dates, december_start_idx = load_and_prepare_data(args.data_path)
    n_december_hours = len(december_dates)
    
    # Set up data module
    data_module = TrafficDataModule(
        file_path=args.data_path,
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
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = LSTMTrafficModel.load_from_checkpoint(
            args.checkpoint,
            input_size=1,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            prediction_horizon=config['data']['prediction_horizon'],
            learning_rate=config['model']['learning_rate'],
            log_to_mlflow=False
        )
    else:
        print(f"Loading model from MLflow run: {args.mlflow_run_id}")
        model_uri = f"runs:/{args.mlflow_run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
    
    # Generate predictions
    print(f"Generating predictions for {n_december_hours} hours...")
    predictions = iterative_forecast(
        model, data_module, historical_data, 
        n_december_hours, config['data']['sequence_length']
    )
    
    # Create visualizations
    historical_dates = df.iloc[:december_start_idx]['DateTime'].values
    fig = create_december_visualization(predictions, december_dates, historical_data, historical_dates)
    
    # Save results
    save_predictions_to_csv(predictions, december_dates, args.output_csv)
    
    if args.show_plots:
        plt.show()
    else:
        plt.savefig('december_2016_predictions.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to: december_2016_predictions.png")
    
    print(f"✅ December 2016 prediction completed!")
    print(f"📊 Generated {len(predictions)} hourly predictions")


if __name__ == "__main__":
    main()
