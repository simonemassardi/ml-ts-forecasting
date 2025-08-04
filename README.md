# LSTM Traffic Forecasting

A PyTorch Lightning-based project for predicting traffic time series data using LSTM neural networks.

![Static Badge](https://img.shields.io/badge/MLflow-grey?style=flat&logo=mlflow&logoColor=%230194E2&logoSize=auto&link=https%3A%2F%2Fmlflow.org%2Fdocs%2Flatest%2F)

![Static Badge](https://img.shields.io/badge/Lightning-grey?style=flat&logo=lightning&logoColor=%23792EE5&logoSize=auto&link=https%3A%2F%2Flightning.ai%2Fdocs%2Fpytorch%2Fstable%2F)



## Project Structure

```
ml-ts-forecasting/
├── data/
│   └── traffic.csv              # Traffic time series data
├── src/
│   ├── data/
│   │   └── data_module.py       # PyTorch Lightning DataModule
│   ├── models/
│   │   └── lstm_model.py        # LSTM model implementation
│   ├── utils/
│   │   └── preprocessing.py     # Data preprocessing utilities
│   └── __init__.py
├── configs/
│   └── config.yaml              # Model and training configuration
├── main.py                      # Main entry point for data exploration
├── train.py                     # Training script with MLflow tracking
├── predict.py                   # Prediction script with MLflow integration
├── mlflow_ui.py                 # MLflow utilities and UI launcher
├── pyproject.toml              # Project dependencies
└── README.md
```

## Features

- **LSTM Model**: Multi-layer LSTM for time series forecasting
- **PyTorch Lightning**: Structured training with automatic logging and checkpointing
- **MLflow Integration**: Complete experiment tracking, model versioning, and artifact management
- **Data Pipeline**: Efficient data loading with normalization and sequence creation
- **Configuration**: YAML-based configuration for easy hyperparameter tuning
- **Visualization**: Built-in plotting for data exploration and results analysis
- **Metrics**: Comprehensive evaluation metrics (MSE, RMSE, MAE, MAPE)

## Installation

1. Install dependencies using uv (recommended) or pip:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Usage

### 1. Data Exploration

Explore the traffic dataset and visualize patterns:

```bash
python main.py --mode explore
```

### 2. Training

Train the LSTM model with MLflow tracking:

```bash
python train.py
```

You can also specify a custom configuration file and run name:

```bash
python train.py --config configs/config.yaml --run_name "lstm_experiment_1"
```

### 3. Making Predictions

Make predictions using a trained model:

**From checkpoint:**
```bash
python predict.py --checkpoint checkpoints/lstm-traffic-epoch=XX-val_loss=X.XX.ckpt
```

**From MLflow model:**
```bash
python predict.py --mlflow_run_id <run_id>
```

**With MLflow logging:**
```bash
python predict.py --checkpoint <path> --log_to_mlflow
```

## Configuration

Edit `configs/config.yaml` to adjust model and training parameters:

- **Data parameters**: sequence length, prediction horizon, train/val/test splits
- **Model parameters**: hidden size, number of layers, dropout rate
- **Training parameters**: batch size, learning rate, early stopping patience
- **MLflow parameters**: experiment name, tracking URI, model logging settings

## Model Architecture

The LSTM model includes:
- Multi-layer LSTM with configurable hidden size and layers
- Dropout for regularization
- Dense output layer for predictions
- Adam optimizer with learning rate scheduling
- Early stopping and model checkpointing

## Data Format

The traffic data should be in CSV format with columns:
- `DateTime`: Timestamp for each observation
- `Date`: Date component
- `Hour`: Hour component (0-23)
- `X`: Traffic value (target variable)

## Experiment Tracking & Monitoring

### MLflow UI

View all experiments, runs, and metrics in the MLflow UI:

```bash
python mlflow_ui.py ui
```

Or start MLflow UI manually:

```bash
mlflow ui --backend-store-uri mlruns
```

### MLflow Utilities

**List all experiments:**
```bash
python mlflow_ui.py experiments
```

**List runs in an experiment:**
```bash
python mlflow_ui.py runs --experiment lstm_traffic_forecasting
```

**Find the best run:**
```bash
python mlflow_ui.py best --metric val_loss
```

### TensorBoard (Alternative)

Training progress can also be monitored using TensorBoard:

```bash
tensorboard --logdir logs/
```

## MLflow Integration Features

### Automatic Experiment Tracking
- **Parameters**: All hyperparameters and configuration settings
- **Metrics**: Training, validation, and test metrics with step tracking
- **Artifacts**: Model checkpoints, plots, and configuration files
- **Model Versioning**: Automatic model registration and versioning

### What Gets Tracked
- Model architecture parameters (hidden_size, num_layers, etc.)
- Training parameters (batch_size, learning_rate, etc.)
- Data parameters (sequence_length, train/val/test splits)
- Loss and metric curves over epochs
- Final test performance metrics
- Model artifacts and checkpoints

### Model Management
- Load models directly from MLflow runs
- Compare different experiment runs
- Model serving capabilities through MLflow
- Reproducible experiment tracking

## Results

The model will output:
- Training and validation loss curves (MLflow + TensorBoard)
- Test set evaluation metrics
- Prediction vs. actual value plots (automatically logged to MLflow)
- Model checkpoints and MLflow model artifacts
- Complete experiment tracking and comparison

## Example Workflow

1. **Explore data**: `python main.py --mode explore`
2. **Train model**: `python train.py --run_name "experiment_1"`
3. **View results**: `python mlflow_ui.py ui`
4. **Find best model**: `python mlflow_ui.py best --metric val_loss`
5. **Make predictions**: `python predict.py --mlflow_run_id <best_run_id> --log_to_mlflow`
