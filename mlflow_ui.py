"""
MLflow UI launcher and experiment utilities.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import mlflow
import pandas as pd
from src.utils.preprocessing import load_config


def start_mlflow_ui(tracking_uri: str = "mlruns", port: int = 5000):
    """
    Start MLflow UI server.
    
    Args:
        tracking_uri: MLflow tracking URI
        port: Port to run the UI on
    """
    try:
        print(f"Starting MLflow UI on port {port}...")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Open http://localhost:{port} in your browser")
        
        cmd = ["mlflow", "ui", "--backend-store-uri", tracking_uri, "--port", str(port)]
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow UI: {e}")
    except FileNotFoundError:
        print("MLflow not found. Please install MLflow: pip install mlflow")


def list_experiments(tracking_uri: str = "mlruns"):
    """
    List all MLflow experiments.
    
    Args:
        tracking_uri: MLflow tracking URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiments = mlflow.search_experiments()
        
        print("Available Experiments:")
        print("-" * 60)
        for exp in experiments:
            print(f"ID: {exp.experiment_id}")
            print(f"Name: {exp.name}")
            print(f"Status: {exp.lifecycle_stage}")
            print(f"Location: {exp.artifact_location}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error listing experiments: {e}")


def list_runs(experiment_name: str = "lstm_traffic_forecasting", tracking_uri: str = "mlruns"):
    """
    List runs from a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return
        
        print(f"Runs in experiment '{experiment_name}':")
        print("-" * 100)
        
        # Select relevant columns
        display_cols = ['run_id', 'status', 'start_time', 'end_time']
        metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        
        # Display basic info
        for _, run in runs.iterrows():
            print(f"Run ID: {run['run_id']}")
            print(f"Status: {run['status']}")
            print(f"Start Time: {run['start_time']}")
            print(f"End Time: {run['end_time']}")
            
            # Display key metrics
            if metric_cols:
                print("Metrics:")
                for col in metric_cols[:5]:  # Show first 5 metrics
                    metric_name = col.replace('metrics.', '')
                    print(f"  {metric_name}: {run[col]:.4f}")
            
            # Display key parameters
            if param_cols:
                print("Parameters:")
                for col in param_cols[:5]:  # Show first 5 parameters
                    param_name = col.replace('params.', '')
                    print(f"  {param_name}: {run[col]}")
            
            print("-" * 100)
            
    except Exception as e:
        print(f"Error listing runs: {e}")


def get_best_run(experiment_name: str = "lstm_traffic_forecasting", 
                metric: str = "val_loss", tracking_uri: str = "mlruns"):
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize (lower is better)
        tracking_uri: MLflow tracking URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )
        
        if runs.empty:
            print(f"No runs found in experiment '{experiment_name}'")
            return None
        
        best_run = runs.iloc[0]
        print(f"Best run based on {metric}:")
        print(f"Run ID: {best_run['run_id']}")
        print(f"{metric}: {best_run[f'metrics.{metric}']:.4f}")
        print(f"Status: {best_run['status']}")
        
        return best_run['run_id']
        
    except Exception as e:
        print(f"Error finding best run: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='MLflow utilities')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Start MLflow UI')
    ui_parser.add_argument('--port', type=int, default=5000, help='Port for MLflow UI')
    ui_parser.add_argument('--tracking_uri', type=str, default='mlruns', help='MLflow tracking URI')
    
    # List experiments command
    exp_parser = subparsers.add_parser('experiments', help='List experiments')
    exp_parser.add_argument('--tracking_uri', type=str, default='mlruns', help='MLflow tracking URI')
    
    # List runs command
    runs_parser = subparsers.add_parser('runs', help='List runs in experiment')
    runs_parser.add_argument('--experiment', type=str, default='lstm_traffic_forecasting', help='Experiment name')
    runs_parser.add_argument('--tracking_uri', type=str, default='mlruns', help='MLflow tracking URI')
    
    # Best run command
    best_parser = subparsers.add_parser('best', help='Find best run')
    best_parser.add_argument('--experiment', type=str, default='lstm_traffic_forecasting', help='Experiment name')
    best_parser.add_argument('--metric', type=str, default='val_loss', help='Metric to optimize')
    best_parser.add_argument('--tracking_uri', type=str, default='mlruns', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    if args.command == 'ui':
        start_mlflow_ui(args.tracking_uri, args.port)
    elif args.command == 'experiments':
        list_experiments(args.tracking_uri)
    elif args.command == 'runs':
        list_runs(args.experiment, args.tracking_uri)
    elif args.command == 'best':
        get_best_run(args.experiment, args.metric, args.tracking_uri)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()