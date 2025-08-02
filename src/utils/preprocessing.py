"""
Utility functions for data preprocessing and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_and_explore_data(file_path: str) -> pd.DataFrame:
    """
    Load traffic data and perform basic exploration.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(file_path)
    
    # Convert DateTime column to proper datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')
    
    print("Dataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Traffic value range: {df['X'].min():.4f} to {df['X'].max():.4f}")
    
    return df


def plot_traffic_data(df: pd.DataFrame, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None, figsize: Tuple[int, int] = (15, 6)):
    """
    Plot traffic data over time.
    
    Args:
        df: DataFrame with traffic data
        start_date: Start date for plotting (optional)
        end_date: End date for plotting (optional)
        figsize: Figure size tuple
    """
    plot_df = df.copy()
    
    if start_date:
        plot_df = plot_df[plot_df['DateTime'] >= start_date]
    if end_date:
        plot_df = plot_df[plot_df['DateTime'] <= end_date]
    
    plt.figure(figsize=figsize)
    plt.plot(plot_df['DateTime'], plot_df['X'], linewidth=0.8)
    plt.title('Traffic Data Over Time')
    plt.xlabel('Date')
    plt.ylabel('Traffic Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_daily_patterns(df: pd.DataFrame, sample_days: int = 7):
    """
    Plot daily traffic patterns for a sample of days.
    
    Args:
        df: DataFrame with traffic data
        sample_days: Number of days to sample and plot
    """
    # Group by date and take a sample
    daily_data = df.groupby('Date').apply(lambda x: x.set_index('Hour')['X']).unstack(level=0)
    sample_dates = daily_data.columns[:sample_days]
    
    plt.figure(figsize=(12, 6))
    for date in sample_dates:
        plt.plot(daily_data.index, daily_data[date], label=str(date), alpha=0.7)
    
    plt.title(f'Daily Traffic Patterns (Sample of {sample_days} days)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Traffic Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def analyze_seasonality(df: pd.DataFrame):
    """
    Analyze seasonal patterns in the traffic data.
    
    Args:
        df: DataFrame with traffic data
    """
    df_copy = df.copy()
    df_copy['DateTime'] = pd.to_datetime(df_copy['DateTime'])
    df_copy['DayOfWeek'] = df_copy['DateTime'].dt.day_name()
    df_copy['Month'] = df_copy['DateTime'].dt.month_name()
    df_copy['Hour'] = df_copy['DateTime'].dt.hour
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Hourly patterns
    hourly_mean = df_copy.groupby('Hour')['X'].mean()
    axes[0, 0].plot(hourly_mean.index, hourly_mean.values)
    axes[0, 0].set_title('Average Traffic by Hour of Day')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Average Traffic')
    
    # Day of week patterns
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_mean = df_copy.groupby('DayOfWeek')['X'].mean().reindex(day_order)
    axes[0, 1].bar(range(len(daily_mean)), daily_mean.values)
    axes[0, 1].set_title('Average Traffic by Day of Week')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Average Traffic')
    axes[0, 1].set_xticks(range(len(day_order)))
    axes[0, 1].set_xticklabels(day_order, rotation=45)
    
    # Monthly patterns
    monthly_mean = df_copy.groupby('Month')['X'].mean()
    axes[1, 0].bar(range(len(monthly_mean)), monthly_mean.values)
    axes[1, 0].set_title('Average Traffic by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Traffic')
    axes[1, 0].set_xticks(range(len(monthly_mean)))
    axes[1, 0].set_xticklabels(monthly_mean.index, rotation=45)
    
    # Distribution
    axes[1, 1].hist(df_copy['X'], bins=50, alpha=0.7)
    axes[1, 1].set_title('Traffic Value Distribution')
    axes[1, 1].set_xlabel('Traffic Value')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'hourly_stats': hourly_mean.describe(),
        'daily_stats': daily_mean.describe(),
        'monthly_stats': monthly_mean.describe()
    }