"""
Main entry point for the LSTM traffic forecasting project.
"""

import argparse
from src.utils.preprocessing import load_and_explore_data, plot_traffic_data, analyze_seasonality


def explore_data(file_path: str = "data/traffic.csv"):
    """
    Explore the traffic dataset.
    
    Args:
        file_path: Path to the traffic CSV file
    """
    print("Loading and exploring traffic data...")
    df = load_and_explore_data(file_path)
    
    # Plot the data
    print("\nPlotting traffic data...")
    plot_traffic_data(df, start_date="2015-01-01", end_date="2015-02-01")
    
    # Analyze seasonality
    print("\nAnalyzing seasonal patterns...")
    seasonality_stats = analyze_seasonality(df)
    
    return df, seasonality_stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='LSTM Traffic Forecasting Project')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['explore', 'train', 'predict'],
        default='explore',
        help='Mode to run: explore data, train model, or make predictions'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data/traffic.csv',
        help='Path to traffic data file'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'explore':
        explore_data(args.data_path)
        print("\nData exploration completed!")
        print("Next steps:")
        print("1. Run 'python train.py' to train the LSTM model")
        print("2. Run 'python predict.py --checkpoint <path_to_checkpoint>' to make predictions")
    
    elif args.mode == 'train':
        print("To train the model, please run: python train.py")
    
    elif args.mode == 'predict':
        print("To make predictions, please run: python predict.py --checkpoint <path_to_checkpoint>")


if __name__ == "__main__":
    main()
