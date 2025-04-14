"""
Filename: adc_sampling_rate_analysis.py

Description:
    Analysis module for evaluating ADC timing consistency and sampling rates.
    Processes and visualizes timing interval data collected under different 
    oversampling ratio (OSR) configurations to evaluate data acquisition performance.
    Used for optimizing MCP356x ADC configuration in the FARE-KE framework.

Author: José Luis Berna Moya
Date: March 2025
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Load and preprocess the data from CSV file, handling potential errors in the file.
    
    Parameters:
        filename (str): Path to the CSV file containing timing data.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with microsecond timestamps and time deltas.
    """
    # Attempt to read the file
    while True:
        try:
            data = pd.read_csv(filename, skiprows=1, header=None, 
                              names=['micros', 'raw', 'butterworth'], dtype=str)
            break
        except pd.errors.ParserError as e:
            # Extract the problematic line number from the error message
            line_num = int(str(e).split("line ")[1].split(",")[0])
            
            # Drop the problematic row and save to a temporary CSV
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            with open(filename, 'w') as f:
                for i, line in enumerate(lines):
                    if i != line_num:
                        f.write(line)
    
    # Convert 'micros' and 'butterworth' columns to appropriate datatypes
    data['micros'] = data['micros'].astype(int)
    data['butterworth'] = data['butterworth'].astype(float)

    # Drop NaN values
    data.dropna(inplace=True)

    # Safely convert 'raw' column to integer
    data['raw'] = data['raw'].astype(int)

    # Compute the time difference between consecutive readings
    data['delta'] = data['micros'].diff().fillna(0)

    # Drop the first 2000 samples (warm-up period)
    data = data.iloc[2000:]

    return data


def process_and_plot(data, filename):
    """
    Process the timing data and generate visualization plots.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the timing data.
        filename (str): Name of the file being processed, for plot titles.
        
    Returns:
        tuple: A tuple containing:
            - average_time (float): Average time between readings in microseconds.
            - std_dev (float): Standard deviation of time intervals.
            - spikes (int): Number of intervals exceeding 50,000 microseconds.
    """
    average_time = data['delta'].mean()
    std_dev = data['delta'].std()
    spikes = (data['delta'] > 50000).sum()
    
    plt.figure(figsize=(15, 7))
    plt.plot(data.index, data['delta'], label=f'Standard Deviation: {std_dev:.2f} microseconds')
    plt.title(f'Reading Time Intervals for {filename}')
    plt.xlabel('Reading Index')
    plt.ylabel('Time Interval (microseconds)')
    plt.axhline(y=std_dev, color='r', linestyle='--', label='Standard Deviation Line')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return average_time, std_dev, spikes


if __name__ == "__main__":
    # File names to analyze
    filenames = ["OSR_32_single_message.csv", 
                "OSR_32_single_message_withFlush.csv", 
                "OSR_32_using_buffer.csv",
                "OSR_32_single_message_withFlush_timeDelay.csv"]

    # Load all datasets
    datasets = [load_data(filename) for filename in filenames]

    # Process and visualize each dataset
    for data, filename in zip(datasets, filenames):
        average_time, std_dev, spikes = process_and_plot(data, filename)
        print(f"\nMetrics for {filename}:")
        print(f"Average Reading Time: {average_time:.2f} microseconds")
        print(f"Standard Deviation of Reading Times: {std_dev:.2f} microseconds")
        print(f"Number of Spikes (above 50,000 microseconds): {spikes}")