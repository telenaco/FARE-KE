"""
Filename: hx711_vs_mcp3564_timing_analysis.py

Description:
    Analysis module for comparing timing performance between HX711 and MCP356x ADCs.
    Processes timing data to quantify sampling rate differences and temporal consistency.
    Generates statistical metrics and visualizations to evaluate relative timing performance.
    Used to determine optimal ADC selection for time-sensitive force measurement applications.

Author: José Luis Berna Moya
Date: March 2025
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Load and prepare timing data from CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file containing the timing data.
        
    Returns:
        pd.DataFrame: Processed DataFrame with raw readings and timing data.
    """
    # Load the data from the CSV file
    data = pd.read_csv(file_path, header=None)
    data.columns = ["MCP_Reading", "HX711_Reading", "Timestamp", "MCP_Flag", "HX711_Flag"]
    return data


def analyze_timing_statistics(data):
    """
    Analyze timing characteristics for both ADC types.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the timing data.
        
    Returns:
        dict: Dictionary containing timing statistics for both ADC types.
    """
    # Filter out the rows where MCP_Flag is 1
    mcp_data = data[data["MCP_Flag"] == 1]
    # Calculate the time differences between consecutive samples for MCP
    mcp_time_diffs = mcp_data["Timestamp"].diff().dropna()

    # Filter out the rows where HX711_Flag is 1
    hx711_data = data[data["HX711_Flag"] == 1]
    # Calculate the time differences between consecutive samples for HX711
    hx711_time_diffs = hx711_data["Timestamp"].diff().dropna()

    # Calculate mean and standard deviation for both MCP and HX711
    mcp_mean_diff = mcp_time_diffs.mean()
    mcp_std_diff = mcp_time_diffs.std()

    hx711_mean_diff = hx711_time_diffs.mean()
    hx711_std_diff = hx711_time_diffs.std()
    
    # Compute the samples per second for both MCP and HX711
    sps_mcp = 1e6 / mcp_mean_diff
    sps_hx711 = 1e6 / hx711_mean_diff
    
    # Create a dictionary to store the statistics
    statistics = {
        'MCP': {
            'mean_interval': mcp_mean_diff,
            'std_dev': mcp_std_diff,
            'samples_per_second': sps_mcp
        },
        'HX711': {
            'mean_interval': hx711_mean_diff,
            'std_dev': hx711_std_diff,
            'samples_per_second': sps_hx711
        }
    }
    
    return statistics, mcp_time_diffs, hx711_time_diffs


def plot_timing_histograms(mcp_time_diffs, hx711_time_diffs, stats):
    """
    Generate histogram plots showing the distribution of time intervals for both ADCs.
    
    Parameters:
        mcp_time_diffs (pd.Series): Time differences for MCP readings.
        hx711_time_diffs (pd.Series): Time differences for HX711 readings.
        stats (dict): Dictionary containing timing statistics.
    """
    # Histogram plotting for MCP
    plt.figure(figsize=(7, 6))
    plt.hist(mcp_time_diffs, bins=50, color='blue', alpha=0.7)
    plt.axvline(stats['MCP']['mean_interval'], color='r', linestyle='dashed', linewidth=2, 
                label=f'Mean: {stats["MCP"]["mean_interval"]:.2f} μs')
    plt.title('Time Differences Between Consecutive MCP Readings')
    plt.xlabel('Time Interval (microseconds)')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    # Histogram plotting for HX711
    plt.figure(figsize=(7, 6))
    plt.hist(hx711_time_diffs, bins=50, color='green', alpha=0.7)
    plt.axvline(stats['HX711']['mean_interval'], color='r', linestyle='dashed', linewidth=2,
                label=f'Mean: {stats["HX711"]["mean_interval"]:.2f} μs')
    plt.title('Time Differences Between Consecutive HX711 Readings')
    plt.xlabel('Time Interval (microseconds)')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


def print_timing_statistics(stats):
    """
    Print timing statistics for both ADC types in a formatted way.
    
    Parameters:
        stats (dict): Dictionary containing timing statistics.
    """
    print("\nMCP Timing Statistics:")
    print(f"Average Time Difference: {stats['MCP']['mean_interval']:.2f} microseconds")
    print(f"Standard Deviation: {stats['MCP']['std_dev']:.2f} microseconds")
    print(f"Samples Per Second: {stats['MCP']['samples_per_second']:.2f}")

    print("\nHX711 Timing Statistics:")
    print(f"Average Time Difference: {stats['HX711']['mean_interval']:.2f} microseconds")
    print(f"Standard Deviation: {stats['HX711']['std_dev']:.2f} microseconds")
    print(f"Samples Per Second: {stats['HX711']['samples_per_second']:.2f}")


# Main execution block
if __name__ == "__main__":
    # Load the data
    data = load_data('ellapsedData.csv')
    
    # Analyze timing data
    statistics, mcp_time_diffs, hx711_time_diffs = analyze_timing_statistics(data)
    
    # Print statistics
    print_timing_statistics(statistics)
    
    # Generate plots
    plot_timing_histograms(mcp_time_diffs, hx711_time_diffs, statistics)