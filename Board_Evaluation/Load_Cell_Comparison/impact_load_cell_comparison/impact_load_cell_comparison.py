"""
Filename: impact_load_cell_comparison.py

Description:
    Analysis module for comparing impact response characteristics across different load cell models.
    Processes impact data from CLZ635, DEGRAW, and TAL220 load cells to evaluate their 
    dynamic response to sudden force application.
    Generates visualizations for both full datasets and specific time ranges to highlight 
    differences in response characteristics.

Author: José Luis Berna Moya
Date: March 2025
"""

import pandas as pd
import matplotlib.pyplot as plt

def load_and_analyze_load_cell_data(file_path, cell_type):
    """
    Load load cell data from CSV file and analyze with visualizations.
    
    Parameters:
        file_path (str): Path to the CSV file containing load cell data.
        cell_type (str): Type of load cell being analyzed.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Plotting the entire dataset
    plt.figure(figsize=(12, 6))
    plt.plot(df['time ()'], df['reading ()'])
    plt.title(f'Load Cell Reading for {cell_type}')
    plt.xlabel('Time (Microseconds)')
    plt.ylabel('Reading (Grams-Force)')
    plt.grid(True)
    plt.show()

    # Specific plot for the time range 5.5050e8 to 5.5125e8 microseconds
    time_range_start = 5.5050e8
    time_range_end = 5.5125e8

    # Filter the data for the specified time range
    data_filtered = df[(df['time ()'] >= time_range_start) & (df['time ()'] <= time_range_end)]

    # Plot the filtered data for the specific time range
    plt.figure(figsize=(12, 6))
    plt.plot(data_filtered['time ()'], data_filtered['reading ()'])
    plt.title(f'Load Cell Reading for {cell_type} (5.5050e8 to 5.5125e8 Microseconds)')
    plt.xlabel('Time (Microseconds)')
    plt.ylabel('Reading (Grams-Force)')
    plt.grid(True)
    plt.show()

# Main execution block
if __name__ == "__main__":
    # File paths
    files = {
        'CLZ635': 'impactLoadCellcComparisonCLZ635.csv',
        'DEGRAW': 'impactLoadCellcComparisonDegraw.csv',
        'TAL220': 'impactLoadCellcComparisonTAL220.csv'
    }

    # Load data and plot for each cell type
    for cell_type, file_name in files.items():
        load_and_analyze_load_cell_data(file_name, cell_type)