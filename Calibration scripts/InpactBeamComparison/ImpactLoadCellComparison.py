import pandas as pd
import matplotlib.pyplot as plt

# File paths
files = {
    'CLZ635': 'impactLoadCellcComparisonCLZ635.csv',
    'DEGRAW': 'impactLoadCellcComparisonDegraw.csv',
    'TAL220': 'impactLoadCellcComparisonTAL220.csv'
}

# Load data and plot
for cell_type, file_name in files.items():
    df = pd.read_csv(file_name)
    
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
