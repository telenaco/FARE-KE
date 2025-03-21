import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import numpy as np
from scipy.signal import correlate


def load_data(file_path):
    data = pd.read_csv(file_path, header=None, names=[
        'microseconds', 'MCP_no_filter', 'MCP_with_filter', 'HX_no_filter', 'HX_with_filter'])
    data['milliseconds'] = data['microseconds'] / 1000
    return data


def plot_dataset_comparison(data):
    plt.figure(figsize=(15, 10))
    plt.plot(data['milliseconds'], data['MCP_no_filter'],
             label='MCP (No Filter)', color='blue')
    plt.plot(data['milliseconds'], data['MCP_with_filter'],
             label='MCP (With Filter)', color='blue', linestyle='--')
    plt.plot(data['milliseconds'], data['HX_no_filter'],
             label='HX711 (No Filter)', color='orange')
    plt.plot(data['milliseconds'], data['HX_with_filter'],
             label='HX711 (With Filter)', color='orange', linestyle='--')
    plt.title('Comparison between MCP3564, HX711 and their Filtered Versions')
    plt.xlabel('Milliseconds')
    plt.ylabel('Reading Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def identify_groups(data, spike_threshold=800, zero_threshold=50, zero_duration=2000, trail_duration=2000):
    groups = []
    is_in_group = False
    start_index = None
    zero_counter = 0
    for i, row in data.iterrows():
        if abs(row['MCP_no_filter']) <= zero_threshold:
            zero_counter += (data['milliseconds'].iloc[i] -
                             data['milliseconds'].iloc[i-1])
        else:
            zero_counter = 0

        if not is_in_group and row['MCP_no_filter'] > spike_threshold:
            is_in_group = True
        elif is_in_group and abs(row['MCP_no_filter']) <= zero_threshold and start_index is None:
            start_index = i
        elif is_in_group and zero_counter >= zero_duration:
            is_in_group = False
            end_index = i - \
                int(trail_duration /
                    (data['milliseconds'].iloc[i] - data['milliseconds'].iloc[i-1]))
            if start_index:
                groups.append((start_index, end_index))
            start_index = None
            zero_counter = 0
    return groups



def plot_individual_groups(data, groups):
    for idx, (start, end) in enumerate(groups, 1):
        plt.figure(figsize=(10, 6))
        subset = data.iloc[start:end]
        plt.plot(subset['milliseconds'], subset['MCP_no_filter'],
                 label='MCP (No Filter)', color='blue')
        plt.plot(subset['milliseconds'], subset['HX_no_filter'],
                 label='HX711 (No Filter)', color='orange')
        plt.title(f'Group {idx} - Comparison between MCP3564 and HX711')
        plt.xlabel('Milliseconds')
        plt.ylabel('Reading Value')
        plt.legend()
        plt.grid(True)
        plt.show()

def compute_energy_and_power(signal, time_interval):
    """
    Compute the energy and power for a given signal.
    
    Parameters:
    - signal: The force signal values.
    - time_interval: The time interval between consecutive samples.
    
    Returns:
    - total_energy: Total energy of the signal.
    - avg_power: Average power of the signal.
    """
    # Calculate the energy for each sample and sum them up to get total energy
    total_energy = np.sum(signal) * time_interval
    
    # Calculate average power
    avg_power = total_energy / (len(signal) * time_interval)
    
    return total_energy, avg_power

def identify_plot_and_extract_regions(subset, column_name, min_threshold=50):
    # Extract the data from the specified column
    signal_data = subset[column_name].values

    # Identify peaks using the find_peaks function
    peak_indices, _ = find_peaks(signal_data, height=100, distance=1000, prominence=100)

    # Plot with peaks
    plt.figure(figsize=(15, 10))
    plt.plot(subset['milliseconds'], signal_data, label=column_name)
    plt.scatter(subset['milliseconds'].iloc[peak_indices], signal_data[peak_indices], color='red', marker='o', s=50, label='Peaks')
    plt.xlabel('Milliseconds')
    plt.ylabel('Reading Value')
    plt.title(f'Identified Peaks in {column_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Extract regions around each peak
    peak_regions = []
    for peak in peak_indices:
        # Find previous minimum
        prev_min_idx = peak
        while prev_min_idx > 0 and signal_data[prev_min_idx] >= min_threshold:
            prev_min_idx -= 1
        
        # Find next minimum
        next_min_idx = peak
        while next_min_idx < len(signal_data) - 1 and signal_data[next_min_idx] >= min_threshold:
            next_min_idx += 1
        
        # Extract the region from the previous minimum to the next minimum
        peak_subset = subset.iloc[prev_min_idx:next_min_idx]
        peak_regions.append(peak_subset)

    return peak_indices, peak_regions

def plot_regions_with_absolute_time(region):
    """
    Plot the region with absolute time axis and return the lag delays of HX711 and HX711 filtered signals.
    """
    mcp_data = np.array(region['MCP_no_filter'])
    hx_data = np.array(region['HX_no_filter'])
    hx_filtered_data = np.array(region['HX_with_filter'])
    
    # Compute energy and power for the region for the three signals
    for signal_name, signal_data in [('MCP_no_filter', mcp_data), ('HX_no_filter', hx_data), ('HX_with_filter', hx_filtered_data)]:
        energy = np.sum(signal_data ** 2)
        power = energy / (region['milliseconds'].iloc[-1] - region['milliseconds'].iloc[0])
        print(f"Energy for {signal_name}: {energy:.2f} grams^2 x ms")
        print(f"Power for {signal_name}: {power:.2f} grams^2")
        print("--------------------------------------------")

    # Handle NaN or infinite values by replacing them with zero
    mcp_data[np.isnan(mcp_data) | np.isinf(mcp_data)] = 0
    hx_data[np.isnan(hx_data) | np.isinf(hx_data)] = 0
    hx_filtered_data[np.isnan(hx_filtered_data) | np.isinf(hx_filtered_data)] = 0

    # Create an absolute time axis starting from 0
    absolute_time = region['milliseconds'] - region['milliseconds'].iloc[0]

    # Find the time corresponding to the peak of each signal
    mcp_peak_time = absolute_time.iloc[np.argmax(mcp_data)]
    hx_peak_time = absolute_time.iloc[np.argmax(hx_data)]
    hx_filtered_peak_time = absolute_time.iloc[np.argmax(hx_filtered_data)]

    # Calculate the lag delays
    hx_lag_delay = hx_peak_time - mcp_peak_time
    hx_filtered_lag_delay = hx_filtered_peak_time - mcp_peak_time

    # Plot the signals
    plt.figure(figsize=(15, 10))
    plt.plot(absolute_time, mcp_data, label='MCP (No Filter)', color='blue')
    plt.plot(absolute_time, hx_data, label='HX711 (No Filter)', color='orange')
    plt.plot(absolute_time, hx_filtered_data, label='HX711 (With Filter)', color='green')
    plt.axvline(x=mcp_peak_time, color='blue', linestyle='--', label=f'MCP Peak')
    plt.axvline(x=hx_peak_time, color='orange', linestyle='--', label=f'HX711 Peak')
    plt.axvline(x=hx_filtered_peak_time, color='green', linestyle='--', label=f'HX711 Filtered Peak')
    plt.title(f'Comparison with Absolute Time Axis | HX711 Lag Delay: {hx_lag_delay:.2f} ms | HX711 Filtered Lag Delay: {hx_filtered_lag_delay:.2f} ms')
    plt.xlabel('Milliseconds (from start of region)')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return hx_lag_delay, hx_filtered_lag_delay

def plot_impacts(subset, impacts):
    """
    Plot the MCP_no_filter data for the group and highlight the detected impacts.
    """
    plt.figure(figsize=(15, 10))
    plt.plot(subset['milliseconds'], subset['MCP_no_filter'], label='MCP (No Filter)', color='blue')
    
    for start, end in impacts:
        plt.axvspan(subset['milliseconds'].iloc[start], subset['milliseconds'].iloc[end], color='red', alpha=0.2)
    
    plt.title('Detected Impacts')
    plt.xlabel('Milliseconds')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def identify_impacts(subset, rapid_rise_threshold=1000, stabilization_threshold=50, min_duration=200):
    """
    Identify impact events in the given subset of data.
    """
    impacts = []
    is_in_impact = False
    start_index = None
    for i, row in subset.iterrows():
        # Start of rapid rise
        if not is_in_impact and row['MCP_no_filter'] >= rapid_rise_threshold:
            is_in_impact = True
            start_index = i
        # End of stabilization
        elif is_in_impact and abs(row['MCP_no_filter']) <= stabilization_threshold:
            if i - start_index >= min_duration:
                impacts.append((start_index, i))
            is_in_impact = False
            start_index = None
    return impacts

def plot_selected_impacts(data, regions, selected_indices):
    """
    Plot the MCP, HX711, and HX711 filtered signals for the selected impact events.
    """
    for idx in selected_indices:
        region = regions[idx]
        
        plt.figure(figsize=(15, 10))
        plt.plot(region['milliseconds'], region['MCP_no_filter'], label='MCP (No Filter)', color='blue')
        plt.plot(region['milliseconds'], region['HX_no_filter'], label='HX711 (No Filter)', color='orange')
        plt.plot(region['milliseconds'], region['HX_with_filter'], label='HX711 (With Filter)', color='green', linestyle='--')
        plt.title(f'Impact {idx + 1}')
        plt.xlabel('Milliseconds')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
def extract_final_impact_signal(subset, peak_index, stabilization_threshold=20, stabilization_duration=10):
    """
    Extract the final oscillating impact signal based on stabilization criteria, compute its energy and power, and then plot it.
    
    Parameters:
    - subset: DataFrame containing the signal data.
    - peak_index: Index of the identified peak in the subset.
    - stabilization_threshold: Threshold value below which the signal is considered stabilizing.
    - stabilization_duration: Duration (in milliseconds) for which the signal must remain below the stabilization_threshold to be considered stabilized.
    
    Returns:
    - impact_signal: DataFrame containing the extracted oscillating impact signal.
    """
    signal_data = subset['MCP_no_filter'].values
    times = subset['milliseconds'].values
    
    # Find the start of the impact signal (first zero-crossing before the peak)
    start_idx = peak_index
    while start_idx > 0 and signal_data[start_idx] * signal_data[start_idx - 1] > 0:  # Check for zero-crossing
        start_idx -= 3

    # Find the end of the impact signal (where the signal stabilizes below stabilization_threshold for at least stabilization_duration)
    end_idx = peak_index
    while end_idx < len(signal_data) - 1:
        if abs(signal_data[end_idx]) < stabilization_threshold:
            # Check if the signal remains below the threshold for the required duration
            current_time = times[end_idx]
            while end_idx < len(signal_data) - 1 and abs(signal_data[end_idx]) < stabilization_threshold and (times[end_idx] - current_time) < stabilization_duration:
                end_idx += 1
            if (times[end_idx] - current_time) >= stabilization_duration:
                break
        end_idx += 15
        
    # Extract the final oscillating impact signal from start to end
    impact_signal = subset.iloc[start_idx:end_idx]
    
    # Adjust time to start from 0 for this impact
    impact_signal['milliseconds'] -= impact_signal['milliseconds'].iloc[0]
    
    # Compute energy and power for the three signals
    for signal in ['MCP_no_filter', 'HX_no_filter', 'HX_with_filter']:
        energy = np.sum(impact_signal[signal] ** 2)
        power = energy / (impact_signal['milliseconds'].iloc[-1] - impact_signal['milliseconds'].iloc[0])
        print(f"Energy for {signal}: {energy:.2f} grams^2 x ms")
        print(f"Power for {signal}: {power:.2f} grams^2")
        print("--------------------------------------------")
    
    
    
    # Plot the signal
    plt.figure(figsize=(15, 10))
    plt.plot(impact_signal['milliseconds'], impact_signal['MCP_no_filter'], label='MCP (No Filter)', color='blue')
    plt.plot(impact_signal['milliseconds'], impact_signal['HX_no_filter'], label='HX711 (No Filter)', color='orange')
    plt.plot(impact_signal['milliseconds'], impact_signal['HX_with_filter'], label='HX711 (With Filter)', linestyle='--', color='green')
    plt.title(f'Impact around {times[peak_index]} ms')
    plt.xlabel('Milliseconds from Impact Start')
    plt.ylabel('Signal Value (grams)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return impact_signal


# Load data
file_path = "dataCapture_2.csv"
data = load_data(file_path)

# Plot dataset comparison
plot_dataset_comparison(data)

# Identify and plot individual groups
groups = identify_groups(data)
plot_individual_groups(data, groups)

# Ask the user about the type of measurement (force or impact)
measurement_type = input("Enter the type of measurement (force or impact): ").strip().lower()

# Prompt user for which group they wish to analyze
group_indices_input = input("Enter the indices of the groups you want to analyze, separated by commas (e.g. '1,2,3'): ")
group_indices = [int(idx) - 1 for idx in group_indices_input.split(',')]

if measurement_type == "force":
    hx_lag_delays = []
    hx_filtered_lag_delays = []

    for idx in group_indices:
        subset = data.iloc[groups[idx][0]:groups[idx][1]]
        _, regions = identify_plot_and_extract_regions(subset, 'MCP_no_filter')
        
        # Prompt user for which region of the group they wish to analyze
        region_indices_input = input(f"Enter the indices of the regions for Group {idx + 1} you want to analyze, separated by commas (e.g. '1,2,3'): ")
        region_indices = [int(r_idx) - 1 for r_idx in region_indices_input.split(',')]
        
        # Plot each region with an absolute time axis and store lag delays
        for r_idx in region_indices:
            hx_delay, hx_filtered_delay = plot_regions_with_absolute_time(regions[r_idx])
            hx_lag_delays.append(hx_delay)
            hx_filtered_lag_delays.append(hx_filtered_delay)

        # Print average delay for HX711 and HX711 filtered signals after plotting all regions for a group
        print(f"Average lag delay for HX711 in Group {idx + 1}: {np.mean(hx_lag_delays):.2f} ms")
        print(f"Average lag delay for HX711 Filtered in Group {idx + 1}: {np.mean(hx_filtered_lag_delays):.2f} ms")

elif measurement_type == "impact":
    for idx in group_indices:
        subset = data.iloc[groups[idx][0]:groups[idx][1]]
        peak_indices, regions = identify_plot_and_extract_regions(subset, 'MCP_no_filter')
        
        # Prompt user to select impacts to plot
        selected_impacts_input = input(f"Enter the indices of the impacts you want to plot, separated by commas (e.g. '1,2,3'): ")
        selected_impacts = [int(impact_idx) - 1 for impact_idx in selected_impacts_input.split(',')]
        
        # Plot the selected impacts using the new function
        for s_idx in selected_impacts:
            impact_signal = extract_final_impact_signal(subset, peak_indices[s_idx])
else:
    print("Invalid measurement type entered.")