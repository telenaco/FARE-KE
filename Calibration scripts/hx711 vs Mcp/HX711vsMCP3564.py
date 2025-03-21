import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import numpy as np
from scipy.signal import correlate

def grams_to_newtons(grams):
    return grams * 0.00981

def load_data(file_path):
    data = pd.read_csv(file_path, header=None, names=[
        'microseconds', 'MCP_no_filter', 'MCP_with_filter', 'HX_no_filter', 'HX_with_filter'])
    
    # Convert to milliseconds
    data['milliseconds'] = data['microseconds'] / 1000
    
    # Compute newton values and add as new columns
    data['MCP_no_filter_newtons'] = grams_to_newtons(data['MCP_no_filter'])
    data['MCP_with_filter_newtons'] = grams_to_newtons(data['MCP_with_filter'])
    data['HX_no_filter_newtons'] = grams_to_newtons(data['HX_no_filter'])
    data['HX_with_filter_newtons'] = grams_to_newtons(data['HX_with_filter'])
    
    return data

def plot_dataset_comparison(data):
    plt.figure(figsize=(15, 10))
    
    # Plot using newton values
    plt.plot(data['milliseconds'], data['MCP_no_filter_newtons'],
             label='MCP (No Filter)', color='blue')
    plt.plot(data['milliseconds'], data['MCP_with_filter_newtons'],
             label='MCP (With Filter)', color='blue', linestyle='--')
    plt.plot(data['milliseconds'], data['HX_no_filter_newtons'],
             label='HX711 (No Filter)', color='orange')
    plt.plot(data['milliseconds'], data['HX_with_filter_newtons'],
             label='HX711 (With Filter)', color='orange', linestyle='--')
    
    plt.title('Comparison between MCP3564, HX711 and their Filtered Versions (in Newtons)')
    plt.xlabel('Milliseconds')
    plt.ylabel('Reading Value (Newtons)')
    plt.legend()
    plt.grid(True)
    plt.show()


def identify_groups(data, spike_threshold=7.8, zero_threshold=0.49, zero_duration=2000, trail_duration=2000):
    """
    Identify groups of data points based on given criteria.
    
    Parameters:
    - data: DataFrame containing the data
    - spike_threshold: threshold for identifying spikes (in Newtons)
    - zero_threshold: threshold for identifying zero values (in Newtons)
    - zero_duration: duration to consider as a zero group (in milliseconds)
    - trail_duration: trailing duration after the zero group (in milliseconds)
    
    Returns:
    - groups: a list of tuples (start_index, end_index) representing the groups
    """
    groups = []
    is_in_group = False
    start_index = None
    zero_counter = 0
    for i, row in data.iterrows():
        if abs(row['MCP_no_filter_newtons']) <= zero_threshold:
            zero_counter += (data['milliseconds'].iloc[i] -
                             data['milliseconds'].iloc[i-1])
        else:
            zero_counter = 0

        if not is_in_group and row['MCP_no_filter_newtons'] > spike_threshold:
            is_in_group = True
        elif is_in_group and abs(row['MCP_no_filter_newtons']) <= zero_threshold and start_index is None:
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
    """
    Plot individual groups of data points.
    
    Parameters:
    - data: DataFrame containing the data
    - groups: a list of tuples (start_index, end_index) representing the groups
    """
    for idx, (start, end) in enumerate(groups, 1):
        plt.figure(figsize=(10, 6))
        subset = data.iloc[start:end]
        plt.plot(subset['milliseconds'], subset['MCP_no_filter_newtons'],
                 label='MCP (No Filter)', color='blue')
        plt.plot(subset['milliseconds'], subset['HX_no_filter_newtons'],
                 label='HX711 (No Filter)', color='orange')
        plt.title(f'Group {idx} - Comparison between MCP3564 and HX711 (in Newtons)')
        plt.xlabel('Milliseconds')
        plt.ylabel('Reading Value (Newtons)')
        plt.legend()
        plt.grid(True)
        plt.show()

def compute_energy_and_power_corrected(signal, time_interval):
    # Calculate the energy for each sample (square the signal values) and sum them up to get total energy
    total_energy = np.sum(signal ** 2) * time_interval
    # Calculate average power
    avg_power = total_energy / (len(signal) * time_interval)
    return total_energy, avg_power

def identify_plot_and_extract_regions(subset, column_name, min_threshold=0.49):
    signal_data = subset[column_name + "_newtons"].values
    peak_indices, _ = find_peaks(signal_data, height=0.98, distance=1000, prominence=0.98)
    
    plt.figure(figsize=(15, 10))
    plt.plot(subset['milliseconds'], signal_data, label=column_name + " (Newtons)")
    plt.scatter(subset['milliseconds'].iloc[peak_indices], signal_data[peak_indices], color='red', marker='o', s=50, label='Peaks')
    plt.xlabel('Milliseconds')
    plt.ylabel('Reading Value (Newtons)')
    plt.title(f'Identified Peaks in {column_name} (Newtons)')
    plt.grid(True)
    plt.legend()
    plt.show()

    peak_regions = []
    for peak in peak_indices:
        prev_min_idx = peak
        while prev_min_idx > 0 and signal_data[prev_min_idx] >= min_threshold:
            prev_min_idx -= 1
        next_min_idx = peak
        while next_min_idx < len(signal_data) - 1 and signal_data[next_min_idx] >= min_threshold:
            next_min_idx += 1
        peak_subset = subset.iloc[prev_min_idx:next_min_idx]
        peak_regions.append(peak_subset)

    return peak_indices, peak_regions


def plot_regions_with_absolute_time(region):
    mcp_data = np.array(region['MCP_no_filter_newtons'])
    hx_data = np.array(region['HX_no_filter_newtons'])
    hx_filtered_data = np.array(region['HX_with_filter_newtons'])
    
    for signal_name, signal_data in [('MCP_no_filter', mcp_data), ('HX_no_filter', hx_data), ('HX_with_filter', hx_filtered_data)]:
        energy = np.sum(signal_data ** 2)
        power = energy / (region['milliseconds'].iloc[-1] - region['milliseconds'].iloc[0])
        print(f"Energy for {signal_name}: {energy:.2f} N^2 x ms")
        print(f"Power for {signal_name}: {power:.2f} N^2")
        print("--------------------------------------------")

    mcp_data[np.isnan(mcp_data) | np.isinf(mcp_data)] = 0
    hx_data[np.isnan(hx_data) | np.isinf(hx_data)] = 0
    hx_filtered_data[np.isnan(hx_filtered_data) | np.isinf(hx_filtered_data)] = 0

    absolute_time = region['milliseconds'] - region['milliseconds'].iloc[0]
    mcp_peak_time = absolute_time.iloc[np.argmax(mcp_data)]
    hx_peak_time = absolute_time.iloc[np.argmax(hx_data)]
    hx_filtered_peak_time = absolute_time.iloc[np.argmax(hx_filtered_data)]
    hx_lag_delay = hx_peak_time - mcp_peak_time
    hx_filtered_lag_delay = hx_filtered_peak_time - mcp_peak_time

    plt.figure(figsize=(15, 10))
    plt.plot(absolute_time, mcp_data, label='MCP (No Filter)', color='blue')
    plt.plot(absolute_time, hx_data, label='HX711 (No Filter)', color='orange')
    plt.plot(absolute_time, hx_filtered_data, label='HX711 (With Filter)', color='green')
    plt.axvline(x=mcp_peak_time, color='blue', linestyle='--', label=f'MCP Peak')
    plt.axvline(x=hx_peak_time, color='orange', linestyle='--', label=f'HX711 Peak')
    plt.axvline(x=hx_filtered_peak_time, color='green', linestyle='--', label=f'HX711 Filtered Peak')
    plt.title(f'Comparison with Absolute Time Axis | HX711 Lag Delay: {hx_lag_delay:.2f} ms | HX711 Filtered Lag Delay: {hx_filtered_lag_delay:.2f} ms')
    plt.xlabel('Milliseconds (from start of region)')
    plt.ylabel('Signal Value (Newtons)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return hx_lag_delay, hx_filtered_lag_delay
       
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