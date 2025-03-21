import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_digital_readings(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Extract the digital readings using a regular expression
    digital_readings = re.findall(r'digital reading -> ([-\d]+)', content)
    return [int(val) for val in digital_readings]

def calculate_metrics(values):
    mean_val = np.mean(values)
    std_dev = np.std(values)
    cv = (std_dev / mean_val) * 100
    range_val = np.max(values) - np.min(values)
    return mean_val, std_dev, cv, range_val

def calculate_snr(values):
    if len(values) == 0:
        return float('nan')
    signal_power = np.mean(values)**2
    noise_power = np.var(values)
    if noise_power == 0:
        return float('inf')
    snr_linear = signal_power / noise_power
    return 10 * np.log10(snr_linear)

def print_metrics(values, osr):
    mean_val, std_dev, cv, range_val = calculate_metrics(values)
    snr = calculate_snr(values)
    print(f"\nAnalysis for OSR {osr}:")
    print(f"Mean (Average) ADC Reading: {mean_val:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Coefficient of Variation (CV): {cv:.2f}%")
    print(f"Range (Max - Min): {range_val}")
    print(f"SNR: {snr:.2f} dB")

def plot_layered_signals(directory="."):
    # Get all .txt files from the specified directory
    file_list = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    data = []

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        values = extract_digital_readings(file_path)
        
        if len(values) == 0:
            print(f"WARNING: No readings extracted from {file_name}. Skipping.")
            continue

        osr = int(re.search(r'osr(\d+)', file_name, re.IGNORECASE).group(1))
        print_metrics(values, osr)
        
        snr = calculate_snr(values)
        
        data.append((values, osr, snr))
    
    # Sort data by OSR value
    data.sort(key=lambda x: x[1], reverse=False)
    
    # Determine relative lengths
    length_reference = len([d for d in data if d[1] == 1024][0][0])
    scaling_factors = {
        4096: 1/3,
        2048: 2/3,
        1024: 1,
        512: 4/3,
        256: 5/3,
        128: 2,
        64: 7/3,
        32: 8/3
    }

    plt.figure(figsize=(15, 8))

    for values, osr, snr in data:
        scaled_length = int(length_reference * scaling_factors[osr])
        if len(values) > scaled_length:
            values = values[:scaled_length]
        plt.plot(values - np.mean(values), label=f'OSR {osr} (SNR: {snr:.2f} dB)')

    plt.title("Layered Signals Aligned to Mean and Adjusted for Length")
    plt.xlabel("Sample Index")
    plt.ylabel("ADC Value (Aligned to Mean)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    directory = input("Enter the directory path (or press Enter for current directory): ")
    plot_layered_signals(directory if directory else ".")
