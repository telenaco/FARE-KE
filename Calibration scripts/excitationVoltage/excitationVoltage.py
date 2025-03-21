import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_adc_data(hx711_file, mcp_file):
    # Load the CSV files
    hx711_data = pd.read_csv(hx711_file).iloc[1:].astype(float)
    mcp_data = pd.read_csv(mcp_file).iloc[1:].astype(float)

    # 1. Basic Statistics
    print("Basic Statistics:")
    for name, data in [("HX711", hx711_data), ("MCP3564", mcp_data)]:
        mean = data['Channel A'].mean()
        std = data['Channel A'].std()
        range_val = data['Channel A'].max() - data['Channel A'].min()
        print(f"\n{name} ADC:")
        print(f"Mean voltage: {mean:.4f} V")
        print(f"Standard deviation: {std:.4f} V")
        print(f"Range (Max - Min): {range_val:.4f} V")

    # 2. Visual Inspection
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(hx711_data['Time'], hx711_data['Channel A'], label='HX711', color='blue')
    plt.title('Voltage Readings from HX711 ADC')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(mcp_data['Time'], mcp_data['Channel A'], label='MCP3564', color='green')
    plt.title('Voltage Readings from MCP3564 ADC')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Stability Analysis
    print("\nStability Analysis:")
    for name, data in [("HX711", hx711_data), ("MCP3564", mcp_data)]:
        diff_std = data['Channel A'].diff().dropna().std()
        print(f"{name} ADC - Standard deviation of differences: {diff_std:.4f} V")

    # 4. Frequency Analysis
    plt.figure(figsize=(15, 8))
    for idx, (name, data) in enumerate([("HX711", hx711_data), ("MCP3564", mcp_data)], 1):
        fft = np.fft.fft(data['Channel A'])
        freq = np.fft.fftfreq(len(fft), d=(data['Time'].iloc[1] - data['Time'].iloc[0]) * 1e-3)
        plt.subplot(2, 1, idx)
        plt.plot(freq, np.abs(fft), label=name)
        plt.title(f'Frequency Components from {name} ADC')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    hx711_file_path = 'hx71120231003-0001_01.csv'
    mcp_file_path = 'mcp20231003-0001_01.csv'
    analyze_adc_data(hx711_file_path, mcp_file_path)
