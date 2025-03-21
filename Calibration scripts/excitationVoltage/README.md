# ADC Performance Analysis Toolkit

This repository contains tools for analyzing and comparing the performance of different Analog-to-Digital Converters (ADCs) used in the FARE-KE framework for haptic device characterization. It specifically focuses on comparing the HX711 and MCP3564 ADCs in terms of excitation voltage stability and noise characteristics.

## Repository Contents

- **Python Scripts**
  - `excitationVoltage.py`: Script for analyzing voltage stability data from both ADC types

- **Data Files**
  - `hx711Data_01.csv`: Voltage readings from HX711 ADC 
  - `hx711202310030001_01.csv`: Timestamped voltage measurements from HX711 ADC
  - `mcpData_01.csv`: Voltage readings from MCP3564 ADC
  - `mcp202310030001_01.csv`: Timestamped voltage measurements from MCP3564 ADC

- **Documentation**
  - `updated-editing-guidelines_v2.md`: Guidelines for standardizing documentation in UK English
  - `thesis-reference-guide.md`: Comprehensive reference guide for the thesis structure and content

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - matplotlib

You can install the required packages using:

```bash
pip install pandas numpy matplotlib
```

### Using the Analysis Script

The `excitationVoltage.py` script compares the performance of HX711 and MCP3564 ADCs by analyzing their voltage stability. To run the analysis:

1. Ensure the CSV data files are in the same directory as the script
2. Run the script with Python:

```bash
python excitationVoltage.py
```

By default, the script analyzes `hx71120231003-0001_01.csv` and `mcp20231003-0001_01.csv`. If you need to analyze different files, you can modify the file paths at the end of the script:

```python
if __name__ == "__main__":
    hx711_file_path = 'your_hx711_file.csv'
    mcp_file_path = 'your_mcp_file.csv'
    analyze_adc_data(hx711_file_path, mcp_file_path)
```

### Analysis Output

The script provides:

1. **Basic Statistics**: Mean voltage, standard deviation, and range for each ADC
2. **Voltage Plots**: Time-series visualization of voltage readings
3. **Stability Analysis**: Standard deviation of consecutive differences to measure voltage stability
4. **Frequency Analysis**: FFT plots showing frequency components in the voltage signals

## Data File Format

The CSV data files should have the following format:
- Column 1: "Time" (in milliseconds)
- Column 2: "Channel A" (voltage readings)

The first row is assumed to be headers and is skipped during analysis.

## Context

This toolkit is part of the FARE-KE (Framework for Affordable, Reliable Kinesthetic Evaluation) project, which aims to standardize haptic device characterization. The specific focus of these scripts is to evaluate and compare the stability of different ADC solutions for load cell measurements in haptic research.

The analysis helps understand the strengths and limitations of each ADC option (HX711 vs. MCP3564) in terms of:
- Voltage stability
- Noise characteristics
- Frequency response
- Overall measurement quality
