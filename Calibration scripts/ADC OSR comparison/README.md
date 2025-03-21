# FARE-KE Sampling Rate Analysis

This repository contains scripts and data for analysing microcontroller sampling rate consistency as part of the FARE-KE framework for haptic device characterisation. The tools help evaluate data acquisition performance under different configurations.

## Overview

The sampling rate analysis evaluates the consistency and reliability of time intervals between consecutive sensor readings from an ADC system. This is crucial for haptic device characterisation, where precise timing is essential for accurate force and torque measurements.

## Contents

- `SamplingRateReadings.py`: Python script for analysing and visualising sampling interval data
- CSV files with different sampling configurations:
  - `OSR_32_single_message.csv`: Data using OverSampling Rate (OSR) of 32 with basic message transmission
  - `OSR_32_single_message_withFlush.csv`: Data using OSR 32 with serial buffer flush
  - `OSR_32_using_buffer.csv`: Data using OSR 32 with data buffering
  - `OSR_32_single_message_withFlush_timeDelay.csv`: Data using OSR 32 with flush and time delay

## Requirements

- Python 3.x
- Required packages:
  - pandas
  - matplotlib

You can install the required packages using:

```bash
pip install pandas matplotlib
```

## Usage

1. Ensure all CSV files are in the same directory as the Python script.
2. Run the script:

```bash
python SamplingRateReadings.py
```

3. The script will:
   - Load and preprocess each CSV file
   - Calculate key metrics (average reading time, standard deviation, number of spikes)
   - Generate plots showing the time intervals between readings
   - Print summary statistics for each configuration

## Key Metrics Explained

- **Average Reading Time**: Mean time interval between consecutive readings (in microseconds)
- **Standard Deviation**: Measure of consistency in reading intervals
- **Number of Spikes**: Count of intervals exceeding 50,000 microseconds (indicating potential issues)

## Data Format

The CSV files contain data in the following format:
- `micros`: Timestamp in microseconds
- `raw`: Raw ADC reading
- `butterworth`: Filtered reading using Butterworth filter

## Script Features

- **Robust Data Loading**: Handles corrupted CSV files by detecting and removing problematic lines
- **Data Preprocessing**: Converts data types and calculates time differentials
- **Warm-up Period Removal**: Discards the first 2000 samples to eliminate initialisation effects
- **Visualisation**: Generates time interval plots with standard deviation indicators

## Use Cases

This analysis helps:
1. Compare different ADC and microcontroller communication methods
2. Identify optimal configurations for specific haptic modality measurements
3. Troubleshoot timing inconsistencies in data acquisition systems
4. Evaluate the impact of buffer management on sampling performance

## Integration with FARE-KE Framework

This sampling rate analysis is part of the FARE-KE framework's Data Acquisition validation process, which ensures reliable measurements for haptic device characterisation. The results from this analysis inform the selection of appropriate sampling configurations for various haptic modalities.

## Known Issues

- The script assumes consistent CSV formatting across files
- Large CSV files may require significant memory resources
- Some configurations may show periodic timing spikes related to OS scheduling or USB communication

## Future Improvements

- Add command-line arguments for custom file selection
- Implement frequency domain analysis of sampling intervals
- Add support for real-time data acquisition and analysis
- Expand visualisation options including histogram views