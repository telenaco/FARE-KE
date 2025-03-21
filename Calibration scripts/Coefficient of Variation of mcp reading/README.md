# MCP3564 ADC Noise Analysis

This repository contains data and analysis tools for evaluating the noise performance of the MCP3564 analog-to-digital converter at different oversampling ratios (OSR).

## Overview

The MCP3564 is a 24-bit delta-sigma ADC with programmable oversampling ratios. This project measures the ADC's noise performance by collecting digital readings at various OSR settings, then analyzing metrics like standard deviation, coefficient of variation, and signal-to-noise ratio (SNR).

## Repository Contents

- **Data Files**: Raw ADC readings at different OSR settings
  - `osr32.txt`: Data collected at OSR = 32
  - `osr64.txt`: Data collected at OSR = 64
  - `osr128.txt`: Data collected at OSR = 128
  - `osr256.txt`: Data collected at OSR = 256
  - `osr512.txt`: Data collected at OSR = 512
  - `osr1024.txt`: Data collected at OSR = 1024
  - `osr2048.txt`: Data collected at OSR = 2048
  - `osr4096.txt`: Data collected at OSR = 4096 (highest oversampling ratio)

- **Analysis Script**:
  - `Variation.py`: Python script for analyzing and visualizing the data

## Installation Requirements

To run the analysis script, you'll need:

- Python 3.x
- NumPy
- Matplotlib
- Regular Expressions (re) - built into Python

Install the required Python packages:

```bash
pip install numpy matplotlib
```

## Usage

1. Place all data files and the `Variation.py` script in the same directory

2. Run the analysis script:

```bash
python Variation.py
```

3. When prompted, press Enter to analyze files in the current directory, or specify a path to the directory containing the data files

## Analysis Output

The script performs the following analyses for each OSR data set:

- **Statistical Metrics**:
  - Mean (average) ADC reading
  - Standard deviation
  - Coefficient of Variation (CV)
  - Range (Max - Min)
  - Signal-to-Noise Ratio (SNR) in dB

- **Visualization**:
  - Overlaid plots of ADC noise for different OSR values
  - Signals are aligned to their means for easier comparison
  - Data lengths are scaled appropriately for visual comparison
  - Legend includes SNR values for each OSR setting

## Understanding the Results

Higher OSR values should generally result in:
- Lower noise (standard deviation)
- Higher SNR
- Better overall ADC performance

The trade-off is reduced sampling speed at higher OSR settings, as indicated by the smaller file sizes for higher OSR data files.

## Notes

- All data was collected on February 10, 2023
- Data was recorded with the ADC measuring approximately -0.02V input
- The MCP3564 is configured in differential measurement mode (DIFF_A)