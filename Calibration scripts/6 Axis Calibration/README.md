# 6-Axis Force/Torque Sensor Calibration Tools

This repository contains Python scripts and data files for calibrating and validating 6-axis force/torque sensors as part of the FARE-KE framework. These tools enable accurate measurement of forces and torques by compensating for cross-talk between channels.

## Contents

- [Overview](#overview)
- [Installation Requirements](#installation-requirements)
- [File Descriptions](#file-descriptions)
- [Usage Instructions](#usage-instructions)
  - [Basic Calibration](#basic-calibration)
  - [Advanced Calibration](#advanced-calibration)
  - [Validation and Error Analysis](#validation-and-error-analysis)
- [Calibration Process](#calibration-process)
- [Output Formats](#output-formats)

## Overview

The 6-axis force/torque sensor is constructed using multiple load cells arranged to measure forces and torques along three orthogonal axes. These scripts process raw sensor readings, compute calibration matrices, and validate the sensor performance through error analysis.

The calibration process compensates for cross-talk between measurement channels, improving measurement accuracy from raw sensor readings to calibrated force and torque values.

## Installation Requirements

The scripts require the following Python packages:
```
numpy
pandas
scipy
scikit-learn
matplotlib
```

Install these packages using pip:
```
pip install numpy pandas scipy scikit-learn matplotlib
```

## File Descriptions

### Python Scripts

- **6axisCalibration.py**: Basic calibration script that loads raw calibration data, computes the sensitivity matrix (K) and its inverse, and generates a markdown report with calibration results.

- **6axis.py**: Advanced calibration script implementing multiple calibration methods (single-point, averaging, optimized) and comparing their performance across different test points.

- **6axisReadingFromForceTorquePlateCalibration.py**: Comprehensive script for analyzing calibration data, with visualization capabilities for error distribution, scatter plots, and crosstalk analysis.

### Data Files

- **rawCalibrationData.csv**: Raw calibration matrix data used by the basic calibration script.

- **calibrationReadings.csv**, **calibrationReadings_1.csv**, **calibrationReadings_2.csv**: Extended datasets of raw sensor readings paired with known applied forces and torques for comprehensive calibration.

### Output Files

- **6axisCalibration.md**: Generated report containing the calibration matrix, its inverse, verification, and measurement compensation results.

- **crosstalk_errors.md**: Report showing crosstalk errors before and after calibration for the first 10 measurements.

## Usage Instructions

### Basic Calibration

To perform basic calibration:

1. Ensure your raw calibration data is prepared in the format of `rawCalibrationData.csv`
2. Run the basic calibration script:
   ```bash
   python 6axisCalibration.py
   ```
3. The script will generate `6axisCalibration.md` with the calibration results

### Advanced Calibration

For more sophisticated calibration comparing different methods:

1. Run the advanced calibration script:
   ```bash
   python 6axis.py
   ```
2. The script will output performance metrics for different calibration approaches directly to the console

### Validation and Error Analysis

To analyze and visualize calibration performance:

1. Ensure your calibration readings are prepared in the format of `calibrationReadings.csv`
2. Run the analysis script:
   ```bash
   python 6axisReadingFromForceTorquePlateCalibration.py
   ```
3. The script will generate visualizations and output performance metrics
4. A `crosstalk_errors.md` file will be created showing the crosstalk error comparison

## Calibration Process

The calibration process involves these key steps:

1. **Data Collection**: Gather raw sensor readings (mV/V) for known applied forces and torques
2. **Sensitivity Matrix Calculation**: Compute the K matrix relating sensor readings to physical forces/torques
3. **Inverse Computation**: Calculate K⁻¹ to convert raw readings to calibrated values
4. **Validation**: Apply K⁻¹ to test readings and compare with known values
5. **Error Analysis**: Analyze crosstalk, accuracy, and precision metrics

### Calibration Matrix Equation

The basic relationship is:
```
F = R × K⁻¹
```
Where:
- F: Calibrated force/torque vector [Fx, Fy, Fz, Mx, My, Mz]
- R: Raw sensor readings vector
- K⁻¹: Inverse of the calibration/sensitivity matrix

## Output Formats

### Sensitivity Matrix (K)

The K matrix represents the relationship between applied loads and sensor readings. Each row corresponds to a force or torque component, and each column represents a load cell channel.

### Inverse Matrix (K⁻¹)

The inverse matrix is used to convert raw readings to calibrated values, compensating for crosstalk between channels.

### Error Metrics

The scripts calculate multiple error metrics:
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- R² score
- Crosstalk magnitude

## Notes

- The calibration matrices are specific to each sensor and must be recalculated if the sensor configuration changes
- Temperature effects are not accounted for in these scripts
- For best results, collect calibration data across the full range of expected forces and torques