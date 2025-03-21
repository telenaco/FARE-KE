# 3-Axis Load Cell Calibration Tools

This repository contains calibration tools and datasets for the FARE-KE framework's 3-axis load cell system. These tools enable accurate force measurement through advanced calibration techniques that compensate for cross-axis sensitivity.

## Contents

- [Overview](#overview)
- [Technical Requirements](#technical-requirements)
- [Repository Structure](#repository-structure)
- [Calibration Methodologies](#calibration-methodologies)
- [Performance Metrics](#performance-metrics)
- [Implementation Guide](#implementation-guide)

## Overview

The 3-axis load cell system measures forces along three orthogonal axes using strain gauge sensors. These scripts process calibration data, implement multiple calibration algorithms, and evaluate measurement accuracy through comprehensive error analysis.

The calibration techniques account for cross-axis sensitivity, non-linear response characteristics, and sensor offset, resulting in significantly improved force measurement accuracy.

## Technical Requirements

Required Python packages:
```
numpy
pandas
sklearn.metrics
matplotlib
```

Install via pip:
```
pip install numpy pandas scikit-learn matplotlib
```

## Repository Structure

### Calibration Datasets

- **calibrationData.xlsx**: Master dataset containing raw measurement data from all load cells.

- **3axisCell1CalibrationData.csv** to **3axisCell4CalibrationData.csv**: Individual calibration datasets for four 3-axis load cell units, each containing:
  - Applied forces (grams) along X, Y, Z axes
  - Raw sensor readings
  - Offset-corrected values
  - Orientation data
  - Compensation parameters

- **validationDataCell2.csv** and **validationDataCell2.xls**: Independent validation datasets containing measurements not used in calibration, critical for verification of calibration accuracy.

### Calibration Algorithms

- **3axisCalibrationData.py**: Primary calibration implementation:
  - Loads and processes calibration data
  - Performs zero-load offset correction
  - Implements two calibration algorithms:
    1. Least Squares method (matrix-based calibration)
    2. Polynomial Regression (second-degree polynomial fitting)
  - Calculates comprehensive error metrics
  - Exports detailed results to Markdown format
  - Validates calibration using independent datasets

- **3axisCalibrationDataComparisonwithSingle.py**: Extended algorithm comparing three calibration approaches:
  - Least Squares calibration
  - Polynomial Regression
  - Single-point calibration
  - Includes visualization functionality for error analysis

### Results Documentation

- **3axisCell[1-4]CalibrationData_log.md**: Generated calibration reports containing:
  - Tabulated force estimations for each calibration method
  - Absolute and percentage errors
  - Statistical performance metrics
  - Calibration matrices and equations for implementation

- **3axisCalibrationData_README.md**: Technical documentation covering:
  - Calibration tool functionality
  - Data format specifications
  - Implementation procedures
  - Output interpretation guide

## Calibration Methodologies

### 1. Least Squares Method

Matrix-based calibration using linear least squares regression to determine the optimal transformation matrix between sensor readings and applied forces. This method accounts for cross-axis sensitivity through a 3×3 calibration matrix:

```
F = R × K
```
Where:
- F: Force vector [Fx, Fy, Fz]
- R: Offset-corrected sensor readings
- K: Calibration matrix

Implementation steps:
1. Collect paired data of known forces and sensor readings
2. Subtract zero-load offset from sensor readings
3. Calculate transformation matrix using least squares algorithm
4. Apply matrix to transform new readings into calibrated forces

### 2. Polynomial Regression

Non-linear calibration fitting second-degree polynomials to each axis, capturing non-linear sensor response characteristics:

```
F_axis = a₀ + a₁R_axis + a₂R_axis²
```

Where:
- F_axis: Force along specific axis
- R_axis: Sensor reading for that axis
- a₀, a₁, a₂: Polynomial coefficients

This method handles non-linearities in sensor response but does not account for cross-axis sensitivity as comprehensively as the matrix approach.

### 3. Single-Point Calibration

Simplified calibration using a single inverse transformation matrix, suitable for rapid deployment with reduced accuracy:

```
F = R × K⁻¹
```

Where K⁻¹ is predetermined through a simplified calibration process.

## Performance Metrics

Each calibration method is evaluated using:

- **Mean Absolute Error (MAE)**: Average absolute difference between measured and true forces
- **Root Mean Square Error (RMSE)**: Square root of the average of squared errors
- **Maximum Error**: Largest observed error across validation datasets
- **Error Percentage**: Error relative to applied load

Comparative analysis shows:
1. Least Squares method typically provides 3-6g average error across the measurement range
2. Polynomial Regression achieves 3-5g average error with improved performance at higher loads
3. Single-Point calibration yields 5-8g average error with simplified implementation

## Implementation Guide

### Basic Calibration Process

1. Mount the load cell in a stable fixture
2. Apply known forces along each axis independently
3. Record sensor readings for each applied force
4. Format data according to CSV structure
5. Run calibration script:
   ```
   python 3axisCalibrationData.py
   ```
6. Extract calibration matrix from generated log file
7. Implement matrix in force measurement system

### Calibration Validation

1. Apply new set of known forces not used in calibration
2. Record sensor readings
3. Convert readings to forces using calibration matrix
4. Calculate errors between measured and applied forces
5. Verify errors are within acceptable limits

### Method Selection Guidance

- Use **Least Squares** for general applications requiring balance of accuracy and simplicity
- Apply **Polynomial Regression** when sensor exhibits significant non-linear behavior
- Employ **Single-Point** calibration for preliminary testing or applications with lower precision requirements

## Limitations

- Temperature sensitivity not addressed in current implementation
- Calibration valid only within tested force range
- Temporal drift effects require periodic recalibration
- Cross-axis sensitivity may not be fully eliminated

## References

Implementation based on methodologies described in Chapter 5 of FARE-KE framework.