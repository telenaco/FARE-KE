# Beam Load Cell Comparison Tools

This repository contains a collection of scripts and data files for conducting comprehensive performance analysis and validation of three low-cost beam load cells: TAL220, Degraw, and CZL635.

## Overview

The Python analysis script `Beam Load cells comparison.py` performs multiple types of analysis on load cell data:

1. **Calibration Analysis**: Evaluates different calibration methods (zero offset, midpoint adjustment, zero reading exclusion)
2. **Comparison of Calibration Methods**: Tests single scaling factor, linear regression, and polynomial regression
3. **Noise Analysis**: Analyzes signal noise characteristics in the no-load condition
4. **Overnight Drift Analysis**: Examines long-term stability and temperature influence
5. **Repetition Load Testing**: Tests consistency during repeated loading/unloading cycles

## Data Files

- `Calibration values gmF.csv`: Weight-to-digital value calibration data for all three load cells
- `noLoad noise - connection 0 - UART COM3.csv`: Baseline noise measurements with no load
- `loading test overnight - connection 0 - UART COM3.csv`: Long-term drift test data with temperature
- `loadingTest - connection 0 - UART COM3.csv`: Repeated loading/unloading test data
- `dataFromTemperatureSignalB - connection 0 - UART COM3.csv`: Temperature signal analysis data
- `temp_tal220.csv`, `temp_degraw.csv`, `temp_CZL635.csv`: Individual calibration files

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - matplotlib
  - numpy
  - scikit-learn
  - seaborn
  - scipy
  - Pillow (PIL)

Install required packages:
```bash
pip install pandas matplotlib numpy scikit-learn seaborn scipy Pillow
```

## Usage Instructions

### Running the Full Analysis

To run the complete analysis:

```bash
python "Beam Load cells comparison.py"
```

This will generate multiple comparative plots and output performance metrics for each load cell.

### Key Analysis Components

The script is organized into functional sections:

#### 1. Calibration Analysis

Evaluates three calibration approaches:
- Zero Offset: Adjusts readings based on true zero
- Midpoint Zero Adjustment: Uses average of zero readings
- Ignoring Zero Readings: Excludes zero points from calibration

```python
# Relevant code sections
true_zero_data = data_df.copy()
for load_cell in ["tal220", "degraw", "CZL635"]:
    true_zero_data[load_cell] -= true_zero[load_cell]
linear_analysis(true_zero_data, "Zero Offset")
```

#### 2. Calibration Method Comparison

Compares three calibration techniques:
- Single Scaling Factor: Simple multiplier
- Linear Regression: First-order polynomial fit
- Polynomial Regression: Second-order polynomial fit

```python
# Call this for each load cell
results = calibration_analysis(sub_data, cell, filename)
```

#### 3. Noise Analysis

Analyzes baseline noise characteristics:
- Statistical metrics (standard deviation, peak-to-peak)
- Signal-to-noise ratio calculation
- Frequency domain analysis (FFT)
- Autocorrelation analysis

```python
# Analyzing noise data for each load cell
readings = noise_data[noise_cell] - noise_data[noise_cell].mean()
```

#### 4. Drift Analysis

Examines long-term stability and temperature influence:
- Smoothed signal tracking over time
- Temperature correlation analysis
- Drift calculation relative to initial values

```python
# Processing drift data
data['635_drift_g'] = data['635_smoothed_g'] - data['635_smoothed_g'].iloc[window_size-1]
```

#### 5. Repetition Testing

Analyzes response consistency during repeated loading/unloading cycles:
- Standardizing event lengths
- Calculating average responses
- Comparing repeatability between load cells

```python
# Standardizing load/unload events
data_debug_updated, new_timestamps_debug_updated = standardize_load_cell_data_debug_updated(load_cell_data, loading_intervals, unloading_intervals, timestamps)
```

## Output

The script generates several plots:
- Calibration plots with different adjustment methods
- Comparison of calibration techniques
- Noise distribution, frequency spectrum, and autocorrelation plots
- Temperature and drift correlation graphs
- Repetition test visualizations

Performance metrics are printed to the console, including:
- Linearity and accuracy percentages
- Calibration equations for each method
- Noise levels in grams-force (gf)
- Temperature sensitivity coefficients
- Repeatability statistics

## Data Collection Setup

The data was collected using:
- Three types of beam load cells (TAL220, Degraw, CZL635)
- Weights ranging from 0-2300 grams
- Data acquisition through a UART interface (COM3)
- Sampling rate of approximately 1000 samples per second

## Known Limitations

- The analysis assumes stable environmental conditions except where temperature is explicitly measured
- The calibration is valid within the tested weight range (0-2300g)
- Sampling rate limitations may affect dynamic response analysis

## Citation

If using this data or analysis in academic work, please cite as:
```
FARE-KE: Framework for Affordable, Reliable Kinesthetic Evaluation (2025)
```