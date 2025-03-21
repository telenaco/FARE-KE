# MCP3564 vs HX711 Load Cell Analysis

Python tools for comparing MCP3564 and HX711 ADCs for load cell data acquisition, focusing on response time, signal quality, and performance characteristics.

## Files

### Data Files
- `dataCapture_2.csv` - Main data with columns for timestamps, MCP3564 (raw/filtered), and HX711 (raw/filtered)
- `dataCapture.csv` - Supplementary raw data

### Analysis Scripts
- `HX711vsMCP3564.py` - Latest version with Newton conversion and enhanced analysis
- `HX711vsMCP3564_v1.py` - Improved version with additional filtering capabilities
- `HX711vsMCP3564_v0.py` - Original script for basic comparison

### Configuration
- `dataCapture_2.txt` - Telemetry Viewer settings for visualization

## Requirements

- Python 3.6+
- Libraries: pandas, matplotlib, numpy, scipy

## Usage

1. Install required packages:
   ```
   pip install pandas matplotlib numpy scipy
   ```

2. Run the analysis script:
   ```
   python HX711vsMCP3564.py
   ```

3. When prompted:
   - Select measurement type (`force` or `impact`)
   - Choose groups to analyze (comma-separated indices)
   - Select specific regions within groups

## Analysis Features

### Force Measurement Analysis
- Identifies and segments measurement regions
- Calculates lag times between MCP3564 and HX711 signals
- Computes energy metrics and visualizes signal quality

### Impact Analysis
- Identifies discrete impact events
- Analyzes oscillation patterns and frequency response
- Calculates impulse and average force

## Key Findings

- MCP3564 provides faster response than HX711 (HX711 shows ~20.69ms lag)
- MCP3564 captures high-frequency components during impacts that HX711 misses
- Filtering further increases HX711 lag to ~32.54ms
- MCP3564 delivers higher resolution during rapid force changes

This comparison helps select the appropriate ADC based on application requirements, particularly for haptic applications where timing is critical.