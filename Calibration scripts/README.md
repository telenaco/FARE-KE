# Load Cell Data Acquisition and Analysis Tools

This repository contains tools for load cell calibration, testing, and data acquisition across various hardware configurations.

## Folder Contents

- **3 Axis Calibration**: Calibration scripts for 3-axis load cells with cross-axis sensitivity compensation
- **6 Axis Calibration**: Tools for 6-axis force/torque sensor calibration and validation
- **ADC OSR comparison**: Analysis of MCP3564 ADC performance at different oversampling ratios
- **Coefficient of Variation of mcp reading**: Noise analysis for MCP3564 ADC
- **Comparison Beam load cells**: Performance comparison between TAL220, Degraw, and CZL635 beam load cells
- **excitationVoltage**: Stability analysis comparing HX711 and MCP3564 excitation voltages
- **hx711 vs Mcp**: Performance comparison between HX711 and MCP3564 ADCs
- **hx711 vs mcp timing**: Sampling rate and timing consistency analysis
- **ImpactBeamComparison**: Load cell performance during impact force measurement
- **SingleBeamCalibration**: Single-axis beam load cell calibration using various methods

## Requirements

Most analysis scripts require Python 3.6+ with libraries including:
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

Refer to individual folder READMEs for specific setup instructions.