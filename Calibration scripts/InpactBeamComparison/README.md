# Load Cell Impact Testing Tools

This repository contains tools for testing and comparing different load cell sensors (CLZ635, DEGRAW, TAL220) for measuring impact forces.

## Contents

### Data Files
- `impactLoadCellcComparisonCLZ635.csv`: Impact test data for CLZ635 load cell
- `impactLoadCellcComparisonCLZ635_withDamping.csv`: Impact test data with damping for CLZ635 load cell
- `impactLoadCellcComparisonDegraw.csv`: Impact test data for DEGRAW load cell  
- `impactLoadCellcComparisonTAL220.csv`: Impact test data for TAL220 load cell

### Code Files
- `ImpactLoadCellComparison.py`: Python script for visualizing and comparing load cell data
- `SingleBeamCellCalibratedPrint.ino`: Arduino sketch for calibrated load cell measurements

### Documentation
- `updated-editing-guidelines_v2.md`: Guidelines for editing technical documentation
- `thesis-reference-guide.md`: Reference guide for haptic feedback research topics

## Using the Python Script

The `ImpactLoadCellComparison.py` script visualizes impact testing data for all three load cell types. To use:

1. Ensure all CSV files are in the same directory as the script
2. Install required dependencies:
   ```
   pip install pandas matplotlib
   ```
3. Run the script:
   ```
   python ImpactLoadCellComparison.py
   ```

The script generates two plots for each load cell:
- Full dataset plot
- Zoomed-in view of a specific time range (5.5050e8 to 5.5125e8 microseconds)

## Arduino Setup

The `SingleBeamCellCalibratedPrint.ino` sketch interfaces with load cells using an MCP356x ADC and applies calibration polynomials.

### Hardware Requirements
- Arduino or Teensy microcontroller
- MCP356x ADC board
- Load cell (CLZ635, DEGRAW, or TAL220)

### Configuration

1. Connect the hardware according to the pin definitions:
   - SDI: Pin 11
   - SDO: Pin 12
   - SCK: Pin 13
   - CS: Pin 2
   - IRQ: Pin 3
   - MCLK: Pin 0

2. Select the appropriate calibration by uncommenting the relevant line:
   ```cpp
   // mcpScale.setPolynomialCalibration(CHANNEL_A, -1.143e-13, 0.0003968, 261.8);    // Degraw calibration 
   // mcpScale.setPolynomialCalibration(CHANNEL_A, -2.923e-13, 0.0005596, 513.9);     // TAL220 calibration
   mcpScale.setPolynomialCalibration(CHANNEL_A, -5.538e-14, 0.0003295, -227.8);   // CLZ635 calibration
   ```

3. Upload the sketch to your Arduino/Teensy

4. Open Serial Monitor at 115200 baud to view calibrated force readings

### Data Format

The Arduino sketch outputs data in CSV format:
```
timestamp_microseconds, force_grams
```

## Load Cell Specifications

Three load cell models are compared in this repository:

1. **CLZ635**: Provides good impact response with polynomial calibration coefficients: -5.538e-14, 0.0003295, -227.8
2. **DEGRAW**: Mid-range load cell with calibration coefficients: -1.143e-13, 0.0003968, 261.8
3. **TAL220**: High-capacity load cell with calibration coefficients: -2.923e-13, 0.0005596, 513.9

## Data Format

The CSV files contain the following columns:
- Sample Number (sequential counter)
- UNIX Timestamp (milliseconds since 1970-01-01)
- time () - Time in microseconds
- reading () - Force reading in grams-force

## License

See the LICENSE file for details.