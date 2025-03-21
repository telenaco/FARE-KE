# Load Cell Calibration Toolkit

This repository contains tools and resources for calibrating and analyzing single-axis beam load cells used in haptic device characterization.

## Contents

- `Beam_loadCell_calibration.py` - Python script for calibrating single-axis load cells using three different methods
- `calibration_results.md` - Markdown file containing the results of the calibration analysis
- `calibration_comparison.png` - Visualization of the different calibration techniques
- `calibrationWeightSingleBeamHX711.csv` - Example dataset from HX711 amplifier
- `calibrationWeightSingleBeamMCP.csv` - Example dataset from MCP3564 amplifier

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Usage

### 1. Data Collection

Before running the calibration script, you need to collect calibration data:

1. Apply known weights to your load cell in ascending and descending order
2. Record the raw sensor readings for each weight
3. Save the data in a CSV file with two columns: 'Weight' and 'Reading'

Example CSV format:
```
Weight,Reading
0,15511
100,174290
200,346926
...
```

### 2. Running the Calibration Script

```bash
python Beam_loadCell_calibration.py
```

By default, the script uses `calibrationWeightSingleBeamMCP.csv`. To use a different dataset, modify the file path in the script.

### 3. Calibration Methods

The script implements three calibration techniques:

1. **Linear Regression** - Fits a linear equation (y = mx + b) to the data
2. **Polynomial Regression** - Fits a second-degree polynomial equation (y = ax² + bx + c)
3. **Scaling Factor** - Calculates an average scaling factor between weight and readings

### 4. Analyzing Results

After running the script, examine `calibration_results.md` to:
- Compare R² values to determine which method provides the best fit
- Use the provided regression equations to convert raw sensor readings to actual force values
- View the visual comparison of different calibration methods

## Selecting the Best Calibration Method

- **Linear Regression**: Best for sensors with a linear response curve. Simple to implement.
- **Polynomial Regression**: Better for sensors that exhibit slight non-linearity.
- **Scaling Factor**: Simplest approach but may be less accurate if the sensor response is not perfectly linear.

## Example Output

The calibration script generates:

1. A visualization comparing all three methods
2. Statistical analysis including R² values
3. Regression equations for converting raw readings to calibrated values

## Troubleshooting

- Ensure your CSV file has correct column names ('Weight' and 'Reading')
- If you encounter errors about missing offset values, check that your dataset includes a reading for 0 weight
- For optimal results, use at least 10 calibration points spanning the entire range of your load cell

## Extending the Script

To modify the script for your specific needs:
- Change the polynomial degree by modifying the `degree` parameter in `PolynomialFeatures`
- Add additional calibration methods by implementing them before the plotting section
- Modify the plotting code to customize the visualization