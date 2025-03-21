"""
Filename: Beam_loadCell_calibration.py

Description:
    This script performs calibration techniques on sensor readings from a single-axis load cell to obtain
    a more accurate representation of actual weights. Three calibration techniques are implemented:
    Linear Regression, Polynomial Regression, and Scaling Factor Calibration. The script visualizes the
    results and provides the regression equations for each technique. The output is saved in a Markdown file for easy review.

Author: [Your Name]
Date: [Date of Creation]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('calibrationWeightSingleBeamMCP.csv')
x = data['Weight'].values.reshape(-1, 1)
y = data['Reading'].values

# Find the offset by finding the reading corresponding to zero weight
offset = data.loc[data['Weight'] == 0, 'Reading'].values[0]

# Apply offset to the readings
y_corrected = y - offset

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(x, y_corrected)
y_pred_linear = linear_regressor.predict(x)

# Polynomial Regression (2nd Degree)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly_regressor = LinearRegression()
poly_regressor.fit(x_poly, y_corrected)
y_pred_poly = poly_regressor.predict(x_poly)

# Scaling Factor Calibration
# Exclude zero readings to prevent division by zero
non_zero_mask = x.flatten() != 0
scaling_factors = y_corrected[non_zero_mask] / x[non_zero_mask].flatten()
average_scaling_factor = np.mean(scaling_factors)
y_pred_scaling = x.flatten() * average_scaling_factor

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y_corrected, color='red', label='Actual Data Points', marker='o')
plt.plot(x, y_pred_linear, color='blue', label='Linear Regression', linestyle='-', marker='>')
plt.plot(x, y_pred_poly, color='green', label='Polynomial Regression', linestyle=':', marker='s')
plt.scatter(x[non_zero_mask], y_pred_scaling[non_zero_mask], color='orange', label='Scaling Factor Calibration', marker='^', zorder=5)
plt.axhline(0, color='black', linewidth=0.5)  # Add a line at zero for reference
plt.title('Calibration Techniques Comparison')
plt.xlabel('Weight')
plt.ylabel('Corrected Reading')
plt.legend()
plt.savefig("calibration_comparison.png")
plt.close()

r2_linear = r2_score(y_corrected, y_pred_linear)
r2_poly = r2_score(y_corrected, y_pred_poly)
r2_scaling = r2_score(y_corrected[non_zero_mask], y_pred_scaling[non_zero_mask])


# Prepare Markdown content
md_content = f"""
# Calibration Results for Single-Axis Load Cell

This document presents the results of applying three different calibration techniques to sensor readings obtained from a single-axis load cell.

## Calibration Techniques
- **Linear Regression**
- **Polynomial Regression**
- **Scaling Factor Calibration**

![Calibration Techniques Comparison](calibration_comparison.png)

## Statistical Analysis

| Technique |  R^2 |
| --------- |  --- |
| Linear Regression | {r2_linear:.4f} |
| Polynomial Regression | {r2_poly:.4f} |
| Scaling Factor | {r2_scaling:.4f} |

## Regression Equations

- **Linear Regression Equation**: $y = {linear_regressor.coef_[0]:.4f}x + {linear_regressor.intercept_:.4f}$
- **Polynomial Regression Equation**: $y = {poly_regressor.coef_[2]:.4f}x^2 + {poly_regressor.coef_[1]:.4f}x + {poly_regressor.intercept_:.4f}$
- **Scaling Factor**: ${average_scaling_factor:.4f}$

The analysis includes Mean Squared Error (MSE) and R^2 scores for each calibration method to help determine the most accurate approach for converting sensor readings into weight measurements.
"""

# Save Markdown content to file
with open('calibration_results.md', 'w') as md_file:
    md_file.write(md_content)

print("Calibration results and analysis saved to calibration_results.md.")
