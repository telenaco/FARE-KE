"""
Filename: beam_load_cell_calibration.py

Description:
    Analysis module for calibrating single-axis load cells using multiple regression methods.
    Processes weight-reading data pairs to determine optimal calibration approach among 
    linear regression, polynomial regression, and scaling factor calibration.
    Generates visualization and statistical comparisons between methods, with results 
    exported to markdown format for documentation.

Author: José Luis Berna Moya
Date: March 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def load_calibration_data(filepath='calibrationWeightSingleBeamMCP.csv'):
    """
    Load load cell calibration data from CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file containing calibration data.
        
    Returns:
        tuple: Tuple containing weight values and reading values.
    """
    # Load the data
    data = pd.read_csv(filepath)
    x = data['Weight'].values.reshape(-1, 1)
    y = data['Reading'].values
    
    return data, x, y

def perform_calibration_analysis(data, x, y):
    """
    Perform comprehensive calibration analysis using multiple regression methods.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the calibration data.
        x (np.ndarray): Weight values.
        y (np.ndarray): Reading values.
        
    Returns:
        tuple: Tuple containing calibration results for each method.
    """
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
    
    return offset, y_corrected, y_pred_linear, y_pred_poly, y_pred_scaling, linear_regressor, poly_regressor, average_scaling_factor, non_zero_mask

def plot_calibration_results(x, y_corrected, y_pred_linear, y_pred_poly, y_pred_scaling, non_zero_mask):
    """
    Plot calibration results comparison.
    
    Parameters:
        x (np.ndarray): Weight values.
        y_corrected (np.ndarray): Offset-corrected reading values.
        y_pred_linear (np.ndarray): Linear regression predictions.
        y_pred_poly (np.ndarray): Polynomial regression predictions.
        y_pred_scaling (np.ndarray): Scaling factor predictions.
        non_zero_mask (np.ndarray): Boolean mask for non-zero weight values.
    """
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

def calculate_metrics(y_corrected, y_pred_linear, y_pred_poly, y_pred_scaling, non_zero_mask):
    """
    Calculate performance metrics for each calibration method.
    
    Parameters:
        y_corrected (np.ndarray): Offset-corrected reading values.
        y_pred_linear (np.ndarray): Linear regression predictions.
        y_pred_poly (np.ndarray): Polynomial regression predictions.
        y_pred_scaling (np.ndarray): Scaling factor predictions.
        non_zero_mask (np.ndarray): Boolean mask for non-zero weight values.
        
    Returns:
        tuple: Tuple containing R-squared values for each method.
    """
    r2_linear = r2_score(y_corrected, y_pred_linear)
    r2_poly = r2_score(y_corrected, y_pred_poly)
    r2_scaling = r2_score(y_corrected[non_zero_mask], y_pred_scaling[non_zero_mask])
    
    return r2_linear, r2_poly, r2_scaling

def generate_markdown_report(r2_linear, r2_poly, r2_scaling, linear_regressor, poly_regressor, average_scaling_factor):
    """
    Generate markdown report summarizing calibration results.
    
    Parameters:
        r2_linear (float): R-squared for linear regression.
        r2_poly (float): R-squared for polynomial regression.
        r2_scaling (float): R-squared for scaling factor.
        linear_regressor (LinearRegression): Fitted linear regressor.
        poly_regressor (LinearRegression): Fitted polynomial regressor.
        average_scaling_factor (float): Average scaling factor.
        
    Returns:
        str: Markdown content for the report.
    """
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
    return md_content

def save_markdown_report(md_content, output_file='calibration_results.md'):
    """
    Save markdown report to file.
    
    Parameters:
        md_content (str): Markdown content to save.
        output_file (str): Output filename.
    """
    # Save Markdown content to file
    with open(output_file, 'w') as md_file:
        md_file.write(md_content)
    print(f"Calibration results and analysis saved to {output_file}")

# Main execution block
if __name__ == "__main__":
    # Load calibration data
    data, x, y = load_calibration_data()
    
    # Perform calibration analysis
    offset, y_corrected, y_pred_linear, y_pred_poly, y_pred_scaling, linear_regressor, poly_regressor, average_scaling_factor, non_zero_mask = perform_calibration_analysis(data, x, y)
    
    # Generate visualization
    plot_calibration_results(x, y_corrected, y_pred_linear, y_pred_poly, y_pred_scaling, non_zero_mask)
    
    # Calculate performance metrics
    r2_linear, r2_poly, r2_scaling = calculate_metrics(y_corrected, y_pred_linear, y_pred_poly, y_pred_scaling, non_zero_mask)
    
    # Generate and save markdown report
    md_content = generate_markdown_report(r2_linear, r2_poly, r2_scaling, linear_regressor, poly_regressor, average_scaling_factor)
    save_markdown_report(md_content)