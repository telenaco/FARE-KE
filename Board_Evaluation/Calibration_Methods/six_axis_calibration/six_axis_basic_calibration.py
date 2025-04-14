"""
Filename: six_axis_basic_calibration.py

Description:
    Basic calibration module for 6-axis force/torque sensors.
    Loads raw calibration data, computes sensitivity and inverse matrices, 
    and generates a detailed calibration report in Markdown format.
    Used for initial calibration of force/torque plate sensors in the FARE-KE framework.

Author: José Luis Berna Moya
Date: March 2025
"""

import numpy as np
import os
from numpy.linalg import lstsq
from scipy.optimize import minimize
from sklearn.metrics import r2_score


def load_calibration_data(csv_filename="raw_calibration_data.csv"):
    """
    Load raw calibration data from CSV file.
    
    Parameters:
        csv_filename (str): Path to the CSV file containing raw ADC readings.
        
    Returns:
        np.ndarray: Matrix of raw ADC readings.
    """
    # Load the data from CSV file
    raw_outputs = np.loadtxt(csv_filename, delimiter=",")
    return raw_outputs


def calculate_sensitivity_matrix(raw_outputs, full_scale_outputs):
    """
    Calculate the sensitivity matrix (K) from raw outputs and full-scale values.
    
    Parameters:
        raw_outputs (np.ndarray): Matrix of raw ADC readings.
        full_scale_outputs (np.ndarray): Array of full-scale output values for each axis.
        
    Returns:
        np.ndarray: Sensitivity matrix (K).
    """
    # Constructing the data matrix 'K'
    K = raw_outputs / full_scale_outputs[:, None]
    return K


def compensate_measurements(K_inv, raw_measurements):
    """
    Apply the inverse calibration matrix to raw measurements to obtain compensated values.
    
    Parameters:
        K_inv (np.ndarray): Inverse of sensitivity matrix.
        raw_measurements (np.ndarray): Raw measurement values.
        
    Returns:
        np.ndarray: Compensated measurement values.
    """
    compensated_measurements = np.dot(K_inv, raw_measurements)
    return compensated_measurements


def matrix_to_md_table(matrix, header, precision=4):
    """
    Convert a matrix to a Markdown-formatted table.
    
    Parameters:
        matrix (np.ndarray): 2D matrix to format.
        header (list): List of column headers.
        precision (int): Number of decimal places for formatting.
        
    Returns:
        str: Markdown-formatted table.
    """
    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "|---" * len(header) + "|\n"
    for row in matrix:
        formatted_row = ["{:.{}f}".format(val, precision) for val in row]
        md_table += "| " + " | ".join(formatted_row) + " |\n"
    return md_table


# Main execution block
if __name__ == "__main__":
    # Define full scale outputs for each axis [Fx, Fy, Fz, Mx, My, Mz]
    full_scale_outputs = np.array([5, 5, 5, 10, 10, 10])
    
    # Load raw calibration data
    raw_outputs = load_calibration_data()
    
    # Calculate the sensitivity matrix (K)
    K = calculate_sensitivity_matrix(raw_outputs, full_scale_outputs)
    
    # Calculate the inverse of matrix 'K'
    K_inv = np.linalg.inv(K)
    
    # Verification: Multiplying 'K' with its inverse to get an identity matrix
    identity_matrix = np.dot(K, K_inv)
    
    # Example raw measurements (in mV/V)
    raw_measurements = np.array([-1.6510, 0.6151, 0.2501, 1.0054, 0.8402, 0.0067])
    
    # Apply the compensation function to the raw measurements
    compensated_measurements = compensate_measurements(K_inv, raw_measurements)
    
    # Calculate the accuracy metrics
    expected_loads = np.array([-4.35, 1.37, 0.69, 5.54, 4.11, -0.31])
    absolute_errors = np.abs(compensated_measurements - expected_loads)
    mean_absolute_error = np.mean(absolute_errors)
    max_absolute_error = np.max(absolute_errors)
    
    # Save the results to a Markdown file
    script_name = os.path.basename(__file__)
    md_filename = script_name.replace(".py", ".md")
    header = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    
    with open(md_filename, "w") as md_file:
        md_file.write("# 6-Axis Load Cell Calibration Results\n\n")
        
        md_file.write("## K Matrix (Sensitivity Matrix)\n")
        md_file.write("This matrix represents the sensitivity of the load cell to the applied loads.\n\n")
        md_file.write(matrix_to_md_table(K, header) + "\n")
        
        md_file.write("## Inverse of K Matrix (K^-1)\n")
        md_file.write("The inverse matrix is used for compensating the cross-talk between different channels.\n\n")
        md_file.write(matrix_to_md_table(K_inv, header) + "\n")
        
        md_file.write("## Verification (K * K^-1 = Identity Matrix)\n")
        md_file.write("Multiplying the K matrix by its inverse should result in an identity matrix, verifying the calculations.\n\n")
        identity_header = ["I" + str(i+1) for i in range(len(header))]
        md_file.write(matrix_to_md_table(identity_matrix, identity_header) + "\n")
        
        md_file.write("## Compensated Measurements\n")
        md_file.write("Applying the inverse matrix to the raw measurements to obtain compensated load values.\n\n")
        md_file.write("| Axis | Raw Measurement (mV/V) | Compensated Load |\n")
        md_file.write("|------|------------------------|------------------|\n")
        for i, axis in enumerate(header):
            md_file.write(f"| {axis} | {raw_measurements[i]:.4f} | {compensated_measurements[i]:.2f} |\n")
        
        md_file.write("\n## Accuracy Metrics\n")
        md_file.write(f"Mean Absolute Error: {mean_absolute_error:.2f}\n")
        md_file.write(f"Max Absolute Error: {max_absolute_error:.2f}\n")

    print(f"Results saved to {md_filename}")