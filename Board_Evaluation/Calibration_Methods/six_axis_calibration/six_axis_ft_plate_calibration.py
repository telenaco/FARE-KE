"""
Filename: six_axis_ft_plate_calibration.py

Description:
    Comprehensive calibration and analysis module for 6-axis force/torque plate sensor systems.
    Processes calibration data from multiple load scenarios, analyzes cross-talk characteristics, 
    and evaluates performance metrics including MAPE, correlation, and signal noise metrics.
    Generates visualizations for cross-talk analysis, error distributions, and frequency characteristics.

Author: José Luis Berna Moya
Date: March 2025
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def load_calibration_data(csv_file="calibrationReadings.csv"):
    """
    Load calibration data from CSV file.
    
    Parameters:
        csv_file (str): Path to the CSV file containing calibration data.
        
    Returns:
        pd.DataFrame: DataFrame containing the calibration data.
    """
    data = pd.read_csv(csv_file)
    # Drop rows with any missing values
    data = data.dropna()
    return data


def separate_calibration_validation_data(data):
    """
    Separate data into calibration and validation sets.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the complete dataset.
        
    Returns:
        tuple: Two DataFrames, one for calibration and one for validation.
    """
    calibration_rows = []
    validation_rows = []

    for i in range(0, len(data), 10):
        axis_data = data.iloc[i:i+10]
        calibration_rows.extend(axis_data.index[1:10:2])  # Select odd-indexed rows starting from index 1
        validation_rows.extend(axis_data.index[2:10:2])   # Select even-indexed rows starting from index 2

    # This will override the above to include all row indices in the calibration
    calibration_rows = data.index.tolist()  

    # Specify calibration data based on indices
    calibration_data = data.loc[calibration_rows]
    validation_data = data.loc[validation_rows]
    
    return calibration_data, validation_data


def calculate_calibration_matrix(calibration_data):
    """
    Compute calibration matrix from calibration data using least squares method.
    
    Parameters:
        calibration_data (pd.DataFrame): DataFrame containing calibration data.
        
    Returns:
        np.ndarray: Calibration matrix.
    """
    # Extract sensor readings (R) and known forces/torques (F) for calibration
    R_cal = calibration_data.iloc[:, :6].values
    F_cal = calibration_data.iloc[:, 6:].values

    # Compute calibration matrices for each calibration point
    K_cal, _, _, _ = lstsq(R_cal, F_cal, rcond=None)
    
    return K_cal, R_cal, F_cal


def calculate_mape(estimated, actual):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Parameters:
        estimated (np.ndarray): Array of estimated values.
        actual (np.ndarray): Array of actual values.
        
    Returns:
        float: MAPE value, or NaN/Infinity for edge cases.
    """
    # Ensure we are not dividing by zero; filter out such cases
    mask = actual != 0
    actual_filtered = actual[mask]
    estimated_filtered = estimated[mask]
    
    # Calculate MAPE using filtered values
    mape = np.mean(np.abs((actual_filtered - estimated_filtered) / actual_filtered)) * 100
    return mape


def compute_performance_metrics(R_cal, F_cal, K_cal):
    """
    Compute performance metrics for calibration.
    
    Parameters:
        R_cal (np.ndarray): Raw sensor readings.
        F_cal (np.ndarray): Known forces/torques.
        K_cal (np.ndarray): Calibration matrix.
        
    Returns:
        dict: Dictionary of performance metrics.
    """
    # Calculate MAPE for the raw sensor readings against the known values
    mape_raw = calculate_mape(R_cal, F_cal)

    # Adjust the estimated forces/torques using the calibration matrix to compensate the errors
    F_cal_est_initial = R_cal @ K_cal

    # Compute MAPE for the calibration matrix
    mape_cal_initial = calculate_mape(F_cal_est_initial, F_cal)  

    # Compute R-squared for the calibration matrix
    r2_cal_initial = r2_score(F_cal, F_cal_est_initial)

    # Compute R-squared for the raw readings
    r2_raw = r2_score(F_cal, R_cal)
    
    # Combine metrics into a dictionary
    performance_metrics = {
        'MAPE_raw': mape_raw,
        'MAPE_calibrated': mape_cal_initial,
        'R2_raw': r2_raw,
        'R2_calibrated': r2_cal_initial
    }
    
    return performance_metrics


def plot_performance_comparison(metrics):
    """
    Plot comparison of performance metrics for raw and calibrated data.
    
    Parameters:
        metrics (dict): Dictionary of performance metrics.
    """
    performance_comparison = pd.DataFrame({
        'Metric': ['MAPE (%)', 'R-squared'],
        'Raw Readings': [metrics['MAPE_raw'], metrics['R2_raw']],
        'Initial Calibration': [metrics['MAPE_calibrated'], metrics['R2_calibrated']]
    })

    print("Performance Metrics Comparison:")
    print(performance_comparison)


def plot_error_distributions(R_cal, F_cal, F_cal_est):
    """
    Plot error distributions for raw and calibrated data.
    
    Parameters:
        R_cal (np.ndarray): Raw sensor readings.
        F_cal (np.ndarray): Known forces/torques.
        F_cal_est (np.ndarray): Estimated forces/torques after calibration.
    """
    # Calculate the errors of these "raw" readings against the known values
    errors_raw = R_cal - F_cal

    # Calculate the errors for the calibrated corrected values
    errors_cal_initial = F_cal_est - F_cal

    # Plotting error distributions
    plt.figure(figsize=(14, 6), dpi=300)

    # Histogram for Raw vs Calibration
    plt.subplot(1, 2, 1)
    plt.hist(errors_raw.flatten(), bins=20, alpha=0.7, label='Raw Readings')
    plt.hist(errors_cal_initial.flatten(), bins=20, alpha=0.7, label='Calibrated')
    plt.title('Error Distribution for Raw and Calibrated Readings')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.legend()

    # Box plot for Raw vs Calibrated
    plt.subplot(1, 2, 2)
    plt.boxplot([errors_raw.flatten(), errors_cal_initial.flatten()], labels=['Raw', 'Calibrated'])
    plt.title('Error Distribution (Box Plot) for Raw and Calibrated Readings')
    plt.ylabel('Error')

    plt.tight_layout()
    plt.show()


def plot_scatter_comparison(F_cal, R_cal, F_cal_est, axis_names):
    """
    Plot scatter comparison between actual, raw, and calibrated values.
    
    Parameters:
        F_cal (np.ndarray): Known forces/torques.
        R_cal (np.ndarray): Raw sensor readings.
        F_cal_est (np.ndarray): Estimated forces/torques after calibration.
        axis_names (list): List of axis names.
    """
    plt.figure(figsize=(14, 8), dpi=300)
    for i, axis_name in enumerate(axis_names, start=1):
        plt.subplot(2, 3, i)
        plt.scatter(F_cal[:, i-1], R_cal[:, i-1], alpha=0.5, label='Raw', marker='o')
        plt.scatter(F_cal[:, i-1], F_cal_est[:, i-1], alpha=0.5, label='Calibrated', marker='x')
        plt.plot(F_cal[:, i-1], F_cal[:, i-1], 'r--')  # Line for perfect agreement
        plt.title(f'Actual vs. Estimated {axis_name}')
        plt.xlabel('Actual')
        plt.ylabel('Estimated')
        plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_mae_by_axis(errors_raw, errors_initial, axis_names):
    """
    Calculate and plot Mean Absolute Error (MAE) by axis.
    
    Parameters:
        errors_raw (np.ndarray): Errors in raw readings.
        errors_initial (np.ndarray): Errors in calibrated readings.
        axis_names (list): List of axis names.
    """
    mae_raw = np.mean(np.abs(errors_raw), axis=0)
    mae_initial = np.mean(np.abs(errors_initial), axis=0)
    
    x = np.arange(len(axis_names))  # the label locations
    width = 0.35  # the width of the bars
    
    # Increase the figure DPI for better resolution
    fig, ax = plt.subplots(dpi=300)  # Set DPI here

    # Adjust the font properties
    plt.rcParams.update({'font.size': 14})

    rects1 = ax.bar(x - width/2, mae_raw, width, label='Raw')
    rects2 = ax.bar(x + width/2, mae_initial, width, label='Calibrated')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_title('MAE by Axis for Raw vs. Calibrated', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(axis_names, fontsize=12)
    ax.legend()

    fig.tight_layout()
    plt.show()


def calculate_crosstalk_errors(readings, known_values):
    """
    Calculate crosstalk errors where known values are zero.
    
    Parameters:
        readings (np.ndarray): Readings to analyze.
        known_values (np.ndarray): Known correct values.
        
    Returns:
        np.ndarray: Absolute crosstalk errors.
    """
    # Crosstalk occurs where known_values are zero
    crosstalk_mask = (known_values == 0)
    crosstalk_errors = readings * crosstalk_mask  
    return np.abs(crosstalk_errors)  # Return the absolute errors


def plot_crosstalk_errors(axis, raw_errors, initial_errors):
    """
    Plot crosstalk errors for a specific axis.
    
    Parameters:
        axis (int): Axis index.
        raw_errors (np.ndarray): Errors in raw readings.
        initial_errors (np.ndarray): Errors in calibrated readings.
    """
    # Each subplot for an axis
    plt.figure(figsize=(12, 3), dpi=300)
    plt.bar(np.arange(len(raw_errors)) - 0.1, raw_errors, width=0.4, label='Raw')
    plt.bar(np.arange(len(initial_errors)) + 0.1, initial_errors, width=0.4, label='Calibrated')
    plt.title(f'Crosstalk Error on Axis {axis}')
    plt.xlabel('Measurement Instance')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


def analyze_crosstalk(R_cal, F_cal, K_cal):
    """
    Analyze and visualize crosstalk between sensor channels.
    
    Parameters:
        R_cal (np.ndarray): Raw sensor readings.
        F_cal (np.ndarray): Known forces/torques.
        K_cal (np.ndarray): Calibration matrix.
    """
    # Calculate crosstalk errors for the first ten values, this is the Fy calibration readings
    raw_crosstalk_errors = calculate_crosstalk_errors(R_cal[:10], F_cal[:10])
    calibrated_crosstalk_errors = calculate_crosstalk_errors((R_cal @ K_cal)[:10], F_cal[:10])

    # Create DataFrames for the error tables
    raw_error_table = pd.DataFrame(raw_crosstalk_errors, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
    calibrated_error_table = pd.DataFrame(calibrated_crosstalk_errors, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])

    # Round the values in the DataFrames to three digits of precision
    raw_error_table_rounded = raw_error_table.round(3)
    calibrated_error_table_rounded = calibrated_error_table.round(3)

    # Convert DataFrames to markdown format with three digits of precision
    raw_error_markdown = raw_error_table_rounded.to_markdown(index=False)
    calibrated_error_markdown = calibrated_error_table_rounded.to_markdown(index=False)

    # Write the markdown tables to a single .md file
    with open('crosstalk_errors.md', 'w') as f:
        f.write("Crosstalk Error From Raw Readings Table (First 10 Values):\n")
        f.write(raw_error_markdown)
        f.write("\n\nCalibrated Crosstalk Error Table (First 10 Values):\n")
        f.write(calibrated_error_markdown)
    
    # Six axes (Fx, Fy, Fz, Mx, My, Mz), iterate and plot crosstalk errors
    for axis in range(6):
        # Skipping the main axis reading and considering only crosstalk errors
        plot_crosstalk_errors(axis+1, 
                              raw_crosstalk_errors[:, axis], 
                              calibrated_crosstalk_errors[:, axis])


def calculate_crosstalk_magnitude(readings, known_values):
    """
    Calculate the crosstalk magnitude for the non-actuated axes.
    
    Parameters:
        readings (np.ndarray): Readings to analyze.
        known_values (np.ndarray): Known correct values.
        
    Returns:
        np.ndarray: Mean absolute crosstalk magnitude by axis.
    """
    crosstalk_mask = known_values == 0
    crosstalk_readings = np.where(crosstalk_mask, readings, np.nan)  # Isolate crosstalk readings
    return np.nanmean(np.abs(crosstalk_readings), axis=0)  # Calculate mean absolute error ignoring NaNs


def analyze_full_dataset_crosstalk(R_cal, F_cal, K_cal):
    """
    Analyze crosstalk characteristics across the entire dataset.
    
    Parameters:
        R_cal (np.ndarray): Raw sensor readings.
        F_cal (np.ndarray): Known forces/torques.
        K_cal (np.ndarray): Calibration matrix.
    """
    # Calculate crosstalk errors for the entire dataset
    raw_crosstalk_errors = calculate_crosstalk_errors(R_cal, F_cal)
    calibrated_crosstalk_errors = calculate_crosstalk_errors((R_cal @ K_cal), F_cal)

    # Function to get statistical summaries for the full dataset
    def full_crosstalk_error_statistics(raw_errors, calibrated_errors):
        # Convert the errors to a DataFrame
        raw_error_df = pd.DataFrame(raw_errors, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
        calibrated_error_df = pd.DataFrame(calibrated_errors, columns=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])

        # Get statistical summaries
        raw_stats = raw_error_df.describe()
        calibrated_stats = calibrated_error_df.describe()

        return raw_stats, calibrated_stats

    # Compute the statistics for the full dataset
    raw_stats, calibrated_stats = full_crosstalk_error_statistics(raw_crosstalk_errors, calibrated_crosstalk_errors)

    # Now you can print these summaries or return them from a function
    print("Raw Crosstalk Error Statistics for the full dataset:")
    print(raw_stats)
    print("\nCalibrated Crosstalk Error Statistics for the full dataset:")
    print(calibrated_stats)
    
    # Calculate crosstalk magnitude for raw readings
    raw_crosstalk_magnitude = calculate_crosstalk_magnitude(R_cal, F_cal)

    # Apply initial calibration matrix and calculate crosstalk magnitude
    calibrated_crosstalk_magnitude = calculate_crosstalk_magnitude(R_cal @ K_cal, F_cal)

    # Output the crosstalk magnitude for comparison
    print("Raw crosstalk magnitude:", raw_crosstalk_magnitude)
    print("Calibrated crosstalk magnitude:", calibrated_crosstalk_magnitude)


# Main execution block
if __name__ == "__main__":
    # Load calibration data
    data = load_calibration_data()
    
    # Separate into calibration and validation sets
    calibration_data, validation_data = separate_calibration_validation_data(data)
    
    # Calculate calibration matrix and perform initial analysis
    K_cal, R_cal, F_cal = calculate_calibration_matrix(calibration_data)
    
    # Calculate estimated forces/torques using calibration matrix
    F_cal_est = R_cal @ K_cal
    
    # Compute performance metrics
    metrics = compute_performance_metrics(R_cal, F_cal, K_cal)
    
    # Display performance comparison
    plot_performance_comparison(metrics)
    
    # Plot error distributions
    plot_error_distributions(R_cal, F_cal, F_cal_est)
    
    # Plot scatter comparison
    axis_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    plot_scatter_comparison(F_cal, R_cal, F_cal_est, axis_names)
    
    # Calculate and plot MAE by axis
    errors_raw = R_cal - F_cal
    errors_cal = F_cal_est - F_cal
    calculate_mae_by_axis(errors_raw, errors_cal, axis_names)
    
    # Analyze crosstalk
    analyze_crosstalk(R_cal, F_cal, K_cal)
    
    # Analyze full dataset crosstalk
    analyze_full_dataset_crosstalk(R_cal, F_cal, K_cal)