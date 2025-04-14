"""
Filename: six_axis_optimization_calibration.py

Description:
    Advanced calibration module for 6-axis force/torque sensors using optimization techniques.
    Implements and compares multiple calibration methods: single-point calibration (per-test-point matrix),
    matrix averaging (arithmetic mean of calibration matrices), and optimized matrix (error minimization).
    Evaluates performance of each method using relative error and R² metrics to determine optimal approach.

Author: José Luis Berna Moya
Date: March 2025
"""

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
from sklearn.metrics import r2_score


def calculate_single_point_calibration_matrices(R_points, A_matrices):
    """
    Calculate calibration matrices for each calibration point independently
    using least squares method.
    
    Parameters:
        R_points (list): List of sensor reading matrices for each calibration point.
        A_matrices (list): List of known applied load matrices for each calibration point.
    
    Returns:
        list: List of calibration matrices for each calibration point.
    """
    calibration_matrices = []
    for R, A in zip(R_points, A_matrices):
        K, _, _, _ = lstsq(R, A, rcond=None)
        calibration_matrices.append(K)
    
    return calibration_matrices


def calculate_average_calibration_matrix(calibration_matrices):
    """
    Compute the calibration matrix by averaging each element across matrices.
    
    Parameters:
        calibration_matrices (list): List of individual calibration matrices.
        
    Returns:
        np.ndarray: Averaged calibration matrix.
    """
    return np.mean(calibration_matrices, axis=0)


def calculate_optimized_calibration_matrix(R_points, A_matrices):
    """
    Calculate an optimized calibration matrix by minimizing the total error
    across all calibration points.
    
    Parameters:
        R_points (list): List of sensor reading matrices for each calibration point.
        A_matrices (list): List of known applied load matrices for each calibration point.
    
    Returns:
        np.ndarray: Optimized 6x6 calibration matrix.
    """
    # Error function for optimization
    def error_function(flattened_matrix):
        """
        Calculate total squared error across all calibration points
        for a given calibration matrix.
        
        Parameters:
            flattened_matrix (np.ndarray): Flattened 6x6 matrix (36 elements).
            
        Returns:
            float: Total squared error.
        """
        matrix = flattened_matrix.reshape(6, 6)
        total_error = 0
        for R, A in zip(R_points, A_matrices):
            predicted = R @ matrix
            error = np.sum((predicted - A) ** 2)
            total_error += error
        return total_error

    # Get the average calibration matrix as initial guess
    calibration_matrices = calculate_single_point_calibration_matrices(R_points, A_matrices)
    initial_guess = calculate_average_calibration_matrix(calibration_matrices).flatten()

    # Minimize the error function using BFGS algorithm
    result = minimize(error_function, initial_guess, method='BFGS')
    
    # Return the optimized matrix reshaped to 6x6
    return result.x.reshape(6, 6)


def evaluate_performance(estimated, actual):
    """
    Calculate performance metrics for a calibration method.
    
    Parameters:
        estimated (np.ndarray): Matrix of estimated forces/torques.
        actual (np.ndarray): Matrix of actual forces/torques.
        
    Returns:
        tuple: Tuple containing (mean_relative_error, r2_score).
    """
    # Calculate absolute error
    absolute_error = np.abs(estimated - actual)
    
    # Handle division by zero in relative error computation
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error_percentage = np.where(actual != 0, (absolute_error / np.abs(actual)) * 100, 0)
    
    # Compute mean relative error
    mean_relative_error = np.nanmean(relative_error_percentage)
    
    # Convert to scientific notation if below threshold
    if mean_relative_error < 0.001:
        mean_relative_error = "{:.2e}".format(mean_relative_error)
    
    # Calculate R^2 value
    r2 = r2_score(actual, estimated)
    
    return mean_relative_error, r2


def main():
    """
    Main function to calculate and evaluate different calibration methods.
    """
    # Sample uncalibrated sensor readings for three calibration points (mV/V)
    # Each row represents an axis (Fx, Fy, Fz, Mx, My, Mz)
    # Each column represents a load cell channel
    
    # Point 1 data (5N forces, 2.5Nm torques)
    R_point1 = np.array([
        [5.3256, 0.0134, 0.0321, 0.0112, 0.0123, 0.0089],
        [0.0098, 5.2876, 0.0156, 0.0145, 0.0167, 0.0101],
        [0.0102, 0.0124, 5.3125, 0.0137, 0.0148, 0.0095],
        [0.0110, 0.0113, 0.0105, 2.4876, 0.0129, 0.0111],
        [0.0099, 0.0130, 0.0127, 0.0114, 2.5023, 0.0103],
        [0.0103, 0.0122, 0.0118, 0.0125, 0.0135, 2.4954]
    ])

    # Point 2 data (10N forces, 5Nm torques)
    R_point2 = np.array([
        [10.3123, 0.0213, 0.0264, 0.0198, 0.0176, 0.0145],
        [0.0156, 10.2956, 0.0167, 0.0123, 0.0189, 0.0154],
        [0.0148, 0.0164, 10.3087, 0.0175, 0.0153, 0.0161],
        [0.0162, 0.0149, 0.0160, 4.9872, 0.0157, 0.0143],
        [0.0154, 0.0173, 0.0159, 0.0166, 4.9931, 0.0136],
        [0.0146, 0.0158, 0.0147, 0.0150, 0.0162, 4.9894]
    ])

    # Point 3 data (20N forces, 10Nm torques)
    R_point3 = np.array([
        [20.3412, 0.0198, 0.0245, 0.0187, 0.0164, 0.0153],
        [0.0176, 20.3289, 0.0145, 0.0167, 0.0156, 0.0142],
        [0.0152, 0.0169, 20.3356, 0.0171, 0.0140, 0.0159],
        [0.0165, 0.0155, 0.0163, 9.9745, 0.0144, 0.0138],
        [0.0174, 0.0160, 0.0148, 0.0152, 9.9823, 0.0130],
        [0.0150, 0.0146, 0.0157, 0.0168, 0.0154, 9.9786]
    ])

    # Known applied forces and torques for each point as matrices
    A_matrix1 = np.array([
        [5, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 0],
        [0, 0, 0, 2.5, 0, 0],
        [0, 0, 0, 0, 2.5, 0],
        [0, 0, 0, 0, 0, 2.5]
    ])

    A_matrix2 = np.array([
        [10, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 0],
        [0, 0, 10, 0, 0, 0],
        [0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 5]
    ])

    A_matrix3 = np.array([
        [20, 0, 0, 0, 0, 0],
        [0, 20, 0, 0, 0, 0],
        [0, 0, 20, 0, 0, 0],
        [0, 0, 0, 10, 0, 0],
        [0, 0, 0, 0, 10, 0],
        [0, 0, 0, 0, 0, 10]
    ])
    
    # Create lists of reading and actual matrices for processing
    R_points = [R_point1, R_point2, R_point3]
    A_matrices = [A_matrix1, A_matrix2, A_matrix3]
    
    # Calculate calibration matrices using different methods
    K_point1, K_point2, K_point3 = calculate_single_point_calibration_matrices(R_points, A_matrices)
    K_avg = calculate_average_calibration_matrix([K_point1, K_point2, K_point3])
    K_opt = calculate_optimized_calibration_matrix(R_points, A_matrices)
    
    # Use the calibration matrices to estimate forces from readings
    estimated_forces_from_K1 = R_point1 @ K_point1
    estimated_forces_from_K2 = R_point2 @ K_point2
    estimated_forces_from_K3 = R_point3 @ K_point3

    # Using the average calibration matrix to estimate forces for the three points
    estimated_forces_avg_1 = R_point1 @ K_avg
    estimated_forces_avg_2 = R_point2 @ K_avg
    estimated_forces_avg_3 = R_point3 @ K_avg

    # Using the optimized matrix to estimate forces for the three points
    estimated_forces_optimized_1 = R_point1 @ K_opt
    estimated_forces_optimized_2 = R_point2 @ K_opt
    estimated_forces_optimized_3 = R_point3 @ K_opt
    
    # Evaluate performance for all methods and points
    mae_percentage_K1, r2_K1 = evaluate_performance(estimated_forces_from_K1, A_matrix1)
    mae_percentage_K2, r2_K2 = evaluate_performance(estimated_forces_from_K2, A_matrix2)
    mae_percentage_K3, r2_K3 = evaluate_performance(estimated_forces_from_K3, A_matrix3)

    mae_percentage_avg_1, r2_avg_1 = evaluate_performance(estimated_forces_avg_1, A_matrix1)
    mae_percentage_avg_2, r2_avg_2 = evaluate_performance(estimated_forces_avg_2, A_matrix2)
    mae_percentage_avg_3, r2_avg_3 = evaluate_performance(estimated_forces_avg_3, A_matrix3)

    mae_percentage_optimized_1, r2_optimized_1 = evaluate_performance(estimated_forces_optimized_1, A_matrix1)
    mae_percentage_optimized_2, r2_optimized_2 = evaluate_performance(estimated_forces_optimized_2, A_matrix2)
    mae_percentage_optimized_3, r2_optimized_3 = evaluate_performance(estimated_forces_optimized_3, A_matrix3)
    
    # Print results in a tabular format for easy comparison
    print("Results for Point 1:")
    print("Method                    Mean Relative Error (%)   R^2")
    print("------------------------------------------------------------")
    print(f"{'K_point1:':<25} {mae_percentage_K1:<25} {r2_K1:.4e}")
    print(f"{'Optimized Matrix:':<25} {mae_percentage_optimized_1:<25} {r2_optimized_1:.4e}")
    print(f"{'Average Matrix:':<25} {mae_percentage_avg_1:<25} {r2_avg_1:.4e}")
    print("\n")

    print("Results for Point 2:")
    print("Method                    Mean Relative Error (%)   R^2")
    print("------------------------------------------------------------")
    print(f"{'K_point2:':<25} {mae_percentage_K2:<25} {r2_K2:.4e}")
    print(f"{'Optimized Matrix:':<25} {mae_percentage_optimized_2:<25} {r2_optimized_2:.4e}")
    print(f"{'Average Matrix:':<25} {mae_percentage_avg_2:<25} {r2_avg_2:.4e}")
    print("\n")

    print("Results for Point 3:")
    print("Method                    Mean Relative Error (%)   R^2")
    print("------------------------------------------------------------")
    print(f"{'K_point3:':<25} {mae_percentage_K3:<25} {r2_K3:.4e}")
    print(f"{'Optimized Matrix:':<25} {mae_percentage_optimized_3:<25} {r2_optimized_3:.4e}")
    print(f"{'Average Matrix:':<25} {mae_percentage_avg_3:<25} {r2_avg_3:.4e}")


if __name__ == "__main__":
    main()