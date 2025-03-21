import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
from sklearn.metrics import r2_score


# uncalibrated sensor readings for Points 1, 2, and 3
R_point1 = np.array([
    [5.3256, 0.0134, 0.0321, 0.0112, 0.0123, 0.0089],
    [0.0098, 5.2876, 0.0156, 0.0145, 0.0167, 0.0101],
    [0.0102, 0.0124, 5.3125, 0.0137, 0.0148, 0.0095],
    [0.0110, 0.0113, 0.0105, 2.4876, 0.0129, 0.0111],
    [0.0099, 0.0130, 0.0127, 0.0114, 2.5023, 0.0103],
    [0.0103, 0.0122, 0.0118, 0.0125, 0.0135, 2.4954]
])

R_point2 = np.array([
    [10.3123, 0.0213, 0.0264, 0.0198, 0.0176, 0.0145],
    [0.0156, 10.2956, 0.0167, 0.0123, 0.0189, 0.0154],
    [0.0148, 0.0164, 10.3087, 0.0175, 0.0153, 0.0161],
    [0.0162, 0.0149, 0.0160, 4.9872, 0.0157, 0.0143],
    [0.0154, 0.0173, 0.0159, 0.0166, 4.9931, 0.0136],
    [0.0146, 0.0158, 0.0147, 0.0150, 0.0162, 4.9894]
])

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

# Step 1: Calculate the calibration matrices K using the least squares method
# This provides a basic calibration matrix for each calibration point.
K_point1, _, _, _ = lstsq(R_point1, A_matrix1, rcond=None)
K_point2, _, _, _ = lstsq(R_point2, A_matrix2, rcond=None)
K_point3, _, _, _ = lstsq(R_point3, A_matrix3, rcond=None)

# Lists to facilitate iteration over each calibration point during optimization
R_points = [R_point1, R_point2, R_point3]  # Uncalibrated sensor readings
A_matrices = [A_matrix1, A_matrix2, A_matrix3]  # Known applied forces and torques

# Error function for optimization
# This function calculates the total error between estimated and actual forces/torques 
# using a given calibration matrix.
def error_function(flattened_matrix):
    matrix = flattened_matrix.reshape(6, 6)  # Reshape the flattened matrix to 6x6
    total_error = 0
    for R, A in zip(R_points, A_matrices):
        predicted = R @ matrix  # Matrix multiplication to get estimated forces/torques
        error = np.sum((predicted - A) ** 2)  # Compute the squared error
        total_error += error  # Accumulate the error
    return total_error

# Initial guess for the calibration matrix by averaging matrices from individual calibration points
initial_guess = np.mean([K_point1, K_point2, K_point3], axis=0).flatten()

# Minimization using BFGS algorithm
# The goal is to find a calibration matrix that minimizes the total error across all calibration points.
result = minimize(error_function, initial_guess, method='BFGS')

# Extract the optimized calibration matrix from the result
K_opt = result.x.reshape(6, 6)

def evaluate_performance(estimated, actual):
    # Calculate absolute error
    absolute_error = np.abs(estimated - actual)
    
    # Handle division by zero in relative error computation
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error_percentage = np.where(actual != 0, (absolute_error / np.abs(actual)) * 100, 0)
    
    # Compute mean relative error
    mean_relative_error = np.nanmean(relative_error_percentage)
    
    # Convert to scientific notation if below threshold
    if mean_relative_error < 0.001:  # Threshold can be adjusted
        mean_relative_error = "{:.2e}".format(mean_relative_error)
    
    # Calculate R^2 value
    r2 = r2_score(actual, estimated)
    
    return mean_relative_error, r2

def average_calibration_matrix(*matrices):
    """Compute the calibration matrix by averaging each element across matrices."""
    return np.mean(matrices, axis=0)

# Compute the average calibration matrix based on K_point_i with all the calibration matrices
K_avg = average_calibration_matrix(K_point1, K_point2, K_point3)

# Using the calibration matrices to estimate forces from readings
estimated_forces_from_K1 = R_point1 @ K_point1
estimated_forces_from_K2 = R_point2 @ K_point2
estimated_forces_from_K3 = R_point3 @ K_point3

# Using the average calibration matrix to estimate forces for the three points
estimated_forces_avg_1 = R_point1 @ K_avg
estimated_forces_avg_2 = R_point2 @ K_avg
estimated_forces_avg_3 = R_point3 @ K_avg

# Using the K_opt to estimate forces for the three points
estimated_forces_optimized_1 = R_point1 @ K_opt
estimated_forces_optimized_2 = R_point2 @ K_opt
estimated_forces_optimized_3 = R_point3 @ K_opt

# Evaluate performance for K matrices
mae_percentage_K1, r2_K1 = evaluate_performance(estimated_forces_from_K1, A_matrix1)
mae_percentage_K2, r2_K2 = evaluate_performance(estimated_forces_from_K2, A_matrix2)
mae_percentage_K3, r2_K3 = evaluate_performance(estimated_forces_from_K3, A_matrix3)

# Evaluate performance for average matrix
mae_percentage_avg_1, r2_avg_1 = evaluate_performance(estimated_forces_avg_1, A_matrix1)
mae_percentage_avg_2, r2_avg_2 = evaluate_performance(estimated_forces_avg_2, A_matrix2)
mae_percentage_avg_3, r2_avg_3 = evaluate_performance(estimated_forces_avg_3, A_matrix3)

# Evaluate performance for optimized matrix
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

