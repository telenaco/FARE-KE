import numpy as np
import os
from numpy.linalg import lstsq
from scipy.optimize import minimize
from sklearn.metrics import r2_score


# Raw outputs (mV/V) for each load direction
# No multiplication by 10, assuming these are the correct readings as provided
csv_filename = "rawCalibrationData.csv"

# Later in the script, or in another script, you can load the raw_outputs from the CSV
raw_outputs = np.loadtxt(csv_filename, delimiter=",")

# Assuming full scale outputs in the same order
full_scale_outputs = np.array([5, 5, 5, 10, 10, 10])  # Adjust as needed

# Constructing the data matrix 'K'
K = raw_outputs / full_scale_outputs[:, None]

# Calculate the inverse of matrix 'K'
K_inv = np.linalg.inv(K)

# Verification: Multiplying 'K' with its inverse to get an identity matrix
identity_matrix = np.dot(K, K_inv)

# Function to apply the inverse matrix to raw measurements
def compensate_measurements(raw_measurements):
    compensated_measurements = np.dot(K_inv, raw_measurements)
    return compensated_measurements

# Example raw measurements (in mV/V)
raw_measurements = np.array([-1.6510, 0.6151, 0.2501, 1.0054, 0.8402, 0.0067])

# Apply the compensation function to the raw measurements
compensated_measurements = compensate_measurements(raw_measurements)

# Calculate the accuracy metrics
expected_loads = np.array([-4.35, 1.37, 0.69, 5.54, 4.11, -0.31])
absolute_errors = np.abs(compensated_measurements - expected_loads)
mean_absolute_error = np.mean(absolute_errors)
max_absolute_error = np.max(absolute_errors)

# Save the results to a Markdown file
script_name = os.path.basename(__file__)
md_filename = script_name.replace(".py", ".md")

def matrix_to_md_table(matrix, header, precision=4):
    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "|---" * len(header) + "|\n"
    for row in matrix:
        formatted_row = ["{:.{}f}".format(val, precision) for val in row]
        md_table += "| " + " | ".join(formatted_row) + " |\n"
    return md_table

with open(md_filename, "w") as md_file:
    md_file.write("# 6-Axis Load Cell Calibration Results\n\n")
    
    md_file.write("## K Matrix (Sensitivity Matrix)\n")
    md_file.write("This matrix represents the sensitivity of the load cell to the applied loads.\n\n")
    header = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
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


