import pandas as pd
import numpy as np
import os
from numpy.linalg import lstsq
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppresses any RuntimeWarnings that may arise during execution, for cleaner output.
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Variable to save messages
log_messages = []

# Load the calibration data from a CSV file into a pandas DataFrame.
file_path = '3axisCell2CalibrationData.csv'  
load_cell_data = pd.read_csv(file_path)

# Selects only the relevant columns for calibration. This includes force measurements (force_x, force_y, force_z)
# and the corresponding sensor readings after initial correction (corrected_x, corrected_y, corrected_z).
data = load_cell_data[['force_x', 'force_y', 'force_z', 'corrected_x', 'corrected_y', 'corrected_z']]
data = data.dropna()  # Drop rows with NaN values

# Extracts the force measurements and the corresponding corrected sensor readings
forces = data[['force_x', 'force_y', 'force_z']].values
corrected_readings = data[['corrected_x', 'corrected_y', 'corrected_z']].values

# Identifies the no-load (zero force) sensor readings to use for offset correction.
# This step is crucial to eliminate any sensor bias or inherent offset.
no_load_readings = corrected_readings[forces[:,0] == 0, :][0]
corrected_readings_offset_corrected = corrected_readings - no_load_readings

# Calculates the calibration matrix using the Least Squares method. This matrix is used to transform
# sensor readings into force measurements. The calculation includes the offset correction.
K_offset_corrected, _, _, _ = lstsq(corrected_readings_offset_corrected, forces, rcond=None)

# Initializes Polynomial Regression for each axis (x, y, z) using the corrected sensor readings.
# This approach fits a second-degree polynomial to the data for each axis.
poly_fits = {}
coefficients_poly = {}
for axis, axis_name in enumerate(['x', 'y', 'z']):
    poly_fit = Polynomial.fit(corrected_readings[:, axis], forces[:, axis], 2)
    poly_fits[axis_name] = poly_fit
    coefficients_poly[axis_name] = poly_fit.convert().coef

# Calculates the estimated forces using the polynomial regression coefficients for each axis.
# This step applies the polynomial regression model to the sensor readings.
estimated_forces_poly = np.column_stack([
    np.polyval(coefficients_poly['x'][::-1], corrected_readings[:, 0]),
    np.polyval(coefficients_poly['y'][::-1], corrected_readings[:, 1]),
    np.polyval(coefficients_poly['z'][::-1], corrected_readings[:, 2])
])

# Calculates the estimated forces using the Least Squares method with the offset-corrected readings.
# matrix multiplication of the corrected readings with the calibration matrix.
grams_force_ls = np.dot(corrected_readings_offset_corrected, K_offset_corrected.T)

# Formats the output of the estimated forces to one decimal point for both Least Squares and Polynomial Regression methods.
formatted_grams_force_ls = np.around(grams_force_ls, 1)
formatted_grams_force_poly = np.around(estimated_forces_poly, 1)

# Function to calculate the absolute errors
def calculate_absolute_errors(actual, estimated):
    return np.abs(actual - estimated)

# Calculate absolute errors for Least Squares and Polynomial Regression methods
absolute_errors_ls = calculate_absolute_errors(forces, formatted_grams_force_ls)
absolute_errors_poly = calculate_absolute_errors(forces, formatted_grams_force_poly)

# Format the estimated forces and error matrices for printing
def format_array(arr):
    return [["\t" + f"{x:.1f}" for x in row] for row in arr]

# Prepare the data for display by formatting the estimated forces and error matrices
formatted_grams_force_ls_print = format_array(formatted_grams_force_ls)
formatted_grams_force_poly_print = format_array(formatted_grams_force_poly)
formatted_absolute_errors_ls = format_array(absolute_errors_ls)
formatted_absolute_errors_poly = format_array(absolute_errors_poly)

# Print the estimated forces and corresponding error matrices
print("Estimated forces (gramsForce) and Absolute Errors using Least Squares Method:")
print("\tCalculated X\tCalculated Y\tCalculated Z | \tError X\tError Y\tError Z")
for est, err in zip(formatted_grams_force_ls_print, formatted_absolute_errors_ls):
    print(" ".join(est), " | ", " ".join(err))

print("\nEstimated forces (gramsForce) and Absolute Errors using Polynomial Regression Method:")
print("\tCalculated X\tCalculated Y\tCalculated Z | \tError X\tError Y\tError Z")
for est, err in zip(formatted_grams_force_poly_print, formatted_absolute_errors_poly):
    print(" ".join(est), " | ", " ".join(err))

# Append the estimated forces and corresponding error matrices messages with headings in Markdown format
log_messages.append("### Estimated forces (gramsForce) and Absolute Errors using Least Squares Method:\n")
log_messages.append("| Calculated X | Calculated Y | Calculated Z | Error X | Error Y | Error Z |\n")
log_messages.append("|--------------|--------------|--------------|---------|---------|---------|\n")
for est, err in zip(formatted_grams_force_ls_print, formatted_absolute_errors_ls):
    log_messages.append("| " + " | ".join(est) + " | " + " | ".join(err) + " |\n")

log_messages.append("\n### Estimated forces (gramsForce) and Absolute Errors using Polynomial Regression Method:\n")
log_messages.append("| Calculated X | Calculated Y | Calculated Z | Error  X | Error Y | Error Z |\n")
log_messages.append("|--------------|--------------|--------------|---------|---------|---------|\n")
for est, err in zip(formatted_grams_force_poly_print, formatted_absolute_errors_poly):
    log_messages.append("| " + " | ".join(est) + " | " + " | ".join(err) + " |\n")
    
    
# Calculate MAE and RMSE for Least Squares and Polynomial Regression Methods
mae_ls = mean_absolute_error(forces, formatted_grams_force_ls)
rmse_ls = np.sqrt(mean_squared_error(forces, formatted_grams_force_ls))
mae_poly = mean_absolute_error(forces, formatted_grams_force_poly)
rmse_poly = np.sqrt(mean_squared_error(forces, formatted_grams_force_poly))

# Print the results
print("\nLeast Squares Method - Mean Absolute Error:", mae_ls, "g")
print("Least Squares Method - Root Mean Square Error:", rmse_ls, "g")
print("Polynomial Regression Method - Mean Absolute Error:", mae_poly, "g")
print("Polynomial Regression Method - Root Mean Square Error:", rmse_poly, "g")

# Append the results in Markdown format to log_messages
log_messages.append("\n### Results\n")
log_messages.append("| Method                              | Metric                | Value |\n")
log_messages.append("|-------------------------------------|-----------------------|-------|\n")
log_messages.append(f"| Least Squares Method                | Mean Absolute Error   | {mae_ls} g |\n")
log_messages.append(f"| Polynomial Regression Method        | Mean Absolute Error   | {mae_poly} g |\n")
log_messages.append(f"| Least Squares Method                | Root Mean Square Error| {rmse_ls} g |\n")
log_messages.append(f"| Polynomial Regression Method        | Root Mean Square Error| {rmse_poly} g |\n")

# Function to format numbers in engineering notation
def to_eng_notation(value):
    return "{:e}".format(value)

# Append the Least Squares Calibration Matrix in engineering notation
log_messages.append("## Least Squares Calibration Matrix:\n")
log_messages.append("| K_x | K_y | K_z |\n")
log_messages.append("|-----|-----|-----|\n")
for row in K_offset_corrected:
    formatted_row = "| " + " | ".join([to_eng_notation(elem) for elem in row]) + " |\n"
    log_messages.append(formatted_row)

# Append a section separator for readability
log_messages.append("\n---\n")

# Polynomial Regression Coefficients formatted as equations
log_messages.append("## Polynomial Regression Equations:\n")
for axis in ['x', 'y', 'z']:
    coeffs = coefficients_poly[axis]
    equation = "F_{}(R_{}) = ".format(axis, axis) + " + ".join([f"{to_eng_notation(coeff)}*R_{axis}^{i}" for i, coeff in enumerate(coeffs[::-1])])
    log_messages.append(f"$$ {equation} $$\n")


# Extract the base name of the CSV file and construct the log file name
csv_base_name = os.path.splitext(file_path)[0]  # This removes the file extension
log_file_name = f"{csv_base_name}_log.md"  # This adds "_log.txt" to the base file name

# Write the log messages to the log file
with open(log_file_name, 'w') as f:
    for message in log_messages:
        f.write(message)


###########################################################################################################

# Load the validation data needs readings differnet that those used for calibration to ensure that it matches the model. 
validation_file_path = 'validationDataCell2.csv'
validation_data = pd.read_csv(validation_file_path)

# Extract corrected sensor readings from the validation data
validation_corrected_readings = validation_data[['corrected_x', 'corrected_y', 'corrected_z']].values

# If validation data includes actual force measurements, use them
if 'force_x' in validation_data.columns and 'force_y' in validation_data.columns and 'force_z' in validation_data.columns:
    validation_forces = validation_data[['force_x', 'force_y', 'force_z']].values
else:
    # Otherwise, use zeros or appropriate placeholder values
    validation_forces = np.zeros_like(validation_corrected_readings)

# Perform offset correction on the validation data
validation_corrected_readings_offset_corrected = validation_corrected_readings - no_load_readings

# Use the calibration matrix to estimate forces from validation data
validation_estimated_forces_ls = np.dot(validation_corrected_readings_offset_corrected, K_offset_corrected.T)

# Use polynomial regression coefficients to estimate forces from validation data
validation_estimated_forces_poly = np.column_stack([
    np.polyval(coefficients_poly['x'][::-1], validation_corrected_readings[:, 0]),
    np.polyval(coefficients_poly['y'][::-1], validation_corrected_readings[:, 1]),
    np.polyval(coefficients_poly['z'][::-1], validation_corrected_readings[:, 2])
])

# Format the estimated forces and absolute errors for printing
formatted_validation_estimated_forces_ls = format_array(validation_estimated_forces_ls)
formatted_validation_estimated_forces_poly = format_array(validation_estimated_forces_poly)

# Function to format arrays for printing
def format_array(arr):
    return [["\t{:.1f}".format(x) for x in row] for row in arr]

# Calculate absolute errors for validation data using unformatted numerical arrays
validation_absolute_errors_ls = calculate_absolute_errors(validation_forces, validation_estimated_forces_ls)
validation_absolute_errors_poly = calculate_absolute_errors(validation_forces, validation_estimated_forces_poly)

# Format the error matrices for validation data
formatted_validation_absolute_errors_ls = format_array(validation_absolute_errors_ls)
formatted_validation_absolute_errors_poly = format_array(validation_absolute_errors_poly)

# Print the estimated forces and corresponding error matrices for validation data
print("\nValidation Data - Estimated Forces and Absolute Errors using Least Squares Method:")
for est, err in zip(formatted_validation_estimated_forces_ls, formatted_validation_absolute_errors_ls):
    print(" ".join(est), " | ", " ".join(err))

print("\nValidation Data - Estimated Forces and Absolute Errors using Polynomial Regression Method:")
for est, err in zip(formatted_validation_estimated_forces_poly, formatted_validation_absolute_errors_poly):
    print(" ".join(est), " | ", " ".join(err))
    
    
# Calculate MAE, RMSE, and Maximum Error for Least Squares method on validation data
mae_ls_validation = mean_absolute_error(validation_forces, validation_estimated_forces_ls)
rmse_ls_validation = np.sqrt(mean_squared_error(validation_forces, validation_estimated_forces_ls))
max_error_ls_validation = np.max(validation_absolute_errors_ls)

# Calculate MAE, RMSE, and Maximum Error for Polynomial Regression method on validation data
mae_poly_validation = mean_absolute_error(validation_forces, validation_estimated_forces_poly)
rmse_poly_validation = np.sqrt(mean_squared_error(validation_forces, validation_estimated_forces_poly))
max_error_poly_validation = np.max(validation_absolute_errors_poly)

# Print the results for validation data
print("\nValidation Data Performance - Least Squares Method:")
print("Mean Absolute Error:", mae_ls_validation)
print("Root Mean Square Error:", rmse_ls_validation)
print("Maximum Error:", max_error_ls_validation)

print("\nValidation Data Performance - Polynomial Regression Method:")
print("Mean Absolute Error:", mae_poly_validation)
print("Root Mean Square Error:", rmse_poly_validation)
print("Maximum Error:", max_error_poly_validation)

# Print Least Squares Calibration Matrix
print("\nLeast Squares Calibration Matrix:")
print(K_offset_corrected)

# Print Polynomial Regression Coefficients
print("\nPolynomial Regression Coefficients:")
for axis in ['x', 'y', 'z']:
    print(f"Axis {axis.upper()}: {coefficients_poly[axis]}")
