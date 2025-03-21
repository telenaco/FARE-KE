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

# Initialize the single-point calibration inverse matrix
single_point_K_inv = np.array([
    [5.97813989e-04, -1.10656949e-05,  5.95689594e-06],
    [3.92138487e-06,  5.87339468e-04,  5.36314213e-06],
    [6.93259796e-06,  1.40420654e-05,  5.97840978e-04]
])

# Estimate forces using the single-point calibration inverse matrix
estimated_forces_single_point = np.dot(corrected_readings_offset_corrected, single_point_K_inv.T)
formatted_grams_force_single_point = np.around(estimated_forces_single_point, 1)


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

# Calculate absolute errors for the single-point method
absolute_errors_single_point = calculate_absolute_errors(forces, formatted_grams_force_single_point)

# Prepare the data for display by formatting the estimated forces and error matrices
formatted_grams_force_single_point_print = format_array(formatted_grams_force_single_point)
formatted_absolute_errors_single_point = format_array(absolute_errors_single_point)
formatted_grams_force_ls_print = format_array(formatted_grams_force_ls)
formatted_grams_force_poly_print = format_array(formatted_grams_force_poly)
formatted_absolute_errors_ls = format_array(absolute_errors_ls)
formatted_absolute_errors_poly = format_array(absolute_errors_poly)

# Print the estimated forces and corresponding error matrices for the single-point method
print("\nEstimated forces (gramsForce) and Absolute Errors using Single-Point Method:")
print("\tCalculated X\tCalculated Y\tCalculated Z | \tError X\tError Y\tError Z")
for est, err in zip(formatted_grams_force_single_point_print, formatted_absolute_errors_single_point):
    print(" ".join(est), " | ", " ".join(err))

# Print the estimated forces and corresponding error matrices
print("Estimated forces (gramsForce) and Absolute Errors using Least Squares Method:")
print("\tCalculated X\tCalculated Y\tCalculated Z | \tError X\tError Y\tError Z")
for est, err in zip(formatted_grams_force_ls_print, formatted_absolute_errors_ls):
    print(" ".join(est), " | ", " ".join(err))

print("\nEstimated forces (gramsForce) and Absolute Errors using Polynomial Regression Method:")
print("\tCalculated X\tCalculated Y\tCalculated Z | \tError X\tError Y\tError Z")
for est, err in zip(formatted_grams_force_poly_print, formatted_absolute_errors_poly):
    print(" ".join(est), " | ", " ".join(err))

# ... [continue with calculating MAE and RMSE for all methods] ...

# Append the results to the log_messages list for documentation
log_messages.append("\n### Estimated forces (gramsForce) and Absolute Errors using Single-Point Method:\n")
log_messages.append("| Calculated X | Calculated Y | Calculated Z | Error X | Error Y | Error Z |\n")
log_messages.append("|--------------|--------------|--------------|---------|---------|---------|\n")
for est, err in zip(formatted_grams_force_single_point_print, formatted_absolute_errors_single_point):
    log_messages.append("| " + " | ".join(est) + " | " + " | ".join(err) + " |\n")

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
mae_single_point = mean_absolute_error(forces, formatted_grams_force_single_point)
rmse_single_point = np.sqrt(mean_squared_error(forces, formatted_grams_force_single_point))
mae_ls = mean_absolute_error(forces, formatted_grams_force_ls)
rmse_ls = np.sqrt(mean_squared_error(forces, formatted_grams_force_ls))
mae_poly = mean_absolute_error(forces, formatted_grams_force_poly)
rmse_poly = np.sqrt(mean_squared_error(forces, formatted_grams_force_poly))

# Print the results for all three methods
print("\nSingle-Point Method - Mean Absolute Error:", mae_single_point, "g")
print("Single-Point Method - Root Mean Square Error:", rmse_single_point, "g")
print("Least Squares Method - Mean Absolute Error:", mae_ls, "g")
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
log_messages.append(f"| Single-Point Method                 | Mean Absolute Error   | {mae_single_point} g |\n")
log_messages.append(f"| Single-Point Method                 | Root Mean Square Error| {rmse_single_point} g |\n")

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

validation_estimated_forces_single_point = np.dot(validation_corrected_readings_offset_corrected, single_point_K_inv.T)


# Format the estimated forces and absolute errors for printing
formatted_validation_estimated_forces_single_point = format_array(validation_estimated_forces_single_point)
formatted_validation_estimated_forces_ls = format_array(validation_estimated_forces_ls)
formatted_validation_estimated_forces_poly = format_array(validation_estimated_forces_poly)

# Function to format arrays for printing
def format_array(arr):
    return [["\t{:.1f}".format(x) for x in row] for row in arr]

# Calculate absolute errors for validation data using unformatted numerical arrays
validation_absolute_errors_single_point = calculate_absolute_errors(validation_forces, validation_estimated_forces_single_point)
validation_absolute_errors_ls = calculate_absolute_errors(validation_forces, validation_estimated_forces_ls)
validation_absolute_errors_poly = calculate_absolute_errors(validation_forces, validation_estimated_forces_poly)

# Format the error matrices for validation data
formatted_validation_absolute_errors_ls = format_array(validation_absolute_errors_ls)
formatted_validation_absolute_errors_poly = format_array(validation_absolute_errors_poly)
formatted_validation_absolute_errors_single_point = format_array(validation_absolute_errors_single_point)


# Print the estimated forces and corresponding error matrices for validation data
print("\nValidation Data - Estimated Forces and Absolute Errors using Least Squares Method:")
for est, err in zip(formatted_validation_estimated_forces_ls, formatted_validation_absolute_errors_ls):
    print(" ".join(est), " | ", " ".join(err))

print("\nValidation Data - Estimated Forces and Absolute Errors using Polynomial Regression Method:")
for est, err in zip(formatted_validation_estimated_forces_poly, formatted_validation_absolute_errors_poly):
    print(" ".join(est), " | ", " ".join(err))
    
# Add printing for single-point calibration errors
print("\nValidation Data - Estimated Forces and Absolute Errors using Single-Point Calibration Method:")
for est, err in zip(formatted_validation_estimated_forces_single_point, formatted_validation_absolute_errors_single_point):
    print(" ".join(est), " | ", " ".join(err))

    
    
# Calculate MAE, RMSE, and Maximum Error for Least Squares method on validation data
mae_ls_validation = mean_absolute_error(validation_forces, validation_estimated_forces_ls)
rmse_ls_validation = np.sqrt(mean_squared_error(validation_forces, validation_estimated_forces_ls))
max_error_ls_validation = np.max(validation_absolute_errors_ls)

# Calculate MAE, RMSE, and Maximum Error for Polynomial Regression method on validation data
mae_poly_validation = mean_absolute_error(validation_forces, validation_estimated_forces_poly)
rmse_poly_validation = np.sqrt(mean_squared_error(validation_forces, validation_estimated_forces_poly))
max_error_poly_validation = np.max(validation_absolute_errors_poly)

mae_single_point_validation = mean_absolute_error(validation_forces, validation_estimated_forces_single_point)
rmse_single_point_validation = np.sqrt(mean_squared_error(validation_forces, validation_estimated_forces_single_point))
max_error_single_point_validation = np.max(validation_absolute_errors_single_point)

# Print the results for validation data
print("\nValidation Data Performance - Least Squares Method:")
print("Mean Absolute Error:", mae_ls_validation)
print("Root Mean Square Error:", rmse_ls_validation)
print("Maximum Error:", max_error_ls_validation)

print("\nValidation Data Performance - Polynomial Regression Method:")
print("Mean Absolute Error:", mae_poly_validation)
print("Root Mean Square Error:", rmse_poly_validation)
print("Maximum Error:", max_error_poly_validation)

print("\nValidation Data Performance - Single Point:")
print("Mean Absolute Error:", mae_single_point_validation, "g")
print("Root Mean Square Error:", rmse_single_point_validation, "g")
print("Maximum Error:", max_error_single_point_validation, "g")

# Print Least Squares Calibration Matrix
print("\nLeast Squares Calibration Matrix:")
print(K_offset_corrected)

# Print Polynomial Regression Coefficients
print("\nPolynomial Regression Coefficients:")
for axis in ['x', 'y', 'z']:
    print(f"Axis {axis.upper()}: {coefficients_poly[axis]}")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Construct the DataFrame
errors = {
    'load_x': [0, -60, -285, -465, -1055, -1925, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'load_y': [0, 0, 0, 0, 0, 0, 0, 60, 285, 465, 1055, 1925, 0, 0, 0, 0, 0, 0],
    'load_z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -60, -285, -465, -1055, -1925],
    'error_x_least_squares': [4.9, 6.1, 8.0, 8.3, 12.5, 18.7, 4.9, 5.8, 9.4, 12.4, 22.8, 41.0, 4.9, 4.8, 5.4, 6.3, 8.5, 9.2],
    'error_y_least_squares': [0.4, 1.0, 0.6, 0.6, 1.3, 3.2, 0.4, 0.8, 1.9, 1.5, 5.4, 15.5, 0.4, 0.3, 1.6, 2.0, 7.4, 16.2],
    'error_z_least_squares': [1.8, 3.2, 8.9, 14.1, 32.8, 67.3, 1.8, 1.1, 0.7, 1.9, 10.0, 36.3, 1.8, 3.1, 4.6, 4.8, 8.0, 12.8],
    'error_x_polynomial': [5.8, 7.2, 9.8, 10.8, 18.3, 31.9, 5.8, 5.6, 5.0, 4.6, 4.2, 6.4, 5.8, 4.8, 2.5, 1.0, 4.7, 15.7],
    'error_y_polynomial': [7.3, 7.6, 6.1, 5.3, 3.3, 1.0, 7.3, 4.5, 0.8, 2.3, 2.0, 22.5, 7.3, 5.9, 2.2, 0.2, 12.0, 30.3],
    'error_z_polynomial': [2.1, 1.3, 2.3, 5.7, 18.8, 44.9, 2.1, 1.3, 2.0, 5.1, 10.8, 4.7, 2.1, 0.1, 4.2, 5.3, 5.9, 10.4],
    'error_x_single_point': [5.2, 6.1, 7.8, 8.0, 11.8, 17.3, 4.9, 5.9, 9.5, 12.6, 23.4, 42.2, 4.9, 4.6, 4.5, 4.9, 5.4, 3.4],
    'error_y_single_point': [0.4, 1.2, 1.2, 1.6, 3.7, 7.4, 0.4, 0.8, 2.1, 1.9, 6.2, 16.9, 0.4, 0.3, 2.0, 2.7, 9.0, 19.0],
    'error_z_single_point': [1.8, 3.3, 9.4, 14.9, 34.5, 70.4, 1.8, 1.1, 1.0, 2.3, 10.9, 37.9, 1.8, 3.0, 4.5, 4.5, 7.3, 11.6]
}


df = pd.DataFrame(errors)

# Calculate the sum of the loads
df['load_sum'] = abs(df['load_x']) + abs(df['load_y']) + abs(df['load_z'])  # Use absolute values to ensure positive denominators

# Drop rows where the load sum is zero BEFORE calculating error percentages
df = df[df['load_sum'] != 0]

# Create a copy of the DataFrame for the second plot with error values in grams
df_grams = df.copy()

# Calculate error percentage relative to the sum of loads, replace error values with percentages
error_columns = df.columns[3:12]  # Adjust based on your DataFrame structure
for col in error_columns:
    # Calculate percentage; handle division by zero by replacing with zero (or handle as needed)
    df[col] = np.where(df['load_sum'] == 0, 0, df[col] / df['load_sum'] * 100)
    # Format the percentage with two decimal points
    df[col] = df[col].map('{:.2f}'.format)

# Drop the 'load_sum' column as it's no longer needed
df.drop('load_sum', axis=1, inplace=True)
df_grams.drop('load_sum', axis=1, inplace=True)

# Custom headers
main_header = ['', 'Load(g)', '', '', 'Least Squares Error', '', '', 'Polynomial Error', '', '', 'Single Point Error', '']
sub_header = ['x', 'y', 'z'] * 4  # This will be the second header row

# Create the first plot with error percentages
fig1, ax1 = plt.subplots(figsize=(12, 8))
table_data1 = [main_header, sub_header] + df.values.tolist()
table1 = ax1.table(cellText=table_data1, loc='center', cellLoc='center')

# Convert percentage strings back to float for coloring
percentage_values = df.iloc[:, 3:].applymap(lambda x: float(x)).to_numpy()

# Normalize these percentage values
normalized_errors = (percentage_values - np.min(percentage_values)) / (np.max(percentage_values) - np.min(percentage_values))

# Create a colormap
cmap = plt.cm.coolwarm

# Color the cells based on the normalized error percentages
for i in range(df.shape[0]):
    for j in range(3, df.shape[1]):  # Skip the load columns for coloring
        cell = table1.get_celld()[(i + 2, j)]  # Offset by 2 for the header rows
        cell.set_facecolor(cmap(normalized_errors[i, j - 3]))

# Modify table properties
table1.auto_set_font_size(False)
table1.set_fontsize(14)
table1.scale(1.0, 2.0)

# Format main header row
for (i, label) in enumerate(main_header):
    cell = table1.get_celld()[(0, i)]
    if label:  # Check if the cell should have a label
        cell.set_text_props(text=label)
        cell.set_facecolor('lightgrey')
        cell.set_edgecolor('black')
    else:  # If no label, make invisible
        cell.set_visible(False)
        # Set border lines of adjacent cells to white to simulate merging
        if i > 0:
            table1.get_celld()[(0, i - 1)].visible_edges = 'open'
        if i < len(main_header) - 1:
            table1.get_celld()[(0, i + 1)].visible_edges = 'open'

# Format sub header row
for (i, label) in enumerate(sub_header):
    cell = table1.get_celld()[(1, i)]
    cell.set_text_props(text=label)
    cell.set_facecolor('lightgrey')
    cell.set_edgecolor('black')

# Hide axes
ax1.axis('off')
ax1.set_title('Error Percentages (%)', fontsize=14)


# Create the second plot with error values in grams
fig2, ax2 = plt.subplots(figsize=(12, 8))
table_data2 = [main_header, sub_header] + df_grams.values.tolist()
table2 = ax2.table(cellText=table_data2, loc='center', cellLoc='center')

# Get the error values as a numpy array
error_values = df_grams.iloc[:, 3:].to_numpy()

# Normalize the error values
normalized_errors_grams = (error_values - np.min(error_values)) / (np.max(error_values) - np.min(error_values))

# Color the cells based on the normalized error values
for i in range(df_grams.shape[0]):
    for j in range(3, df_grams.shape[1]):  # Skip the load columns for coloring
        cell = table2.get_celld()[(i + 2, j)]  # Offset by 2 for the header rows
        cell.set_facecolor(cmap(normalized_errors_grams[i, j - 3]))

# Modify table properties
table2.auto_set_font_size(False)
table2.set_fontsize(14)
table2.scale(1.0, 2.0)

# Format main header row
for (i, label) in enumerate(main_header):
    cell = table2.get_celld()[(0, i)]
    if label:  # Check if the cell should have a label
        cell.set_text_props(text=label)
        cell.set_facecolor('lightgrey')
        cell.set_edgecolor('black')
    else:  # If no label, make invisible
        cell.set_visible(False)
        # Set border lines of adjacent cells to white to simulate merging
        if i > 0:
            table2.get_celld()[(0, i - 1)].visible_edges = 'open'
        if i < len(main_header) - 1:
            table2.get_celld()[(0, i + 1)].visible_edges = 'open'

# Format sub header row
for (i, label) in enumerate(sub_header):
    cell = table2.get_celld()[(1, i)]
    cell.set_text_props(text=label)
    cell.set_facecolor('lightgrey')
    cell.set_edgecolor('black')

# Hide axes
ax2.axis('off')
ax2.set_title('Error Values (grams)', fontsize=14)

# Display the plots
plt.show()