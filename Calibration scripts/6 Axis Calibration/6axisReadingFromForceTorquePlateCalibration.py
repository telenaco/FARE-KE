import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load your data
csv_file = "calibrationReadings.csv"
data = pd.read_csv(csv_file)

# Drop rows with any missing values
#data = data.dropna()

# Separate into calibration and validation sets
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

# Extract sensor readings (R) and known forces/torques (F) for calibration
R_cal = calibration_data.iloc[:, :6].values
F_cal = calibration_data.iloc[:, 6:].values

# Compute calibration matrices for each calibration point
K_cal, _, _, _ = lstsq(R_cal, F_cal, rcond=None)


#### 1. Function to calculate MAPE

def calculate_mape(estimated, actual):
    # Ensure we are not dividing by zero; filter out such cases
    mask = actual != 0
    actual_filtered = actual[mask]
    estimated_filtered = estimated[mask]
    
    # Calculate MAPE using filtered values
    mape = np.mean(np.abs((actual_filtered - estimated_filtered) / actual_filtered)) * 100
    return mape

# Calculate MAPE for the raw sensor readings against the known values
mape_raw = calculate_mape(R_cal, F_cal)  # R_cal here represents the force and torque calculated using the values from the calibrated 3 axis sensor readings

# Adjust the estimated forces/torques using the calibration matrix to compensate the errors
F_cal_est_initial = R_cal @ K_cal

# Compute MAPE for the  calibration matrix
mape_cal_initial = calculate_mape(F_cal_est_initial, F_cal)  

# Compute R-squared for the calibration matrix
r2_cal_initial = r2_score(F_cal, F_cal_est_initial)

# Compute R-squared for the raw readings
r2_raw = r2_score(F_cal, R_cal)

performance_comparison = pd.DataFrame({
    'Metric': ['MAPE (%)', 'R-squared'],
    'Raw Readings': [mape_raw, r2_raw],
    'Initial Calibration': [mape_cal_initial, r2_cal_initial]
})

print("Performance Metrics Comparison:")
print(performance_comparison)

######     2. Error Distribution Plots

# Calculate the errors of these "raw" readings against the known values
errors_raw = R_cal - F_cal

# Calculate the errors for the calibrated corrected values
errors_cal_initial = F_cal_est_initial - F_cal

# Plotting error distributions
plt.figure(figsize=(14, 6), dpi=300)

# Histogram for Raw vs  Calibration
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

#####      3. Scatter Plot of Estimated vs. Actual Forces/Torques

def plot_scatter_comparison(actual, estimated_raw, estimated_initial, axis_names):
    plt.figure(figsize=(14, 8),dpi=300)
    for i, axis_name in enumerate(axis_names, start=1):
        plt.subplot(2, 3, i)
        plt.scatter(actual[:, i-1], estimated_raw[:, i-1], alpha=0.5, label='Raw', marker='o')
        plt.scatter(actual[:, i-1], estimated_initial[:, i-1], alpha=0.5, label='Calibrated', marker='x')
        plt.plot(actual[:, i-1], actual[:, i-1], 'r--')  # Line for perfect agreement
        plt.title(f'Actual vs. Estimated {axis_name}')
        plt.xlabel('Actual')
        plt.ylabel('Estimated')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Axis names for your force/torque components
axis_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

# Plot scatter comparison
plot_scatter_comparison(F_cal, R_cal, F_cal_est_initial, axis_names)

########       4. Mean Absolute Error (MAE) by Axis

def calculate_mae_by_axis(errors_raw, errors_initial, axis_names):
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
    
# Calculate errors for raw and corrected against known values
errors_raw = R_cal - F_cal
errors_initial = F_cal_est_initial - F_cal

# Calculate and plot MAE by axis
calculate_mae_by_axis(errors_raw, errors_initial, axis_names)

##########################################

# Function to calculate crosstalk errors
def calculate_crosstalk_errors(readings, known_values):
    # Crosstalk occurs where known_values are zero
    crosstalk_mask = (known_values == 0)
    crosstalk_errors = readings * crosstalk_mask  
    return np.abs(crosstalk_errors)  # Return the absolute errors

# Calculate crosstalk errors before and after calibration
raw_crosstalk_errors = calculate_crosstalk_errors(R_cal, F_cal)
initial_calibrated_errors = calculate_crosstalk_errors(R_cal @ K_cal, F_cal)

# function to plot crosstalk errors
def plot_crosstalk_errors(axis, raw_errors, initial_errors):
    # Each subplot for an axis
    plt.figure(figsize=(12, 3), dpi=300)
    plt.bar(np.arange(len(raw_errors)) - 0.1, raw_errors, width=0.4, label='Raw')
    plt.bar(np.arange(len(initial_errors)) + 0.1, initial_errors, width=0.4, label='Calibrated')
    plt.title(f'Crosstalk Error on Axis {axis}')
    plt.xlabel('Measurement Instance')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

# Six axes (Fx, Fy, Fz, Mx, My, Mz), iterate and plot crosstalk errors
for axis in range(6):
    # Skipping the main axis reading and considering only crosstalk errors
    plot_crosstalk_errors(axis+1, 
                          raw_crosstalk_errors[:, axis], 
                          initial_calibrated_errors[:, axis])

##################################################

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
    
##########################################

# Function to isolate crosstalk readings
def isolate_crosstalk(readings, known_values):
    # Crosstalk occurs where known_values are zero
    crosstalk_mask = known_values == 0
    # Return readings where crosstalk occurs, else NaN to ignore in error calculation
    return np.where(crosstalk_mask, readings, np.nan)

# Function to calculate the crosstalk magnitude for the non-actuated axes 
def calculate_crosstalk_magnitude(readings, known_values):
    crosstalk_mask = known_values == 0
    crosstalk_readings = np.where(crosstalk_mask, readings, np.nan)  # Isolate crosstalk readings
    return np.nanmean(np.abs(crosstalk_readings), axis=0)  # Calculate mean absolute error ignoring NaNs

# Calculate crosstalk magnitude for raw readings
raw_crosstalk_magnitude = calculate_crosstalk_magnitude(R_cal, F_cal)

# Apply initial calibration matrix and calculate crosstalk magnitude
calibrated_crosstalk_magnitude_initial = calculate_crosstalk_magnitude(R_cal @ K_cal, F_cal)

# Output the crosstalk magnitude for comparison
print("Raw crosstalk magnitude:", raw_crosstalk_magnitude)
print("Calibrated crosstalk magnitude:", calibrated_crosstalk_magnitude_initial)


###################################################

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


