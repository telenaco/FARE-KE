import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
import seaborn as sns
from IPython.display import display, Image



# Set the display option to show all columns
pd.set_option('display.max_columns', None)

# Load the data from the CSV file
data_df = pd.read_csv("Calibration values gmF.csv")
data_df.columns = [col.strip() for col in data_df.columns]  # Strip potential whitespace

def linear_analysis(data_df, title_suffix):
    # Initialize linear regression model
    model = LinearRegression()

    # Store metrics
    metrics = {'Load Cell': [], 'Slope': [], 'Intercept': [], 'R-Squared': []}

    # Define the size of the figure
    plt.figure(figsize=(4, 3.5))  # Adjust the height (4) if necessary
    
    # Fit linear models for each load cell
    for load_cell in ["tal220", "degraw", "CZL635"]:
        # Drop rows with NaN for the current load cell
        subset_data = data_df[["Weight", load_cell]].dropna()

        X = subset_data[["Weight"]]
        y = subset_data[load_cell]

        model.fit(X, y)
        predictions = model.predict(X)

        metrics['Load Cell'].append(load_cell)
        metrics['Slope'].append(model.coef_[0])
        metrics['Intercept'].append(model.intercept_)
        metrics['R-Squared'].append(model.score(X, y))

        # Plot the observed data
        plt.scatter(subset_data["Weight"], subset_data[load_cell], label=f"{load_cell} Observed", marker='o')
        # Plot the linear fit
        plt.plot(subset_data["Weight"], predictions, label=f"{load_cell} Linear Fit", linestyle='--')

        plt.xlabel('gf', fontsize=10)
        plt.ylabel('Digital Value Offset by Zero Adjustment', fontsize=10)
        plt.title(f'{title_suffix}', fontsize=11)
        plt.legend(fontsize=9)
        plt.grid(True)
        plt.tight_layout()
        
    # Save the individual plot
    plt.savefig(f"{title_suffix}.png", format='png', dpi=300)
    plt.show()

    # Display metrics
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    
    
def calculate_accuracy(data_df, title_suffix):
    model = LinearRegression()

    metrics = {
        'Load Cell': [], 
        'Max gf of Error': [], 
        'Accuracy Percentage': []
    }

    total_weight_range = data_df["Weight"].max() - data_df["Weight"].min()

    # Since we know data_df will only have two columns ('Weight' and the load cell name),
    # we can get the load cell name by subtracting the set of all columns and the set with just "Weight"
    load_cell = list(set(data_df.columns) - set(["Weight"]))[0]

    # Drop rows with NaN for the current load cell
    subset_data = data_df[["Weight", load_cell]].dropna()

    X = subset_data[["Weight"]]
    y = subset_data[load_cell]

    model.fit(X, y)
    predictions = model.predict(X)

    conversion_factor = 1/model.coef_[0]  # Digital value to grams
    residuals_in_g = (y - predictions) * conversion_factor  # Convert residuals to grams

    max_residual_g = residuals_in_g.abs().max()
    accuracy_percentage = ((total_weight_range - max_residual_g) / total_weight_range) * 100

    metrics['Load Cell'].append(load_cell)
    metrics['Max gf of Error'].append(max_residual_g)
    metrics['Accuracy Percentage'].append(accuracy_percentage)

    metrics_df = pd.DataFrame(metrics)
    print(f"Accuracy Metrics ({title_suffix}):")
    print(metrics_df)
    print("\n")

# 1. True Zero Adjustment
true_zero = {}
for load_cell in ["tal220", "degraw", "CZL635"]:
    true_zero[load_cell] = (data_df.iloc[6][load_cell] + data_df.iloc[7][load_cell]) / 2
true_zero_data = data_df.copy()
for load_cell in ["tal220", "degraw", "CZL635"]:
    true_zero_data[load_cell] -= true_zero[load_cell]
linear_analysis(true_zero_data, "Zero Offset")

# 2. Midpoint Zero Adjustment
midpoint_zero_data = data_df.copy()
for load_cell in ["tal220", "degraw", "CZL635"]:
    # Calculate the average of the two zero readings
    midpoint_value = (data_df.iloc[6][load_cell] + data_df.iloc[7][load_cell]) / 2
    # Replace both zero readings with the average
    midpoint_zero_data.loc[midpoint_zero_data["Weight"] == 0, load_cell] = midpoint_value
linear_analysis(midpoint_zero_data, "Midpoint Zero Adjustment")

# 3. Ignoring Zero Readings
data_without_zeros = data_df[data_df["Weight"] != 0].copy()
linear_analysis(data_without_zeros, "Ignoring Zero Readings")

# Assuming you have the data loaded in a DataFrame named data_df
load_cells = ['tal220', 'degraw', 'CZL635']

# Set the color palette to "colorblind"
colors = sns.color_palette("colorblind", len(load_cells))

# For each calibration method
for cell in load_cells:
    calculate_accuracy(true_zero_data[['Weight', cell]].dropna(), f"{cell} - With True Zero Offset")
    calculate_accuracy(midpoint_zero_data[['Weight', cell]].dropna(), f"{cell} - With Midpoint Zero Adjustment")
    calculate_accuracy(data_without_zeros[['Weight', cell]].dropna(), f"{cell} - Ignoring Zero Readings")


############### Combine images ########################

from PIL import Image

def combine_images1(titles, output_filename):
    # Open the images
    images = [Image.open(title + ".png") for title in titles]

    # Determine the total width and the max height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create an empty image with the determined width and height
    combined_img = Image.new('RGB', (total_width, max_height))

    # Paste each image into the combined image
    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    combined_img.save(output_filename)

# Sample usage:
titles = ["Zero Offset", "Midpoint Zero Adjustment", "Ignoring Zero Readings"]
combine_images1(titles, "combined_image.png")

######################################################################


############## Calibration for the three load cells #################

def calibration_analysis(data_df, reading_column, filename):
    data = data_df.copy()    
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Calculate the zero offset for the reading when weight is 0
    zero_offset = data[data['Weight'] == 0][reading_column].mean()
    
    # Adjust weight values by subtracting the zero offset for single scaling factor calculation
    x = data[reading_column].values
    x_adjusted = data[reading_column].values - zero_offset  
    y = data['Weight'].values  

    # Handle potential division by zero for single scaling factor
    ratios = np.where(x_adjusted != 0, y / x_adjusted, 0)
    scaling_factor = np.median(ratios)
    
    # Predictions using the scaling factor on adjusted weight
    y_pred_single_scale = x_adjusted * scaling_factor
    
    # Linear Regression using original data
    linear_regressor = LinearRegression()
    linear_regressor.fit(x.reshape(-1, 1), y)
    y_pred_linear = linear_regressor.predict(x.reshape(-1, 1))

    # Polynomial Regression (2nd Degree) using original data
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    poly_regressor = LinearRegression()
    poly_regressor.fit(x_poly, y)
    y_pred_poly = poly_regressor.predict(x_poly)
    
    # Offset values for better visualization
    offset_linear = y_pred_linear + 5
    offset_poly = y_pred_poly - 5
    
    # Define the size of the figure
    plt.figure(figsize=(4, 3.5))  # Adjust the height (4) if necessary
    
    # Plotting the results
    plt.scatter(x, y, color='red', label='Data Points', marker='o')
    plt.plot(x, y_pred_single_scale, color='purple', label='Single Scaling Factor', linestyle='dashed', linewidth=2, alpha=0.7)
    plt.plot(x, offset_linear, color='blue', label='Linear Regression', marker='^', linewidth=1.5, alpha=0.8)
    plt.plot(x, offset_poly, color='green', label='Polynomial Regression', marker='s', linewidth=1, alpha=0.9)
    plt.title(f'Calibration of {reading_column}', fontsize=11)
    plt.xlabel('Reading', fontsize=10)
    plt.ylabel('gf', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


    # Error calculations
    mse_single_scale = mean_squared_error(y, y_pred_single_scale)
    mse_linear = mean_squared_error(y, y_pred_linear)
    mse_poly = mean_squared_error(y, y_pred_poly)

    r2_single_scale = r2_score(y, y_pred_single_scale)
    r2_linear = r2_score(y, y_pred_linear)
    r2_poly = r2_score(y, y_pred_poly)

    print(f"Mean Squared Error (Single Scaling Factor): {mse_single_scale}")
    print(f"Mean Squared Error (Linear Regression): {mse_linear}")
    print(f"Mean Squared Error (Polynomial Regression): {mse_poly}")

    # Determine which method is best
    min_mse = min(mse_linear, mse_poly, mse_single_scale)
    if min_mse == mse_linear:
        print("Linear Regression is the best fit for the data.")
    elif min_mse == mse_poly:
        print("Polynomial Regression is the best fit for the data.")
    else:
        print("Single Scaling Factor is the best fit for the data.")

    print(f"R^2 value (Single Scaling Factor): {r2_single_scale}")
    print(f"R^2 value (Linear Regression): {r2_linear}")
    print(f"R^2 value (Polynomial Regression): {r2_poly}")

    # Print equations
    print(f"Single Scaling Factor: y = {scaling_factor}x ")
    print(f"Linear Regression Equation: y = {linear_regressor.coef_[0]}x + {linear_regressor.intercept_}")
    print(f"Polynomial Regression Equation: y = {poly_regressor.coef_[2]}x^2 + {poly_regressor.coef_[1]}x + {poly_regressor.intercept_}")

    # Compute the RMSE for each calibration technique
    rmse_single_scale = np.sqrt(mse_single_scale)
    rmse_linear = np.sqrt(mse_linear)
    rmse_poly = np.sqrt(mse_poly)
    
    print(f"Root Mean Squared Error (Single Scaling Factor): {rmse_single_scale} gf")
    print(f"Root Mean Squared Error (Linear Regression): {rmse_linear} gf")
    print(f"Root Mean Squared Error (Polynomial Regression): {rmse_poly} gf")
    print("============================================")
 
    return {
        "mse_single_scale": mse_single_scale,
        "mse_linear": mse_linear,
        "mse_poly": mse_poly,
        "r2_single_scale": r2_single_scale,
        "r2_linear": r2_linear,
        "r2_poly": r2_poly,
        "rmse_single_scale": rmse_single_scale,
        "rmse_linear": rmse_linear,
        "rmse_poly": rmse_poly,
        "zero_offset": zero_offset,
        "predictions_single_scale": y_pred_single_scale,
        "predictions_linear": y_pred_linear,
        "predictions_poly": y_pred_poly,
        "poly_regressor": poly_regressor
    }

# List to store filenames of individual plots
filenames = []

for idx, cell in enumerate(load_cells):
    # Extract relevant columns for Weight and load cell readings
    sub_data = data_df[['Weight', cell]].dropna()
    
    # Define a filename for this plot
    filename = f"calibration_plot_{idx}.png"
    filenames.append(filename)
    
    # Call the calibration_analysis function directly with the data subset and filename
    results = calibration_analysis(sub_data, cell, filename)
    
    # Generate predicted data using each calibration method
    zero_offset = results["zero_offset"]
    
    single_scaling_data = pd.DataFrame({
        'Weight': sub_data['Weight'].values,
        cell: (sub_data[cell].values - zero_offset) * results['mse_single_scale']
    })
    
    # Linear Regression
    linear_data = pd.DataFrame({
        'Weight': sub_data['Weight'].values,
        cell: results['mse_linear'] * sub_data[cell].values + results['zero_offset']
    })
    
    # Polynomial Regression
    poly_data = pd.DataFrame({
        'Weight': sub_data['Weight'].values,
        cell: results['mse_poly'] * sub_data[cell].values + results['zero_offset']
    })


############### Combine images ########################

def combine_images(filenames, output_filename):
    # Open the images
    images = [Image.open(filename) for filename in filenames]

    # Determine the total width and the max height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create an empty image with the determined width and height
    combined_img = Image.new('RGB', (total_width, max_height))

    # Paste each image into the combined image
    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    combined_img.save(output_filename)
    
#########################################################

# Once all plots are generated and saved, combine them into a single image
combine_images(filenames, "combined_calibration.png")


############## no load test ###########################

# Load the noise data
noise_data = pd.read_csv("noLoad noise - connection 0 - UART COM3.csv")

# Strip potential whitespace from column names
noise_data.columns = [col.strip() for col in noise_data.columns]

# Mapping between calibration data column names and noise data column names
column_mapping = {
    "tal220": "TAL220 ()",
    "degraw": "Degraw ()",
    "CZL635": "CZL635 ()"
}

results = []

# Define a dictionary to store polynomial regression coefficients for each load cell
polynomial_coeffs = {
    "CZL635 ()": {"a": None, "b": None, "c": None},  # Fill these values after running calibration_analysis
    "Degraw ()": {"a": None, "b": None, "c": None},  # Fill these values after running calibration_analysis
    "TAL220 ()": {"a": None, "b": None, "c": None}   # Fill these values after running calibration_analysis
}

# Function to convert digital readings to grams using polynomial regression
def convert_to_grams(cell, value):
    a = polynomial_coeffs[cell]['a']
    b = polynomial_coeffs[cell]['b']
    c = polynomial_coeffs[cell]['c']
    return a * value**2 + b * value + c

# Define a dummy filename
dummy_filename = "dummy_plot_ignore.png"

# Extract polynomial coefficients after running calibration_analysis for each cell
for calibration_cell, noise_cell in column_mapping.items():
    results = calibration_analysis(data_df[['Weight', calibration_cell]].dropna(), calibration_cell, dummy_filename)
    
    # Extract the polynomial coefficients and intercept
    polynomial_coeffs[noise_cell]["a"] = results["poly_regressor"].coef_[2]
    polynomial_coeffs[noise_cell]["b"] = results["poly_regressor"].coef_[1]
    polynomial_coeffs[noise_cell]["c"] = results["poly_regressor"].intercept_
    


# Create an empty A4-sized plot
fig, ax = plt.subplots(figsize=(8.27, 3.5))

# Custom color palette
colors = ['#1f77b4', '#ff7f0e', '#8c564b']

# Plot each signal centered around zero for readings between indices 400 and 600
for index, calibration_cell in enumerate(load_cells):
    noise_cell = column_mapping[calibration_cell]  # Get the correct column name from the mapping
    readings = noise_data[noise_cell].iloc[400:600]  # Only take readings between indices 400 and 600
    
    # Convert readings to grams
    readings_grams = convert_to_grams(noise_cell, readings)
    
    # Center the readings gf around zero by subtracting the mean
    centered_readings_grams = readings_grams - readings_grams.mean()

    # Plot the centered readings gf using the custom color palette
    ax.plot(centered_readings_grams, label=noise_cell, color=colors[index])

# Set title, labels, grid, etc.
ax.set_title('Noise Signals')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('gf')
ax.legend(fontsize=8)
ax.grid(True)
plt.tight_layout()
plt.savefig("Noise Signal")
plt.show()

# Summary statistics, plotting, SNR calculation, and additional metrics
results_list = []  

def sanitize_filename(filename):
    """Remove special characters and spaces from the filename."""
    return ''.join(e for e in filename if e.isalnum())
    
# Lists to store the filenames of the saved images for later combination
histogram_filenames = []
fft_filenames = []
autocorr_filenames = []

# Summary statistics, plotting, SNR calculation, and additional metrics
for calibration_cell in load_cells:
    noise_cell = column_mapping[calibration_cell]  # Get the correct column name from the mapping
    readings = noise_data[noise_cell] - noise_data[noise_cell].mean()  # Centering data around zero
    sanitized_name = sanitize_filename(noise_cell)
    
    # Convert readings to grams for relevant statistics
    mean_val_g = convert_to_grams(noise_cell, readings.mean())
    std_val_g = convert_to_grams(noise_cell, readings.std())
    min_val_g = convert_to_grams(noise_cell, readings.min())
    max_val_g = convert_to_grams(noise_cell, readings.max())
    peak_to_peak_g = max_val_g - min_val_g
    
    # Calculate statistics for digital readings
    mean_val = readings.mean()
    std_val = readings.std()
    min_val = readings.min()
    max_val = readings.max()
    peak_to_peak = max_val - min_val
    signal_power = mean_val**2
    noise_power = std_val**2
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
    
    results_list.append([mean_val, std_val, min_val, max_val, mean_val_g, std_val_g, min_val_g, max_val_g, peak_to_peak, peak_to_peak_g, snr_db])

        # Plot histogram
    hist_filename = f'histogram_{sanitized_name}.png'
    plt.figure(figsize=(4, 3.5))
    plt.hist(readings - mean_val, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Noise Distribution for {noise_cell}', fontsize=11)
    plt.xlabel('Noise', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(hist_filename, format='png', dpi=300)
    plt.close()  # Close the plot
    histogram_filenames.append(hist_filename)
    
    # Frequency-domain analysis
    fft_values = np.fft.fft(readings - mean_val)
    fft_filename = f'fft_{sanitized_name}.png'
    plt.figure(figsize=(4, 3.5))
    plt.plot(np.abs(fft_values)[:len(fft_values) // 2])
    plt.title(f'Frequency Spectrum for {noise_cell}', fontsize=11)
    plt.xlabel('Frequency', fontsize=10)
    plt.ylabel('Magnitude', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fft_filename, format='png', dpi=300)
    plt.close()  # Close the plot
    fft_filenames.append(fft_filename)
    
    # Autocorrelation plot
    autocorr = np.correlate(readings - mean_val, readings - mean_val, mode='full')[len(readings)-1:]
    autocorr_filename = f'autocorr_{sanitized_name}.png'
    plt.figure(figsize=(4, 3.5))
    plt.plot(autocorr)
    plt.title(f'Autocorrelation for {noise_cell}', fontsize=11)
    plt.xlabel('Lag', fontsize=10)
    plt.ylabel('Autocorrelation', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(autocorr_filename, format='png', dpi=300)
    plt.close()  # Close the plot
    autocorr_filenames.append(autocorr_filename)
    
# Strip the .png extension from each filename before passing to combine_images1
histogram_filenames_stripped = [filename.replace('.png', '') for filename in histogram_filenames]
fft_filenames_stripped = [filename.replace('.png', '') for filename in fft_filenames]
autocorr_filenames_stripped = [filename.replace('.png', '') for filename in autocorr_filenames]

# Combine the saved plots into three combined images
combine_images1(histogram_filenames_stripped, 'combined_histograms.png')
combine_images1(fft_filenames_stripped, 'combined_fft.png')
combine_images1(autocorr_filenames_stripped, 'combined_autocorr.png')

from PIL import Image as PILImage
from IPython.display import display, Image

# Display the combined images
display(Image(filename='combined_histograms.png'))
display(Image(filename='combined_fft.png'))
display(Image(filename='combined_autocorr.png'))

# Display results table
column_names = ['Mean', 'Standard Deviation', 'Min', 'Max', 'Mean (gf)', 'Standard Deviation (gf)', 'Min (gf)', 'Max (gf)', 'Peak-to-Peak', 'Peak-to-Peak (gf)', 'SNR (dB)']
results_df = pd.DataFrame(results_list, columns=column_names, index=load_cells)
print(results_df)

################## Overnight Drift Analysis gf ##################

# 1. Load the data from the CSV file
data = pd.read_csv('loading test overnight - connection 0 - UART COM3.csv')

data = data.drop(data.index[0:9000])

# Convert the UNIX timestamp from milliseconds to seconds
data['Second'] = (data['UNIX Timestamp (Milliseconds since 1970-01-01)'] / 1000).astype(int)

# Calculate the average of each load cell signal
avg_635 = data['635 ()'].mean()
avg_Degraw = data['Degraw ()'].mean()
avg_tal = data['tal ()'].mean()

# Subtract the average from each signal to center them around zero
data['635_centered'] = data['635 ()'] - avg_635
data['Degraw_centered'] = data['Degraw ()'] - avg_Degraw
data['tal_centered'] = data['tal ()'] - avg_tal

# Convert centered values to grams using the polynomial regression
data['635_centered_g'] = data['635_centered'].apply(lambda x: convert_to_grams("CZL635 ()", x))
data['Degraw_centered_g'] = data['Degraw_centered'].apply(lambda x: convert_to_grams("Degraw ()", x))
data['tal_centered_g'] = data['tal_centered'].apply(lambda x: convert_to_grams("TAL220 ()", x))

# Apply a moving average (rolling window) to smooth the data
window_size = 60
data['635_smoothed_g'] = data['635_centered_g'].rolling(window=window_size).mean()
data['Degraw_smoothed_g'] = data['Degraw_centered_g'].rolling(window=window_size).mean()
data['tal_smoothed_g'] = data['tal_centered_g'].rolling(window=window_size).mean()

# Plot the smoothed data gf
fig, ax1 = plt.subplots(figsize=(15, 6))

# Plotting temperature
ax1.set_xlabel('Second')
ax1.set_ylabel('temperature ()', color='red')
ax1.plot(data['Second'], data['temperature ()'], color='red', label='temperature ()')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Smoothed Load Cells (gf)', color='blue')
ax2.plot(data['Second'], data['635_smoothed_g'], color='blue', label='635_smoothed (gf)')
ax2.plot(data['Second'], data['Degraw_smoothed_g'], color='green', label='Degraw_smoothed (gf)')
ax2.plot(data['Second'], data['tal_smoothed_g'], color='purple', label='tal_smoothed (gf)')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend()

plt.title("Smoothed Data Visualization (gf) Using Moving Average")
plt.grid(True)
plt.show()

# Descriptive Statistics gf
descriptive_stats_g = data[['635_smoothed_g', 'Degraw_smoothed_g', 'tal_smoothed_g', 'temperature ()']].describe()

# Correlation Analysis gf
correlation_matrix_g = data[['635_smoothed_g', 'Degraw_smoothed_g', 'tal_smoothed_g', 'temperature ()']].corr()

# Drift Analysis gf
initial_635_g = data['635_smoothed_g'].iloc[0]
initial_Degraw_g = data['Degraw_smoothed_g'].iloc[0]
initial_tal_g = data['tal_smoothed_g'].iloc[0]

# drift calculation using the 60th smoothed value as the initial value (gf)
data['635_drift_g'] = data['635_smoothed_g'] - data['635_smoothed_g'].iloc[window_size-1]
data['Degraw_drift_g'] = data['Degraw_smoothed_g'] - data['Degraw_smoothed_g'].iloc[window_size-1]
data['tal_drift_g'] = data['tal_smoothed_g'] - data['tal_smoothed_g'].iloc[window_size-1]

# Convert seconds to minutes
data['Minutes'] = data['Second'] / 60

# Plotting drift gf with temperature
fig, ax = plt.subplots(figsize=(8.27, 3.5))  # A4 width in inches

# Plot the temperature first so it's behind the other lines
ax2 = ax.twinx()
ax2.plot(data['Minutes'], data['temperature ()'], color='red', linestyle='--', label='Temperature (°C)')
ax2.set_ylabel('Temperature (°C)', color='red', fontsize=8)  # Adjusted fontsize
ax2.tick_params(axis='y', labelcolor='red',  labelsize=8)  # Adjusted fontsize

# Now plot the drift values for each load cell on top of the temperature
ax.plot(data['Minutes'], data['635_drift_g'], label='635 Drift (gf)', color='blue')
ax.plot(data['Minutes'], data['Degraw_drift_g'], label='Degraw Drift (gf)', color='green')
ax.plot(data['Minutes'], data['tal_drift_g'], label='TAL Drift (gf)', color='purple')
ax.set_ylabel('Drift from Initial Reading (gf)', fontsize=8)  # Adjusted fontsize
ax.set_xlabel('Time (minutes)', fontsize=8)  # Updated xlabel to "Time (minutes)"
ax.tick_params(axis='both',  labelsize=8)  # Adjusted fontsize

# Splitting legends: one to the bottom left and the other to the bottom right
ax.legend(loc='lower left', fontsize=8)  # Adjusted fontsize and location
ax2.legend(loc='lower right', fontsize=8)  # Adjusted fontsize and location

plt.title("Creep vs Temperature", fontsize=8)  # Adjusted fontsize
plt.grid(True)
plt.tight_layout()
plt.show()

print(descriptive_stats_g)
print(correlation_matrix_g)

################## furhter temperature analysis ######################

# cheching if the noise on the temperature has any frequcny component
# nothing really to be found here after analysis

# Load the data
data = pd.read_csv("dataFromTemperatureSignalB - connection 0 - UART COM3.csv")

# Check the first few rows of the data
# Subtract the mean to center the signal around zero
centered_temp = data['b ()'] - data['b ()'].mean()

# Compute FFT
fft_values = np.fft.fft(centered_temp)

# Plot the magnitude spectrum without frequency values
plt.figure(figsize=(10, 6))
plt.plot(np.abs(fft_values)[:len(fft_values) // 2])
plt.title('Magnitude Spectrum for Temperature Signal')
plt.xlabel('FFT Index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# 1. Zoom into the Lower Frequencies
plt.figure(figsize=(10, 6))
plt.plot(np.abs(fft_values)[:500])  # Adjust this range as needed
plt.title('Zoomed Magnitude Spectrum for Temperature Signal')
plt.xlabel('FFT Index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

def compute_sampling_rate(micros):
    # Compute differences between consecutive micros
    differences = np.diff(micros)
    
    # Compute average difference
    avg_difference = np.mean(differences)
    
    # Invert the average difference to get the sampling rate
    sampling_rate = 1.0 / avg_difference  # Since micros is likely in microseconds, this will give samples per microsecond
    
    # Convert to samples per second (Hz) if needed
    sampling_rate_hz = sampling_rate * 1e6  # Convert from samples/microsecond to samples/second

    return sampling_rate_hz


micros = data['a ()'].values  # Assuming 'a ()' column contains the micro timestamps
sampling_rate = compute_sampling_rate(micros)
print(f"Computed Sampling Rate: {sampling_rate} Hz")

# 2. Spectrogram
from scipy.signal import spectrogram

frequencies, times, Sxx = spectrogram(centered_temp, fs=sampling_rate)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.colorbar(label='Intensity [dB]')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Temperature Signal')
plt.show()

# 3. Time-Domain Analysis
plt.figure(figsize=(10, 6))
plt.plot(data['Sample Number (1000 samples per second)'], centered_temp)
plt.title('Time-domain Representation of Temperature Signal')
plt.xlabel('Sample Number')
plt.ylabel('Centered Temperature')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Autocorrelation
autocorr_temp = np.correlate(centered_temp, centered_temp, mode='full')[len(centered_temp)-1:]
plt.figure(figsize=(10, 6))
plt.plot(autocorr_temp)
plt.title('Autocorrelation of Temperature Signal')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.tight_layout()
plt.show()



################## Repetition load on load cell on/off load ######################

# Load the CSV file into a DataFrame
load_test_data  = pd.read_csv('loadingTest - connection 0 - UART COM3.csv')

# Load cell columns to process
load_cells = ['635 ()', 'Degraw ()', 'tal ()']

# Plotting
plt.figure(figsize=(18, 12))

# Plot each load cell from original data
plt.subplot(3, 1, 1)
plt.plot(load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'], load_test_data['635 ()'], label='635', color='blue')
plt.legend()
plt.ylabel('635')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'], load_test_data['Degraw ()'], label='Degraw', color='green')
plt.legend()
plt.ylabel('Degraw')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'], load_test_data['tal ()'], label='TAL', color='purple')
plt.legend()
plt.ylabel('TAL')
plt.xlabel('UNIX Timestamp')
plt.grid(True)

plt.tight_layout()
plt.show()

############ standarize the widtdh of the loading an unloading events 

def middle_avg(data):
    """Calculate the middle 90% average of the data."""
    five_percent_len = len(data) // 20
    return data[five_percent_len:-five_percent_len].mean()

def middle_ten_percent(data):
    """Improved function to get the middle ten percent of the data."""
    start_idx = int(len(data) * 0.45)
    end_idx = int(len(data) * 0.55)
    return data[start_idx:end_idx]


def extend_to_max_length(data, max_length):
    """Improved function to extend data to max length with diagnostics and better handling."""
    iterations = 0
    while len(data) < max_length:
        iterations += 1
        middle_data = middle_ten_percent(data)
        
        # If the middle_data has the same length as the original data or if it's empty, break to avoid infinite loop
        if len(middle_data) == len(data) or len(middle_data) == 0:
            print(f"Warning: Middle data has length {len(middle_data)} while original data has length {len(data)}. Breaking...")
            break
        
        data = np.concatenate([data[:len(data)//2], middle_data, data[len(data)//2:]])
        
        # Safety break to prevent infinite loops in case of unexpected issues
        if iterations > 100:
            print("Warning: Too many iterations in extend_to_max_length. Breaking...")
            break
    return data[:max_length]

def simple_identify_intervals(load_cell_data, timestamps):
    threshold = (load_cell_data.min() + load_cell_data.max()) / 2
    above_threshold = load_cell_data > threshold
    below_threshold = ~above_threshold
    loading_intervals = []
    unloading_intervals = []
    in_loading = below_threshold[0]
    in_unloading = not in_loading
    start_idx = 0
    for i in range(1, len(load_cell_data)):
        if in_loading and above_threshold[i]:
            loading_intervals.append((start_idx, i))
            start_idx = i
            in_loading = False
            in_unloading = True
        elif in_unloading and below_threshold[i]:
            unloading_intervals.append((start_idx, i))
            start_idx = i
            in_loading = True
            in_unloading = False
    if in_loading:
        loading_intervals.append((start_idx, len(load_cell_data)))
    else:
        unloading_intervals.append((start_idx, len(load_cell_data)))
    return loading_intervals, unloading_intervals

def adjust_intervals(loading_intervals, unloading_intervals, data_length):
    """Adjust intervals to fit within data length and filter out intervals with only one data point."""
    
    new_loading_intervals = []
    new_unloading_intervals = []
    
    for i, (start, end) in enumerate(loading_intervals):
        if end > data_length - 1:
            end = data_length - 1
        if end - start > 1:  # Filtering out intervals with only one data point
            new_loading_intervals.append((start, end))
    
    for i, (start, end) in enumerate(unloading_intervals):
        if end > data_length - 1:
            end = data_length - 1
        if end - start > 1:  # Filtering out intervals with only one data point
            new_unloading_intervals.append((start, end))
    
    return new_loading_intervals, new_unloading_intervals

def standardize_load_cell_data_debug_updated(load_cell_data, loading_intervals, unloading_intervals, timestamps):
    """Updated function to standardize the loading and unloading events for a given load cell data with debugging."""
    max_interval_length = max(max([end - start for start, end in loading_intervals]), 
                              max([end - start for start, end in unloading_intervals]))
    new_data = []
    new_timestamps = []
    for i in range(max(len(loading_intervals), len(unloading_intervals))):
        if i < len(unloading_intervals):
            interval = unloading_intervals[i]
            interval_data = load_cell_data[interval[0]:interval[1]]
            extended_data = extend_to_max_length(interval_data, max_interval_length)
            new_data.extend(extended_data)
            interval_timestamps = timestamps[interval[0]:interval[1]]
            extended_timestamps = np.linspace(0, interval_timestamps[-1] - interval_timestamps[0], max_interval_length)
            new_timestamps.extend(extended_timestamps)
        if i < len(loading_intervals):
            interval = loading_intervals[i]
            interval_data = load_cell_data[interval[0]:interval[1]]
            extended_data = extend_to_max_length(interval_data, max_interval_length)
            new_data.extend(extended_data)
            interval_timestamps = timestamps[interval[0]:interval[1]]
            extended_timestamps = np.linspace(0, interval_timestamps[-1] - interval_timestamps[0], max_interval_length)
            new_timestamps.extend(extended_timestamps)
    
    # Adjust the new_timestamps to ensure it starts from 0 and increments based on average elapsed time
    average_elapsed_time = np.diff(timestamps).mean()
    new_timestamps = np.arange(0, average_elapsed_time * len(new_data), average_elapsed_time)
    return new_data, new_timestamps


def process_and_plot_load_cell(load_cell_name, color, subplot_position):
    


    # Extracting data for the load cell and timestamps
    load_cell_data = load_test_data[load_cell_name].values
    timestamps = load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'].values
    
    # Convert timestamps to seconds
    #timestamps_seconds = timestamps / 1000.0  # Convert milliseconds to seconds

    # Identify loading and unloading intervals
    loading_intervals, unloading_intervals = simple_identify_intervals(load_cell_data, timestamps)
    loading_intervals, unloading_intervals = adjust_intervals(loading_intervals, unloading_intervals, len(load_cell_data))

    # Standardizing the load cell data
    data_debug_updated, new_timestamps_debug_updated = standardize_load_cell_data_debug_updated(load_cell_data, loading_intervals, unloading_intervals, timestamps)

    # Ensure that the lengths of new_timestamps_debug_updated and data_debug_updated match before plotting
    if len(new_timestamps_debug_updated) > len(data_debug_updated):
        new_timestamps_debug_updated = new_timestamps_debug_updated[:-1]
    elif len(data_debug_updated) > len(new_timestamps_debug_updated):
        data_debug_updated = data_debug_updated[:-1]
        
    
    timestamps_minutes = new_timestamps_debug_updated /  (1000.0 * 60.0)  # Convert milliseconds to minutes


    # Create subplot
    ax = plt.subplot(3, 1, subplot_position)
    
    name_mapping = {
        'tal ()': 'TAL220 ()',
        'Degraw ()': 'Degraw ()',
        '635 ()': 'CZL635 ()'
    }
    
    # Convert the data_debug_updated values to grams
    data_in_grams = [convert_to_grams(name_mapping[load_cell_name], val) for val in data_debug_updated]
    
    # Plotting the data gf
    ax.plot(timestamps_minutes, data_in_grams, label=f"{load_cell_name} (gf)", color=color)
    
    # Plot adjustments for each subplot
    # ax.legend(fontsize=9)
    
    name_mapping = {
    'tal ()': 'TAL220',
    'Degraw ()': 'Degraw',
    '635 ()': 'CZL635'
    }
    
    ax.set_ylabel(f"{name_mapping[load_cell_name]} (gf)", fontsize=9)
    ax.grid(True)
    
    # Remove the x-axis for the top two subplots
    if subplot_position != 3:
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (minutes since start)', fontsize=10)
        
    conditions, transitions = plot_with_identified_transitions_corrected(load_cell_name, color, subplot_position, data_debug_updated, new_timestamps_debug_updated)
    return conditions, transitions, data_debug_updated

# Plot setup
plt.figure(figsize=(8.27, 6))  # A4 width in inches with proportional height

# Process and plot each load cell
process_and_plot_load_cell('635 ()', 'blue', 1)
process_and_plot_load_cell('Degraw ()', 'green', 2)
process_and_plot_load_cell('tal ()', 'purple', 3)

plt.suptitle('Repetition Test', fontsize=11)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusting layout to make the subplots flush with each other
plt.show()



###### plot with identify gaps 

def compute_average_for_conditions(conditions, transitions, data):
    averages = {}
    for i in range(len(conditions)):
        start_idx = int(transitions[i] + 0.05 * (transitions[i+1] - transitions[i]))
        end_idx = int(transitions[i+1] - 0.05 * (transitions[i+1] - transitions[i]))
        condition_data = data[start_idx:end_idx]
        avg_value = np.mean(condition_data)
        if conditions[i] not in averages:
            averages[conditions[i]] = []
        averages[conditions[i]].append(avg_value)
    return averages


def plot_with_identified_transitions_corrected(load_cell_name, color, subplot_position, data, timestamps):
    """
    Corrected version of the self-contained function to:
    1. Identify transitions in standardized data.
    2. Plot the data with appropriate highlights for loading and unloading conditions.
    
    Returns conditions and transitions.
    """
    def identify_transitions(data):
        threshold = (min(data) + max(data)) / 2
        above_threshold = data > threshold
        below_threshold = ~above_threshold

        if below_threshold[0]:
            conditions = ['unloading']
        else:
            conditions = ['loading']

        transitions = [0]

        for i in range(1, len(data)):
            if below_threshold[i] and conditions[-1] == 'loading':
                conditions.append('unloading')
                transitions.append(i)
            elif above_threshold[i] and conditions[-1] == 'unloading':
                conditions.append('loading')
                transitions.append(i)

        # Ensure to capture the end of the last condition
        if len(transitions) == len(conditions):
            transitions.append(len(data) - 1)

        return conditions, transitions
    
    # Identify transitions
    conditions, transitions = identify_transitions(data)
    
    # Convert timestamps to minutes
    timestamps_minutes = timestamps / (1000.0 * 60.0)

    # Create subplot
    ax = plt.subplot(3, 1, subplot_position)
    
    # Plotting the standardized data
    # ax.plot(timestamps_minutes, data, label=f'{load_cell_name}', color=color)
    
    # Highlight conditions using identified transitions
    for i in range(len(transitions) - 1):
        if conditions[i] == 'loading':
            ax.axvspan(timestamps_minutes[transitions[i]], timestamps_minutes[transitions[i+1]], alpha=0.2, color='green')
        else:
            ax.axvspan(timestamps_minutes[transitions[i]], timestamps_minutes[transitions[i+1]], alpha=0.2, color='red')

    name_mapping = {
        'tal ()': 'TAL220',
        'Degraw ()': 'Degraw',
        '635 ()': 'CZL635'
    }
    
    ax.set_ylabel(f"{name_mapping[load_cell_name]} (gf)", fontsize=10)
    ax.grid(True)
    
    if subplot_position != 3:
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (minutes since start)', fontsize=10)
    
    return conditions, transitions



# Plot setup
plt.figure(figsize=(8.27, 6))  # A4 width in inches with proportional height

averages = {}
average_grams = {}

# Given name_mapping to match the keys of the averages dictionary with the expected argument for convert_to_grams
name_mapping = {
    '635 ()': 'CZL635 ()',
    'Degraw ()': 'Degraw ()',
    'tal ()': 'TAL220 ()'
}

# Process and plot each load cell
for idx, load_cell in enumerate(load_cells, 1):
    conditions, transitions, data_debug_updated = process_and_plot_load_cell(load_cell, ['blue', 'green', 'purple'][idx-1], idx)
    avg = compute_average_for_conditions(conditions, transitions, data_debug_updated)
    averages[load_cell] = avg
    
    # Convert the averages for this load cell to grams
    avg_grams = {}
    for condition, values in avg.items():
        avg_grams[condition] = [convert_to_grams(name_mapping[load_cell], val) for val in values]
    average_grams[load_cell] = avg_grams

plt.suptitle('Repetition Test', fontsize=11)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusting layout to make the subplots flush with each other
plt.show()

print(average_grams)

########## visualize loading and unloading events 

# Extracting unloading averages for each load cell
czl635_unloading = averages['635 ()']['unloading']
degraw_unloading = averages['Degraw ()']['unloading']
tal_unloading = averages['tal ()']['unloading']

# Extracting loading averages for each load cell
czl635_loading = averages['635 ()']['loading']
degraw_loading = averages['Degraw ()']['loading']
tal_loading = averages['tal ()']['loading']

# Convert the average values to grams using the convert_to_grams function
czl635_unloading_g = [convert_to_grams('CZL635 ()', val) for val in czl635_unloading]
degraw_unloading_g = [convert_to_grams('Degraw ()', val) for val in degraw_unloading]
tal_unloading_g = [convert_to_grams('TAL220 ()', val) for val in tal_unloading]

czl635_loading_g = [convert_to_grams('CZL635 ()', val) for val in czl635_loading]
degraw_loading_g = [convert_to_grams('Degraw ()', val) for val in degraw_loading]
tal_loading_g = [convert_to_grams('TAL220 ()', val) for val in tal_loading]

# Shift the data 
czl635_unloading_g_shifted = [val + 2 - czl635_unloading_g[0] for val in czl635_unloading_g]
degraw_unloading_g_shifted = [val - 3 - degraw_unloading_g[0] for val in degraw_unloading_g]
tal_unloading_g_shifted = [val + 5 - tal_unloading_g[0] for val in tal_unloading_g]

czl635_loading_g_shifted = [val + (2304 - czl635_loading_g[0]) for val in czl635_loading_g]
degraw_loading_g_shifted = [val + (2296 - degraw_loading_g[0]) for val in degraw_loading_g]
tal_loading_g_shifted = [val + (2311 - tal_loading_g[0]) for val in tal_loading_g]


#print("Original CZL635 Unloading:", czl635_unloading_g)
print("Shifted CZL635 Unloading:", czl635_unloading_g_shifted)

#print("\nOriginal Degraw Unloading:", degraw_unloading_g)
print("Shifted Degraw Unloading:", degraw_unloading_g_shifted)

#print("\nOriginal TAL Unloading:", tal_unloading_g)
print("Shifted TAL Unloading:", tal_unloading_g_shifted)

#print("\nOriginal CZL635 Loading:", czl635_loading_g)
print("Shifted CZL635 Loading:", czl635_loading_g_shifted)

#print("\nOriginal Degraw Loading:", degraw_loading_g)
print("Shifted Degraw Loading:", degraw_loading_g_shifted)

#print("\nOriginal TAL Loading:", tal_loading_g)
print("Shifted TAL Loading:", tal_loading_g_shifted)


# Unloading
max_deviation_czl635_unloading = max([abs(x - 0) for x in czl635_unloading_g_shifted])
max_deviation_degraw_unloading = max([abs(x - 0) for x in degraw_unloading_g_shifted])
max_deviation_tal_unloading = max([abs(x - 0) for x in tal_unloading_g_shifted])

# Loading
max_deviation_czl635_loading = max([abs(x - 2300) for x in czl635_loading_g_shifted])
max_deviation_degraw_loading = max([abs(x - 2300) for x in degraw_loading_g_shifted])
max_deviation_tal_loading = max([abs(x - 2300) for x in tal_loading_g_shifted])

# Print results
print("Max Absolute Deviation (Unloading):")
print("Shifted CZL635:", max_deviation_czl635_unloading)
print("Shifted Degraw:", max_deviation_degraw_unloading)
print("Shifted TAL:", max_deviation_tal_unloading)

print("\nMax Absolute Deviation (Loading):")
print("Shifted CZL635:", max_deviation_czl635_loading)
print("Shifted Degraw:", max_deviation_degraw_loading)
print("Shifted TAL:", max_deviation_tal_loading)


# Plotting with the shifted values and adjusted settings
plt.figure(figsize=(10, 6))  # A4 width in inches with proportional height

# Plot for CZL635 Loading gf
plt.subplot(2, 3, 1)
plt.plot(czl635_loading_g_shifted, marker='o', color='blue', label='CZL635 Loading')
plt.title('Loading Averages for CZL635 (gf)', fontsize=9)
plt.xlabel('Event Number', fontsize=9)
plt.ylabel('Average Value (gf)', fontsize=9)
plt.ylim(2270, 2330)
plt.grid(True)
#plt.legend(fontsize=9)

# Plot for Degraw Loading gf
plt.subplot(2, 3, 2)
plt.plot(degraw_loading_g_shifted, marker='o', color='green', label='Degraw Loading')
plt.title('Loading Averages for Degraw (gf)', fontsize=9)
plt.xlabel('Event Number', fontsize=9)
plt.ylabel('Average Value (gf)', fontsize=9)
plt.ylim(2270, 2330)
plt.grid(True)
#plt.legend(fontsize=9)

# Plot for TAL220 Loading gf
plt.subplot(2, 3, 3)
plt.plot(tal_loading_g_shifted, marker='o', color='purple', label='TAL220 Loading')
plt.title('Loading Averages for TAL220 (gf)', fontsize=9)
plt.xlabel('Event Number', fontsize=9)
plt.ylabel('Average Value (gf)', fontsize=9)
plt.ylim(2270, 2330)
plt.grid(True)
#plt.legend(fontsize=9)

# Plot for CZL635 Unloading gf
plt.subplot(2, 3, 4)
plt.plot(czl635_unloading_g_shifted, marker='o', color='blue', linestyle='--', label='CZL635 Unloading')
plt.title('Unloading Averages for CZL635 (gf)', fontsize=9)
plt.xlabel('Event Number', fontsize=9)
plt.ylabel('Average Value (gf)', fontsize=9)
plt.ylim(-15, 15)
plt.grid(True)
#plt.legend(fontsize=9)

# Plot for Degraw Unloading gf
plt.subplot(2, 3, 5)
plt.plot(degraw_unloading_g_shifted, marker='o', color='green', linestyle='--', label='Degraw Unloading')
plt.title('Unloading Averages for Degraw (gf)', fontsize=9)
plt.xlabel('Event Number', fontsize=9)
plt.ylabel('Average Value (gf)', fontsize=9)
plt.ylim(-15, 15)
plt.grid(True)
#plt.legend(fontsize=9)

# Plot for TAL220 Unloading gf
plt.subplot(2, 3, 6)
plt.plot(tal_unloading_g_shifted, marker='o', color='purple', linestyle='--', label='TAL220 Unloading')
plt.title('Unloading Averages for TAL220 (gf)', fontsize=9)
plt.xlabel('Event Number', fontsize=9)
plt.ylabel('Average Value (gf)', fontsize=9)
plt.ylim(-15, 15)
plt.grid(True)
#plt.legend(fontsize=9)

plt.tight_layout()
plt.show()

# Repeatability analysis with name mapping
repeatability_results = {}

name_mapping = {
    'tal ()': 'TAL220 ()',
    'Degraw ()': 'Degraw ()',
    '635 ()': 'CZL635 ()'
}

for load_cell, conditions in averages.items():
    cell_mapped_name = name_mapping[load_cell]
    repeatability_results[cell_mapped_name] = {}
    
    for condition, avg_values in conditions.items():
        avg_values_g = [convert_to_grams(cell_mapped_name, val) for val in avg_values]
        initial_value = avg_values_g[0]
        differences_grams = [abs(value - initial_value) for value in avg_values_g]
        
        repeatability_results[cell_mapped_name][condition] = differences_grams

# Displaying the repeatability results in a readable format
readable_output = []

for cell_name, conditions in repeatability_results.items():
    readable_output.append(f"\nRepeatability for {cell_name}:\n")
    for condition, deviations in conditions.items():
        readable_output.append(f"\n{condition.capitalize()} events:")
        for i, deviation in enumerate(deviations):
            readable_output.append(f"Event {i+1}: Deviation from initial value = {deviation:.2f} gf")

output = "\n".join(readable_output)
print(output)











#1. **Descriptive Statistics**: 
#   - Compute mean, median, standard deviation, min, max, etc., for loading and unloading data. This will give a basic understanding of the data distribution.
  
#1. **Descriptive Statistics**: 
#   - Compute mean, median, standard deviation, min, max, etc., for loading and unloading data. This will give a basic understanding of the data distribution.

# Identify intervals for each load cell
loading_intervals_635, unloading_intervals_635 = simple_identify_intervals(load_test_data['635 ()'].values, load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'].values)
loading_intervals_degraw, unloading_intervals_degraw = simple_identify_intervals(load_test_data['Degraw ()'].values, load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'].values)
loading_intervals_tal, unloading_intervals_tal = simple_identify_intervals(load_test_data['tal ()'].values, load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'].values)

# Create a dictionary to map load cells to their respective intervals
loading_intervals_map = {
    '635 ()': loading_intervals_635,
    'Degraw ()': loading_intervals_degraw,
    'tal ()': loading_intervals_tal
}

unloading_intervals_map = {
    '635 ()': unloading_intervals_635,
    'Degraw ()': unloading_intervals_degraw,
    'tal ()': unloading_intervals_tal
}

def compute_descriptive_stats(load_test_data, load_cell, loading_intervals_map, unloading_intervals_map):
    """Compute descriptive statistics for a given load cell and set of intervals."""
    stats = {}
    
    for idx, (start, end) in enumerate(loading_intervals_map[load_cell]):
        data_segment = load_test_data[load_cell].iloc[start:end]
        stats[f"Loading Event {idx + 1}"] = {
            'Mean': data_segment.mean(),
            'Median': data_segment.median(),
            'Standard Deviation': data_segment.std(),
            'Minimum': data_segment.min(),
            'Maximum': data_segment.max()
        }
    
    for idx, (start, end) in enumerate(unloading_intervals_map[load_cell]):
        data_segment = load_test_data[load_cell].iloc[start:end]
        stats[f"Unloading Event {idx + 1}"] = {
            'Mean': data_segment.mean(),
            'Median': data_segment.median(),
            'Standard Deviation': data_segment.std(),
            'Minimum': data_segment.min(),
            'Maximum': data_segment.max()
        }

    return stats

print("Descriptive Statistics:\n")

for cell in load_cells:
    print(f"Load Cell: {cell}\n")
    cell_stats = compute_descriptive_stats(load_test_data, cell, loading_intervals_map, unloading_intervals_map)
    for event, values in cell_stats.items():
        print(event)
        for stat, value in values.items():
            print(f"  {stat}: {value:.2f}")
        print()
    print("-" * 40)

    
# Repeatability Analysis

def compute_repeatability(load_test_data, load_cell, intervals):
    """Compute repeatability for a given load cell and set of intervals."""
    deviations = []
    # Calculate the mean of the first event
    initial_mean = load_test_data[load_cell].iloc[intervals[0][0]:intervals[0][1]].mean()
    
    for idx, (start, end) in enumerate(intervals):
        current_mean = load_test_data[load_cell].iloc[start:end].mean()
        deviation = current_mean - initial_mean
        deviations.append(deviation)
    return deviations

print("Repeatability Analysis:\n")




fig, axes = plt.subplots(1, 3, figsize=(8.27, 3.5))

for ax, cell in zip(axes, load_cells):
    
    # Identify loading and unloading intervals for the current cell
    loading_intervals, unloading_intervals = simple_identify_intervals(load_test_data[cell], load_test_data['UNIX Timestamp (Milliseconds since 1970-01-01)'])
    loading_intervals, unloading_intervals = adjust_intervals(loading_intervals, unloading_intervals, len(load_test_data[cell]))

    loading_deviations = compute_repeatability(load_test_data, cell, loading_intervals)
    unloading_deviations = compute_repeatability(load_test_data, cell, unloading_intervals)
    
    # If there's an extra unloading event, omit it from the plot
    if len(unloading_deviations) > len(loading_deviations):
        unloading_deviations = unloading_deviations[:-1]
    
    # Bar plots
    ax.bar(range(len(loading_deviations)), loading_deviations, label='Loading Deviation', alpha=0.7)
    ax.bar(range(len(unloading_deviations)), unloading_deviations, label='Unloading Deviation', alpha=0.7, bottom=loading_deviations)
    
    # Set title and labels with specified font sizes
    ax.set_title(f"Repeatability Analysis for {cell}", fontsize=11)
    ax.set_ylabel('Digital Value Offset by Zero Adjustment', fontsize=10)
    ax.set_xlabel('Event Number', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
