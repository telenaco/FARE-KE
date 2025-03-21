import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('ellapsedData.csv', header=None)
data.columns = ["MCP_Reading", "HX711_Reading", "Timestamp", "MCP_Flag", "HX711_Flag"]

# Filter out the rows where MCP_Flag is 1
mcp_data = data[data["MCP_Flag"] == 1]
# Calculate the time differences between consecutive samples for MCP
mcp_time_diffs = mcp_data["Timestamp"].diff().dropna()

# Filter out the rows where HX711_Flag is 1
hx711_data = data[data["HX711_Flag"] == 1]
# Calculate the time differences between consecutive samples for HX711
hx711_time_diffs = hx711_data["Timestamp"].diff().dropna()

# Calculate mean and standard deviation for both MCP and HX711
mcp_mean_diff = mcp_time_diffs.mean()
mcp_std_diff = mcp_time_diffs.std()

hx711_mean_diff = hx711_time_diffs.mean()
hx711_std_diff = hx711_time_diffs.std()

# Print the calculated mean and standard deviation values
print("MCP Average Time Difference (microseconds):", mcp_mean_diff)
print("MCP Standard Deviation (microseconds):", mcp_std_diff)
print("\nHX711 Average Time Difference (microseconds):", hx711_mean_diff)
print("HX711 Standard Deviation (microseconds):", hx711_std_diff)

# Histogram plotting for MCP
plt.figure(figsize=(7, 6))
plt.hist(mcp_time_diffs, bins=50, color='blue', alpha=0.7)
plt.axvline(mcp_mean_diff, color='r', linestyle='dashed', linewidth=2)
plt.title('Time Differences Between Consecutive MCP Readings')
plt.xlabel('Time Difference (microseconds)')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Histogram plotting for HX711
plt.figure(figsize=(7, 6))
plt.hist(hx711_time_diffs, bins=50, color='green', alpha=0.7)
plt.axvline(hx711_mean_diff, color='r', linestyle='dashed', linewidth=2)
plt.title('Time Differences Between Consecutive HX711 Readings')
plt.xlabel('Time Difference (microseconds)')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Compute the samples per second for both MCP and HX711
sps_mcp = 1e6 / mcp_mean_diff
sps_hx711 = 1e6 / hx711_mean_diff

print("\nMCP Samples Per Second:", sps_mcp)
print("HX711 Samples Per Second:", sps_hx711)
