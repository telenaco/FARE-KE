# DAQ Comparison Tools

This repository contains files for comparing data acquisition systems and documentation references.

## Repository Contents

- `updated-editing-guidelines_v2.md`: Guidelines for editing thesis text, including padding word removal and UK English standardization
- `thesis-reference-guide.md`: Reference document outlining the structure and content of thesis chapters
- `hx711VsMcp.py`: Python script that compares performance between HX711 and MCP3564 data acquisition systems
- `ellapsedData.csv`: Raw data file containing timestamped readings from both DAQ systems

## Using the HX711 vs MCP Comparison Script

### Requirements
- Python 3.x
- pandas
- matplotlib

### Running the Script
```
python hx711VsMcp.py
```

### What the Script Does
The script analyzes sampling rates and timing consistency of both DAQ systems by:
1. Loading and parsing data from `ellapsedData.csv`
2. Calculating average time between samples for each system
3. Computing standard deviation of sampling intervals
4. Generating histograms of sampling time distributions
5. Calculating and displaying samples per second for each system

### CSV Data Format
The `ellapsedData.csv` file has 5 columns:
1. MCP_Reading: Raw sensor reading from MCP3564
2. HX711_Reading: Raw sensor reading from HX711
3. Timestamp: Time in microseconds
4. MCP_Flag: Indicates an MCP data point (1 = yes, 0 = no)
5. HX711_Flag: Indicates an HX711 data point (1 = yes, 0 = no)