import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_uv_data(file_path):
    """
    Read UV data from a file and return as a pandas DataFrame.
    """
    # Read the data - each row is of format: wavelength,value1,wavelength,value2,wavelength,value3,...
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        data = [line.strip().split(',') for line in lines if line.strip()]

        # Convert data to numeric values, ignoring empty strings
        clean_data = []
        for row in data:
            clean_row = [float(val) if val.strip() and val.strip().replace('.', '').replace('-', '').replace('E', '').isdigit() else np.nan for val in row]
            clean_data.append(clean_row)

        # Create a DataFrame from the parsed data
        # Assuming the first column of each pair is wavelength, and the second is the measurement
        wavelengths = []
        values = []
        for row in clean_data:
            # Skip rows with insufficient data
            if len(row) < 2:
                continue

            for i in range(0, len(row), 2):
                if i+1 < len(row):
                    wavelengths.append(row[i])
                    values.append(row[i+1])

        # Reshape data for DataFrame
        num_columns = len(clean_data[0]) // 2
        indices = np.arange(len(wavelengths) // num_columns)

        df = pd.DataFrame()
        for i in range(num_columns):
            col_indices = np.arange(i, len(wavelengths), num_columns)
            df[f'Wavelength_{i+1}'] = [wavelengths[j] for j in col_indices]
            df[f'Value_{i+1}'] = [values[j] for j in col_indices]

        return df

    except Exception as e:
        print(f"Error reading UV data: {e}")
        return None

def plot_uv_data(df, title="UV Data", save_path=None):
    """
    Plot UV data from DataFrame.
    """
    plt.figure(figsize=(12, 8))

    # Get the number of data series
    num_series = len(df.columns) // 2

    # Create a plot for each data series
    for i in range(num_series):
        wavelength_col = f'Wavelength_{i+1}'
        value_col = f'Value_{i+1}'

        plt.plot(df[wavelength_col], df[value_col], label=f'Series {i+1}')

    # Add labels and legend
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Tight layout to optimize the plot space
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

if __name__ == "__main__":
    # Path to UV data file
    data_file = "uv_data.csv"  # Update this path to your data file

    # Read the data
    df = read_uv_data(data_file)

    if df is not None:
        # Plot the data
        plot_uv_data(df, title="UV Spectrum Analysis")