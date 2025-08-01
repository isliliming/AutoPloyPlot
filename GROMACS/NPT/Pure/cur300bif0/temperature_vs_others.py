import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_xvg_data(filepath: str) -> tuple:
    """Parses the XVG-like data file, extracting time and data columns.

    Args:
        filepath (str): The path to the XVG-like file.

    Returns:
        tuple: A tuple containing:
            - times (numpy.ndarray): Array of time values.
            - data (numpy.ndarray): 2D array of data columns.  data[:, i] is
              the i-th data column.
            - legends (list): List of legend labels for each data column.
            - units (list): List of units for each data column.
    """
    times = []
    data_list = []  # Store data as lists initially for appending
    legends = []
    units = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('@'):
                if line.startswith('@') and 'legend' in line and '"' in line:
                    legend_text = line.split('"')[1]
                    legends.append(legend_text)
                if line.startswith('@') and 'yaxis' in line and 'label' in line:
                    # Extract and parse units from the yaxis label
                    units_str = line.split('"')[1]
                    # Split by commas and extract individual units
                    units = [u.strip().replace('(', '').replace(')', '').replace('^3', '³')  for u in units_str.split(',')]

                continue  # Skip header lines

            parts = line.split()
            times.append(float(parts[0]))
            data_list.append([float(x) for x in parts[1:]])

    # Convert lists to NumPy arrays for easier plotting
    times = np.array(times)
    data = np.array(data_list)

    #Check if number of columns is equal to the number of extracted legends
    if data.shape[1] != len(legends):
      raise ValueError(
            f"Number of data columns ({data.shape[1]}) does not match "
            f"the number of legends ({len(legends)})."
        )

    if len(units) != len(legends) and len(units) !=0 :
        raise ValueError("Number of units does not match the number of legends.")

    if not units: #if empty
        units = [""] * len(legends)  # Fill with empty strings if no units are found

    return times, data, legends, units



def process_data(data,legends, units):
    """Process data by merging legends and units, then creating a DataFrame.

    Args:
        data (numpy.ndarray): 2D array of data columns
        legends (list): List of legend labels for each data column
        units (list): List of units for each data column

    Returns:
        pandas.DataFrame: DataFrame with data and column names combining legends and units
    """

    # Merge legends and units to create column names
    column_names = []
    for i, legend in enumerate(legends):
        if i < len(units) and units[i]:
            column_names.append(f"{legend} ({units[i]})")
        else:
            column_names.append(legend)

    # Create DataFrame with the data and column names
    new_data = pd.DataFrame(data, columns=column_names)
    # calculate the heat capacity from enthalpy and temperature
    if 'Enthalpy (kJ/mol)' in column_names and 'Temperature (K)' in column_names:
        enthalpy_col = new_data['Enthalpy (kJ/mol)']
        temperature_col = new_data['Temperature (K)']
        if not temperature_col.empty and not enthalpy_col.empty:
            heat_capacity = enthalpy_col.diff() / temperature_col.diff()
            new_data['Heat Capacity (kJ/(mol K))'] = heat_capacity
    else:
        print("Enthalpy or Temperature data not found for heat capacity calculation.")

    return new_data

def plot_data(new_data, short_name):
    # Sort data by temperature in descending order (high to low)
    if 'Temperature (K)' in new_data.columns:
        new_data = new_data.sort_values('Temperature (K)', ascending=False)

    # Temperature vs Density
    if 'Temperature (K)' in new_data.columns and 'Density (kg/m³)' in new_data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(new_data['Temperature (K)'], new_data['Density (kg/m³)'], linestyle='-', color='b')
        plt.title(short_name+'Temperature vs Density')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Density (kg/m³)')
        plt.gca().invert_xaxis()  # Reverse x-axis to show high to low temperature
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(short_name+'temperature_vs_density.png', dpi=300)
        plt.show()
        plt.close()
    else :
        print("Temperature or Density data not found for plotting.")

    # Temperature vs volume
    if 'Temperature (K)' in new_data.columns and 'Volume (nm³)' in new_data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(new_data['Temperature (K)'], new_data['Volume (nm³)'], linestyle='-', color='g')
        plt.title(short_name+'Temperature vs Volume')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Volume (nm³)')
        plt.gca().invert_xaxis()  # Reverse x-axis to show high to low temperature
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(short_name+'temperature_vs_volume.png', dpi=300)
        plt.show()
        plt.close()
    else :
        print("Temperature or Volume data not found for plotting.")

    # Temperature vs Heat Capacity
    if 'Temperature (K)' in new_data.columns and 'Heat Capacity (kJ/(mol K))' in new_data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(new_data['Temperature (K)'], new_data['Heat Capacity (kJ/(mol K))'], linestyle='-', color='r')
        plt.title(short_name+'Temperature vs Heat Capacity')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Heat Capacity (kJ/(mol K))')
        plt.gca().invert_xaxis()  # Reverse x-axis to show high to low temperature
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(short_name+'temperature_vs_heat_capacity.png', dpi=300)
        plt.show()
        plt.close()
    else :
        print("Temperature or Heat Capacity data not found for plotting.")

# Example usage

filepath = 'density_temp01.xvg'
short_name = 'cur300bif0'+'_'
# yaxis  label "(K), (nm^3), (kg/m^3), (kJ/mol)"

time, data, legends, units = parse_xvg_data(filepath)
new_data = process_data(data, legends, units)
plot_data(new_data,short_name)