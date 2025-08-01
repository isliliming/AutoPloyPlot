import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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

# Get Tg from Temperature (k) and Density (kg/m³) data
# fit the two liner regions of volume and density for the high-temperature liquid and the low-temperature solids
# try to find the intersection of the two lines to get Tg
def get_tg_density(new_data):
    """
    Calculate glass transition temperature (Tg) by fitting two linear regions
    to the density vs temperature data and finding their intersection.
    
    Args:
        new_data (pd.DataFrame): DataFrame with Temperature and Density columns
    
    Returns:
        float or None: Calculated Tg value in Kelvin, or None if calculation fails
    """

    
    # Check if required columns exist
    if 'Temperature (K)' not in new_data.columns or 'Density (kg/m³)' not in new_data.columns:
        print("Temperature or Density data not found for Tg calculation.")
        return None
    
    # Sort data by temperature (high to low for cooling curve)
    data_sorted = new_data.sort_values('Temperature (K)', ascending=False).copy()
    
    # Remove any NaN values
    data_clean = data_sorted.dropna(subset=['Temperature (K)', 'Density (kg/m³)'])
    
    if len(data_clean) < 10:
        print("Insufficient data points for Tg calculation.")
        return None
    
    # Split data into high and low temperature regions
    # Use middle 80% of data, avoiding first and last 10% to avoid edge effects
    n_points = len(data_clean) # return 150001 int
    
    # Calculate start and end indices for middle 80%
    start_idx = int(0.1 * n_points)  # Skip first 5%
    end_idx = int(1* n_points)    # Skip last 5%
    middle_data = data_clean.iloc[start_idx:end_idx]
    
    # Split middle data into high and low temperature regions
    middle_points = len(middle_data)
    high_temp_n = max(5, int(0.25 * middle_points))  # Use 40% of middle data for high temp
    low_temp_n = max(5, int(0.25 * middle_points))   # Use 40% of middle data for low temp
    
    high_temp_data = middle_data.head(high_temp_n) # return dataframe, high_temp_data.shape = (48000, 5)
    low_temp_data = middle_data.tail(low_temp_n) # return dataframe, low_temp_data.shape = (48000, 5)
    
    # Fit linear regression to high temperature region (liquid phase)
    try:
        slope_high, intercept_high, r_value_high, p_value_high, std_err_high = stats.linregress(
            high_temp_data['Temperature (K)'], high_temp_data['Density (kg/m³)']
        )
        
        # Fit linear regression to low temperature region (solid/glassy phase)
        slope_low, intercept_low, r_value_low, p_value_low, std_err_low = stats.linregress(
            low_temp_data['Temperature (K)'], low_temp_data['Density (kg/m³)']
        )
        
        print(f"\nHigh temperature region (liquid):")
        print(f"  Slope: {slope_high:.6f} kg/(m³·K)")
        print(f"  R²: {r_value_high**2:.4f}")
        print(f"  Temperature range: {high_temp_data['Temperature (K)'].min():.1f}K ({high_temp_data['Temperature (K)'].min()-273.15:.1f}°C) - {high_temp_data['Temperature (K)'].max():.1f}K ({high_temp_data['Temperature (K)'].max()-273.15:.1f}°C)")
        
        print(f"\nLow temperature region (solid):")
        print(f"  Slope: {slope_low:.6f} kg/(m³·K)")
        print(f"  R²: {r_value_low**2:.4f}")
        print(f"  Temperature range: {low_temp_data['Temperature (K)'].min():.1f}K ({low_temp_data['Temperature (K)'].min()-273.15:.1f}°C) - {low_temp_data['Temperature (K)'].max():.1f}K ({low_temp_data['Temperature (K)'].max()-273.15:.1f}°C)")
        
    except Exception as e:
        print(f"Error in linear regression: {e}")
        return None
    
    # Calculate intersection point (Tg)
    # At intersection: slope_high * Tg + intercept_high = slope_low * Tg + intercept_low
    # Therefore: Tg = (intercept_low - intercept_high) / (slope_high - slope_low)
    if abs(slope_high - slope_low) > 1e-10:  # Avoid division by zero
        tg = (intercept_low - intercept_high) / (slope_high - slope_low)
        
        # Check if Tg is within reasonable range of the data
        temp_min = data_clean['Temperature (K)'].min()
        temp_max = data_clean['Temperature (K)'].max()
        
        if temp_min <= tg <= temp_max:
            print(f"\nCalculated Tg: {tg:.2f} K ({tg-273.15:.2f}°C)")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot all data points
            plt.scatter(data_clean['Temperature (K)'], data_clean['Density (kg/m³)'], 
                       alpha=0.6, s=20, color='lightblue', label='All data points')
            
            # Plot high temperature region and fit
            plt.scatter(high_temp_data['Temperature (K)'], high_temp_data['Density (kg/m³)'], 
                       color='red', s=40, alpha=0.8, label=f'High T region (R²={r_value_high**2:.3f})')
            
            # Extend the high temperature fit line
            temp_extended_high = np.linspace(temp_min, temp_max, 100)
            density_high = slope_high * temp_extended_high + intercept_high
            plt.plot(temp_extended_high, density_high, 'r--', linewidth=2, 
                    label=f'Liquid fit (slope: {slope_high:.4f})')
            
            # Plot low temperature region and fit
            plt.scatter(low_temp_data['Temperature (K)'], low_temp_data['Density (kg/m³)'], 
                       color='green', s=40, alpha=0.8, label=f'Low T region (R²={r_value_low**2:.3f})')
            
            # Extend the low temperature fit line
            temp_extended_low = np.linspace(temp_min, temp_max, 100)
            density_low = slope_low * temp_extended_low + intercept_low
            plt.plot(temp_extended_low, density_low, 'g--', linewidth=2, 
                    label=f'Solid fit (slope: {slope_low:.4f})')
            
            # Mark Tg point
            density_at_tg = slope_high * tg + intercept_high
            plt.scatter(tg, density_at_tg, color='black', s=150, marker='*', 
                       label=f'Tg = {tg:.1f} K ({tg-273.15:.1f}°C)', edgecolors='white', linewidth=2)
            
            # Add vertical line at Tg
            plt.axvline(x=tg, color='black', linestyle=':', alpha=0.7)
            
            plt.xlabel('Temperature (K)', fontsize=14)
            plt.ylabel('Density (kg/m³)', fontsize=14)
            plt.title(short_name+'_Temperature vs Density_Tg', fontsize=16)
            plt.gca().invert_xaxis()  # High to low temperature
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(short_name+'_Temperature vs Density_Tg.png', dpi=300)
            plt.show()
            
            return tg
        else:
            print(f"Calculated Tg ({tg:.2f} K ({tg-273.15:.2f}°C)) is outside the data range ({temp_min:.1f}K ({temp_min-273.15:.1f}°C) - {temp_max:.1f}K ({temp_max-273.15:.1f}°C))")
            return None
    else:
        print("Slopes are too similar - cannot calculate meaningful Tg")
        print(f"High temp slope: {slope_high:.6f}, Low temp slope: {slope_low:.6f}")
        return None

def get_tg_volume(new_data):
    """
    Calculate glass transition temperature (Tg) by fitting two linear regions
    to the volume vs temperature data and finding their intersection.
    
    Args:
        new_data (pd.DataFrame): DataFrame with Temperature and Volume columns
    
    Returns:
        float or None: Calculated Tg value in Kelvin, or None if calculation fails
    """
    
    # Check if required columns exist
    if 'Temperature (K)' not in new_data.columns or 'Volume (nm³)' not in new_data.columns:
        print("Temperature or Volume data not found for Tg calculation.")
        return None
    
    # Sort data by temperature (high to low for cooling curve)
    data_sorted = new_data.sort_values('Temperature (K)', ascending=False).copy()
    
    # Remove any NaN values
    data_clean = data_sorted.dropna(subset=['Temperature (K)', 'Volume (nm³)'])
    
    if len(data_clean) < 10:
        print("Insufficient data points for Tg calculation.")
        return None
    
    # Split data into high and low temperature regions
    # Use middle 80% of data, avoiding first and last 10% to avoid edge effects
    n_points = len(data_clean) # return 150001 int
    
    # Calculate start and end indices for middle 80%
    start_idx = int(0.1 * n_points)  # Skip first 10%
    end_idx = int(1 * n_points)    # Skip last 10%
    middle_data = data_clean.iloc[start_idx:end_idx]
    
    # Split middle data into high and low temperature regions
    middle_points = len(middle_data)
    high_temp_n = max(5, int(0.25 * middle_points))  # Use 25% of middle data for high temp
    low_temp_n = max(5, int(0.25 * middle_points))   # Use 25% of middle data for low temp
    
    high_temp_data = middle_data.head(high_temp_n) # return dataframe
    low_temp_data = middle_data.tail(low_temp_n) # return dataframe
    
    # Fit linear regression to high temperature region (liquid phase)
    try:
        slope_high, intercept_high, r_value_high, p_value_high, std_err_high = stats.linregress(
            high_temp_data['Temperature (K)'], high_temp_data['Volume (nm³)']
        )
        
        # Fit linear regression to low temperature region (solid/glassy phase)
        slope_low, intercept_low, r_value_low, p_value_low, std_err_low = stats.linregress(
            low_temp_data['Temperature (K)'], low_temp_data['Volume (nm³)']
        )
        
        print(f"\nHigh temperature region (liquid):")
        print(f"  Slope: {slope_high:.6f} nm³/K")
        print(f"  R²: {r_value_high**2:.4f}")
        print(f"  Temperature range: {high_temp_data['Temperature (K)'].min():.1f}K ({high_temp_data['Temperature (K)'].min()-273.15:.1f}°C) - {high_temp_data['Temperature (K)'].max():.1f}K ({high_temp_data['Temperature (K)'].max()-273.15:.1f}°C)")
        
        print(f"\nLow temperature region (solid):")
        print(f"  Slope: {slope_low:.6f} nm³/K")
        print(f"  R²: {r_value_low**2:.4f}")
        print(f"  Temperature range: {low_temp_data['Temperature (K)'].min():.1f}K ({low_temp_data['Temperature (K)'].min()-273.15:.1f}°C) - {low_temp_data['Temperature (K)'].max():.1f}K ({low_temp_data['Temperature (K)'].max()-273.15:.1f}°C)")
        
    except Exception as e:
        print(f"Error in linear regression: {e}")
        return None
    
    # Calculate intersection point (Tg)
    # At intersection: slope_high * Tg + intercept_high = slope_low * Tg + intercept_low
    # Therefore: Tg = (intercept_low - intercept_high) / (slope_high - slope_low)
    if abs(slope_high - slope_low) > 1e-10:  # Avoid division by zero
        tg = (intercept_low - intercept_high) / (slope_high - slope_low)
        
        # Check if Tg is within reasonable range of the data
        temp_min = data_clean['Temperature (K)'].min()
        temp_max = data_clean['Temperature (K)'].max()
        
        if temp_min <= tg <= temp_max:
            print(f"\nCalculated Tg: {tg:.2f} K ({tg-273.15:.2f}°C)")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot all data points
            plt.scatter(data_clean['Temperature (K)'], data_clean['Volume (nm³)'], 
                       alpha=0.6, s=20, color='lightblue', label='All data points')
            
            # Plot high temperature region and fit
            plt.scatter(high_temp_data['Temperature (K)'], high_temp_data['Volume (nm³)'], 
                       color='red', s=40, alpha=0.8, label=f'High T region (R²={r_value_high**2:.3f})')
            
            # Extend the high temperature fit line
            temp_extended_high = np.linspace(temp_min, temp_max, 100)
            volume_high = slope_high * temp_extended_high + intercept_high
            plt.plot(temp_extended_high, volume_high, 'r--', linewidth=2, 
                    label=f'Liquid fit (slope: {slope_high:.4f})')
            
            # Plot low temperature region and fit
            plt.scatter(low_temp_data['Temperature (K)'], low_temp_data['Volume (nm³)'], 
                       color='green', s=40, alpha=0.8, label=f'Low T region (R²={r_value_low**2:.3f})')
            
            # Extend the low temperature fit line
            temp_extended_low = np.linspace(temp_min, temp_max, 100)
            volume_low = slope_low * temp_extended_low + intercept_low
            plt.plot(temp_extended_low, volume_low, 'g--', linewidth=2, 
                    label=f'Solid fit (slope: {slope_low:.4f})')
            
            # Mark Tg point
            volume_at_tg = slope_high * tg + intercept_high
            plt.scatter(tg, volume_at_tg, color='black', s=150, marker='*', 
                       label=f'Tg = {tg:.1f} K ({tg-273.15:.1f}°C)', edgecolors='white', linewidth=2)
            
            # Add vertical line at Tg
            plt.axvline(x=tg, color='black', linestyle=':', alpha=0.7)
            
            plt.xlabel('Temperature (K)', fontsize=14)
            plt.ylabel('Volume (nm³)', fontsize=14)
            plt.title(short_name+'_Temperature vs Volume_Tg', fontsize=16)
            plt.gca().invert_xaxis()  # High to low temperature
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(short_name+'_Temperature vs Volume_Tg.png', dpi=300)
            plt.show()
            
            return tg
        else:
            print(f"Calculated Tg ({tg:.2f} K ({tg-273.15:.2f}°C)) is outside the data range ({temp_min:.1f}K ({temp_min-273.15:.1f}°C) - {temp_max:.1f}K ({temp_max-273.15:.1f}°C))")
            return None
    else:
        print("Slopes are too similar - cannot calculate meaningful Tg")
        print(f"High temp slope: {slope_high:.6f}, Low temp slope: {slope_low:.6f}")
        return None




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
        # plt.savefig(short_name+'temperature_vs_density.png', dpi=300)
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
        # plt.savefig(short_name+'temperature_vs_volume.png', dpi=300)
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
        # plt.savefig(short_name+'temperature_vs_heat_capacity.png', dpi=300)
        plt.show()
        plt.close()
    else :
        print("Temperature or Heat Capacity data not found for plotting.")

# Example usage

filepath = 'GROMACS/NPT/Pure/cur300bif0/density_temp01.xvg'
short_name = 'cur300bif0'+'_'
# yaxis  label "(K), (nm^3), (kg/m^3), (kJ/mol)"

time, data, legends, units = parse_xvg_data(filepath)
new_data = process_data(data, legends, units)
plot_data(new_data,short_name)

# Calculate and display Tg from density
tg_value_density = get_tg_density(new_data)
if tg_value_density is not None:
    print(f"\n=== Final Result from Density ===")
    print(f"Glass transition temperature (Tg): {tg_value_density:.2f} K ({tg_value_density-273.15:.2f}°C)")
else:
    print("Tg calculation from density was unsuccessful.")

# Calculate and display Tg from volume
tg_value_volume = get_tg_volume(new_data)
if tg_value_volume is not None:
    print(f"\n=== Final Result from Volume ===")
    print(f"Glass transition temperature (Tg): {tg_value_volume:.2f} K ({tg_value_volume-273.15:.2f}°C)")
else:
    print("Tg calculation from volume was unsuccessful.")

# Compare results if both calculations were successful
if tg_value_density is not None and tg_value_volume is not None:
    print(f"\n=== Comparison ===")
    print(f"Tg from density: {tg_value_density:.2f} K ({tg_value_density-273.15:.2f}°C)")
    print(f"Tg from volume:  {tg_value_volume:.2f} K ({tg_value_volume-273.15:.2f}°C)")
    difference = abs(tg_value_density - tg_value_volume)
    print(f"Difference: {difference:.2f} K ({difference:.2f}°C)")