import pandas as pd
import matplotlib.pyplot as plt

def read_dsc_data(filename):
    # Read the file lines
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the data under [step]
    start_index = None
    for i, line in enumerate(lines):
        if line.strip() == '[step]':
            # Search for the header line
            for j in range(i+1, len(lines)):
                if lines[j].startswith('Time\tTemperature\tHeat Flow (Normalized)'):
                    # Data starts two lines after the header (skip units line)
                    start_index = j + 2
                    break
            if start_index:
                break
    
    if start_index is None:
        raise ValueError("Data section not found in the file.")
    
    # Extract data lines
    data_lines = []
    for line in lines[start_index:]:
        parts = line.strip().split('\t')
        # Pad parts to ensure 3 columns
        parts += [''] * (3 - len(parts))
        # Check if the first column is a valid number
        try:
            float(parts[0])
            data_lines.append('\t'.join(parts[:3]))
        except:
            # Stop if a new section or invalid line is encountered
            if line.strip().startswith('[') or not line.strip():
                break
    
    # Create DataFrame from extracted data
    from io import StringIO
    data_str = StringIO('\n'.join(data_lines))
    df = pd.read_csv(
        data_str, 
        sep='\t', 
        names=['Time (min)', 'Temperature (°C)', 'Heat Flow (W/g)'], 
        engine='python'
    )
    # Convert Heat Flow to numeric, handle missing values
    df['Heat Flow (W/g)'] = pd.to_numeric(df['Heat Flow (W/g)'], errors='coerce')
    return df

def plot_dsc_curve(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Temperature (°C)'], df['Heat Flow (W/g)'], 'b-', linewidth=1)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Heat Flow (W/g)')
    plt.title('DSC Curve')
    plt.grid(True)
    plt.show()

# Example usage
filename = 'chabif01.txt'
df = read_dsc_data(filename)
plot_dsc_curve(df)
