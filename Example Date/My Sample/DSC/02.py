import pandas as pd
import matplotlib.pyplot as plt

def read_dsc_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find all [step] sections and extract data from each
    step_indices = [i for i, line in enumerate(lines) if line.strip() == '[step]']
    all_data = []

    for step_idx in step_indices:
        # Find the header for this [step]
        header_found = False
        for j in range(step_idx + 1, len(lines)):
            if lines[j].startswith('Time\tTemperature\tHeat Flow (Normalized)'):
                header_line = j
                header_found = True
                break
        if not header_found:
            continue

        # Extract data lines until next section or EOF
        data_lines = []
        for line in lines[header_line + 2:]:  # Skip header and units line
            if line.strip().startswith('[') or not line.strip():
                break  # Stop at next section or empty line
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                data_lines.append(parts[:3])  # Keep Time, Temp, Heat Flow

        # Convert to DataFrame
        df_segment = pd.DataFrame(data_lines, columns=['Time (min)', 'Temperature (°C)', 'Heat Flow (W/g)'])
        df_segment = df_segment.apply(pd.to_numeric, errors='coerce')
        all_data.append(df_segment)

    # Combine all segments into a single DataFrame
    if not all_data:
        raise ValueError("No valid data found in the file.")
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

def plot_dsc_curve(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Temperature (°C)'], df['Heat Flow (W/g)'], 'b-', linewidth=1)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Heat Flow (W/g)', fontsize=12)
    plt.title('Full DSC Curve with Multiple Segments', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Example usage
filename = 'chabif01.txt'
df = read_dsc_data(filename)
print(f"Total data points loaded: {len(df)}")
plot_dsc_curve(df)