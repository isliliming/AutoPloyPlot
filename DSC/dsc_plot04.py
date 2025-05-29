# add the function of finding the Tg
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

cooling_color = '#332288'
heating_color = '#882255'
isothermal_color = '#BBBBBB'

def read_dsc_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find all [step] sections and extract data from each
    step_indices = [i for i, line in enumerate(lines) if line.strip() == '[step]']
    all_data = []
    step_ranges = []
    current_idx = 0

    for step_num, step_idx in enumerate(step_indices):
        # Find the header for this [step]
        header_found = False
        for j in range(step_idx + 1, len(lines)):
            if lines[j].startswith('Time\tTemperature\tHeat Flow (Normalized)'):
                header_line = j
                header_found = True
                break
        if not header_found:
            continue

        # Extract data lines until next section or EOF(End of File)
        data_lines = []
        for line in lines[header_line + 2:]:  # Skip header and units line
            if line.strip().startswith('[') or not line.strip():
                break  # Stop at next section or empty line
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                data_lines.append(parts[:3])  # Keep Time, Temp, Heat Flow

        # Convert to DataFrame
        df_segment = pd.DataFrame(data_lines, columns=['Time (min)', 'Temperature (°C)', 'Heat Flow (W/g)'])
        df_segment = df_segment.apply(pd.to_numeric, errors='coerce') # Convert to numeric and ignore errors
        n_rows = len(df_segment)
        if n_rows > 0:
            all_data.append(df_segment)
            step_ranges.append({'step': step_num+1, 'start_idx': current_idx, 'end_idx': current_idx + n_rows - 1})
            current_idx += n_rows

    # Combine all segments into a single DataFrame
    if not all_data:
        raise ValueError("No valid data found in the file.")
    full_df = pd.concat(all_data, ignore_index=True)
    steps_df = pd.DataFrame(step_ranges)
    return full_df, steps_df

def detect_step_temperature_changes(full_df, steps_df):
    # Add a column to store the step type
    step_types = []
    for _, row in steps_df.iterrows():
        start = row['start_idx']
        end = row['end_idx']
        step_df = full_df.iloc[start:end+1]
        temps = step_df['Temperature (°C)'].dropna()
        if len(temps) < 2:
            step_types.append('unknown')
            continue
        n = max(1, int(0.1 * len(temps)))
        top10 = temps.iloc[:n].mean()
        bottom10 = temps.iloc[-n:].mean()
        diff = bottom10 - top10
        if abs(diff) < 10:
            step_types.append('isothermal')
        elif diff > 0:
            step_types.append('heating')
        else:
            step_types.append('cooling')
    steps_df = steps_df.copy()
    steps_df['step_type'] = step_types
    return steps_df

# find the Tg
def find_Tg(steps_df, full_df, folder_path):
    3
        


def color_for_steps(steps_df):
    # Create a copy to avoid modifying the original DataFrame
    steps_df = steps_df.copy()
    
    # Initialize color and alpha columns
    steps_df['color'] = ''
    steps_df['alpha'] = 1.0
    
    # Count occurrences of each step type
    type_counts = steps_df['step_type'].value_counts()
    
    # Track the current count for each type
    type_counters = {'cooling': 0, 'heating': 0, 'isothermal': 0}
    
    for idx, row in steps_df.iterrows():
        step_type = row['step_type']
        
        # Assign color based on step type
        if step_type == 'cooling':
            color = cooling_color
        elif step_type == 'heating':
            color = heating_color
        else:  # isothermal
            color = isothermal_color
            
        # Get total count for this type
        total_count = type_counts[step_type]
        current_count = type_counters[step_type]
        
        # Calculate alpha value (transparency)
        # For single step of a type, use full opacity
        if total_count == 1:
            alpha = 1.0
        else:
            # Distribute alpha values between 0.5 and 1.0
            alpha = 0.5 + (0.5 * current_count / (total_count - 1))
        
        # Update the DataFrame
        steps_df.at[idx, 'color'] = color
        steps_df.at[idx, 'alpha'] = alpha
        
        # Increment the counter for this type
        type_counters[step_type] += 1
    
    return steps_df

def plot_dsc_curve(df,folder_path, file_name, steps_df=None): # None means not necessary to plot the steps
    plt.figure(figsize=(12, 6))
    if steps_df is not None:
        for _, row in steps_df.iterrows():
            start = row['start_idx']
            end = row['end_idx']
            color = row['color']
            alpha = row['alpha']
            plt.plot(
                df['Temperature (°C)'].iloc[start:end+1],
                df['Heat Flow (W/g)'].iloc[start:end+1],
                color=color,
                alpha=alpha,
                linewidth=1.5
            )
    else:
        plt.plot(df['Temperature (°C)'], df['Heat Flow (W/g)'], color='#332288', linewidth=1.5)
    plt.xlabel('Temperature / °C', fontsize=20)
    plt.ylabel('Heat Flow / W$\cdot$g$^{-1}$', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title('Full DSC Curve with Multiple Segments', fontsize=14)
    # set x and y axis limits
    plt.xlim(df['Temperature (°C)'].min()-25, df['Temperature (°C)'].max()+25)
    plt.ylim(df['Heat Flow (W/g)'].min()-0.1, df['Heat Flow (W/g)'].max()+0.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    # save the plot
    plt.savefig(os.path.join(folder_path, file_name[:-3]+'png'), dpi=300)
    print(f"Plot saved to {os.path.join(folder_path, file_name[:-3]+'png')}")
    plt.show()


# Usage
# change the filename to the file you want to load
# filename = '/Users/liming/Desktop/AutoPloyPlot/DSC/data_dsc/LLM_EP09/LLM_EP0901.txt'
# df, steps_df = read_dsc_data(filename)
# steps_df = detect_step_temperature_changes(df, steps_df)
# steps_df = color_for_steps(steps_df)
# plot_dsc_curve(df, steps_df)


def process_folder(folder_path):
    """Process all .txt files in the specified folder"""
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    # Get all .txt files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        print(f"No .txt files found in {folder_path}")
        return
    
    print(f"Found {len(files)} .txt files to process")
    
    # Process each file
    for file_name in files:
        full_path = os.path.join(folder_path, file_name)
        print(f"\nProcessing file: {file_name}")
        try:
            df, steps_df = read_dsc_data(full_path)
            steps_df = detect_step_temperature_changes(df, steps_df)
            steps_df = color_for_steps(steps_df)
            find_Tg(steps_df, df, folder_path)
            plot_dsc_curve(df, folder_path, file_name, steps_df)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

# Usage example:
# Specify the folder path containing your DSC data files
folder_path = '/Users/liming/Desktop/AutoPloyPlot/DSC/data_dsc/dsc_test'
process_folder(folder_path)

