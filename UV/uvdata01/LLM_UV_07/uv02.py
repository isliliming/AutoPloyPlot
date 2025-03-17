import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_uv(file_path):
    # Read data and find header
    # find the data
    header_line = 0
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line.startswith(",Wavelength (nm),Abs"):
                header_line = line_num
                break

    # find the sample namess
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Name,Sample'):
                sample_names = line.split(',')[1:]
                sample_names = [name for name in sample_names if name.strip()]
                # print(sample_names)
                sample_num = len(sample_names)
                break



    # Read data and reverse orders
    df = pd.read_csv(file_path, skiprows=header_line)
    df = df.sort_values('Wavelength (nm)', ascending=True)  # Sort from low to high wavelength
    # rename Abs columns with sample names
    for name, index in zip(sample_names, list(range(1, sample_num+1))):
        column_title = df.columns.tolist()
        column_title[index*2] = name
        df.columns = column_title


    # Extract data
    wavelengths = df['Wavelength (nm)']
    # creat a new dataframe to store the abs values
    df_abs = pd.DataFrame()
    for name in sample_names:
        df_abs[name] = df[name]
    # merge the wavelength and abs values
    df_abs.insert(0, 'Wavelength (nm)', wavelengths)

    # Plot
    plt.figure()
    for name in sample_names:
        plt.plot(df_abs['Wavelength (nm)'], df_abs[name], label=name)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Abs')
    plt.legend()
    plt.savefig(file_path[:-3]+'png', dpi=300)
    plt.show()

# read all the .csv files in the folder and plot
files = os.listdir()
for file in files:
    if file.endswith('.csv'):
        file_path = file
        print(f"Processing file: {file_path}")
        plot_uv(file_path)
        print("Plot saved as: ", file_path[:-3]+'png')
        print("\n")