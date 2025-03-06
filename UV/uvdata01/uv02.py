import pandas as pd
import matplotlib.pyplot as plt

# Read data and find header
# find the data
header_line = 0
with open('LLM_UV_01.csv', 'r') as f:
    for line_num, line in enumerate(f):
        if line.startswith(",Wavelength (nm),Abs"):
            header_line = line_num
            break

# find the sample namess
with open('LLM_UV_01.csv', 'r') as f:
    for line in f:
        if line.startswith('Name,Sample'):
            sample_names = line.split(',')[1:]
            print(sample_names)
            break



# Read data and reverse orders
df = pd.read_csv('LLM_UV_01.csv', skiprows=header_line)
df = df.sort_values('Wavelength (nm)', ascending=True)  # Sort from low to high wavelength

# Extract data
wavelengths = df['Wavelength (nm)']
abs_samples = {
    'Sample 1': df['Abs'],
    'Sample 2': df['Abs.1'],
    'Sample 3': df['Abs.2'],
    'Sample 4': df['Abs.3'],
    'Sample 5': df['Abs.4']
}

# Plotting
plt.figure(figsize=(12, 7))
for sample, absorbance in abs_samples.items():
    plt.plot(wavelengths, absorbance, label=sample)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Absorbance', fontsize=12)
plt.title('UV-Vis Absorbance Spectra (200-800 nm)', fontsize=14)
plt.legend()
plt.xlim(200, 800)  # Explicitly set axis limits
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()