# read Tg vs different wt% PMMA and plot them
# add predict data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cooling_color = '#332288'
heating_color = '#882255'
Tg_PMMA = 118.05 # the last haeting loop, midpoint
# Tg_PMMA= 500 # the last heating loop, midpoint
Tg_eutectic = 62.83 # the last heating loop, midpoint

# Read the Excel file
file_path = '/Users/liming/Desktop/AutoPloyPlot/DSC/data_dsc/LLM_EP09/all_Tg.xlsx'
df = pd.read_excel(file_path)

# Print the first few rows to inspect the data
print(df.head())

x = df['pmma']
y_heating = df['heating']
y_cooling = df['cooling']

# Fox Equation: 1/Tg = w1/Tg1 + w2/Tg2
# Tg1: PMMA, Tg2: other component (use Tg at 0% PMMA)
Tg1 = Tg_PMMA+273.15
Tg2 = Tg_eutectic+273.15

x_pred = np.linspace(0, 20, 100)
w1_pred = x_pred / 100  # PMMA fraction (0-1)
w2_pred = 1 - w1_pred   # eutectic fraction
Tg_pred = 1 / (w1_pred / Tg1 + w2_pred / Tg2)
Tg_pred = Tg_pred - 273.15

plt.figure(figsize=(8, 6))
plt.plot(x, y_heating, marker='o', linestyle='-', label='Heating', color=heating_color)
plt.plot(x, y_cooling, marker='s', linestyle='--', label='Cooling', color=cooling_color)
plt.plot(x_pred, Tg_pred, linestyle='-.', label='Fox Eq. Prediction', color='black')
plt.xlabel('wt% PMMA', fontsize=20)
plt.ylabel('Tg / °C', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.title('Tg vs wt% PMMA')
plt.grid(True)
plt.legend()
# plt.savefig('tg_wt01_with_prediction.png', dpi=300)
plt.show()