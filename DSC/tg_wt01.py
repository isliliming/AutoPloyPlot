# read Tg vs different wt% PMMA and plot them

import pandas as pd
import matplotlib.pyplot as plt

cooling_color = '#332288'
heating_color = '#882255'

# Read the Excel file
file_path = '/Users/liming/Desktop/AutoPloyPlot/DSC/data_dsc/LLM_EP09/all_Tg.xlsx'
df = pd.read_excel(file_path)

# Print the first few rows to inspect the data
print(df.head())

# Replace 'wt% PMMA' and 'Tg' with the actual column names if they are different
x = df['pmma']
y_heating = df['heating']
y_cooling = df['cooling']


plt.figure(figsize=(8, 6))
plt.plot(x, y_heating, marker='o', linestyle='-', label='Heating', color=heating_color)
plt.plot(x, y_cooling, marker='s', linestyle='--', label='Cooling', color=cooling_color)
plt.xlabel('wt% PMMA', fontsize=20)
plt.ylabel('Tg / °C', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.title('Tg vs wt% PMMA')
plt.grid(True)
plt.legend()
plt.savefig('tg_wt01.png', dpi=300)
plt.show()