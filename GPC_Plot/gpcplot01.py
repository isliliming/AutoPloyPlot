# read data from excel file and plot GPC curve
import pandas as pd
import matplotlib.pyplot as plt

# read data from excel file
path_file = '../Example Date/GPC YN7plus12/7+1290.xlsx'
data = pd.read_excel(path_file, sheet_name='Slice Table', header=None)



