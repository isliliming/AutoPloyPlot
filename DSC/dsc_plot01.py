# read the data from a text file and convert it to a CSV file
# import csv
# # Load the data from the text file
# file_path ='/Users/liming/Desktop/AutoPloyPlot/DSC/data_dsc/LLM_EP09/LLM_EP0901.txt'
# # Read the file content
# with open(file_path, 'r') as file:
#     data = file.readlines()
# # Initialize a list to hold all sets of columns
# all_columns = []
# current_columns = []
# # Header to identify when to start a new set of columns
# header = "Time\tTemperature\tHeat Flow (Normalized)"
# # Process each line to extract values based on the header occurrence
# for line in data:
#     # Remove any extra whitespace from the line
#     line = line.strip()
#     # Check if the line matches the header
#     if line == header:
#         if current_columns:
#             all_columns.append(current_columns)
#             current_columns = []
#     else:
#         # Split the line by whitespace to get individual values
#         values = line.split()
#         if len(values) == 3:  # Assuming the line has all three values
#             current_columns.append(values)
# # Append the last set if not empty
# if current_columns:
#     all_columns.append(current_columns)
# # Determine the maximum length of data to align all columns
# max_length = max(len(set_columns) for set_columns in all_columns)
# # Pad shorter columns with empty strings to align with the longest set
# for i in range(len(all_columns)):
#     if len(all_columns[i]) < max_length:
#         all_columns[i].extend([["", "", ""]] * (max_length - len(all_columns[i])))
# # Transpose the data to group by rows for each set
# transposed_data = list(zip(*all_columns))
# # Flatten the transposed data into rows for CSV
# flattened_data = []
# for group in transposed_data:
#     flattened_row = []
#     for set_columns in group:
#         flattened_row.extend(set_columns)
#     flattened_data.append(flattened_row)
# # Define the path for the output CSV file
# output_file_path = '/Users/liming/Desktop/AutoPloyPlot/DSC/data_dsc/LLM_EP09/LLM_EP0901.csv'
# # Generate the header row for the CSV
# header_row = []
# for i in range(1, len(all_columns) + 1):
#     header_row.extend([f"Time (min) Set {i}", f"Temperature (°C) Set {i}", f"Heat Flow (W/g) Set {i}"])
# # Write the extracted data to a CSV file
# with open(output_file_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # Write the dynamically generated header row
#     writer.writerow(header_row)
#     # Write the flattened data rows
#     writer.writerows(flattened_data)
# print(f"Data has been exported to {output_file_path}")


import pandas as pd
import matplotlib.pyplot as plt
# Define the path to your CSV file
csv_file_path = '/Users/liming/Desktop/AutoPloyPlot/DSC/data_dsc/LLM_EP09/LLM_EP0901.csv'
# Try reading the CSV file with a different encoding
try:
    df = pd.read_csv(csv_file_path, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 encoding failed, trying ISO-8859-1.")
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
#print(df.head)
df.drop(0,inplace=True)
cols = [0,1,2]
df.drop(df.columns[cols],axis=1,inplace=True)
df.reset_index
#print(df.head)
# Convert necessary columns to numeric for plotting
for i in range(0, len(df.columns), 3):
    df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')  # Time
    df.iloc[:, i+1] = pd.to_numeric(df.iloc[:, i+1], errors='coerce')  # Temperature
    df.iloc[:, i+2] = pd.to_numeric(df.iloc[:, i+2], errors='coerce')  # Heat Flow
try:
    total_sets = len(df.columns) // 3
    print("Found:", total_sets, "Sets to plot")
except:
    print("Found:", total_sets, "Sets to plot, cannot divide by 3")
StartColour = 'grey'
GreyColour = 'grey'
ColourList=[]    # this defines the colour list
ColourCounter=0  # for each cycle an additional mark is added to this list for each cycle
HotColour=['#F7374F','#88304E','#522546','#2C2C2C']   #these are the hex codes for the hot colours
CoolColour=['#009990','#578FCA','#074799','#D1F8EF']# these are the hex codes for the cool colours
CoolCounter=0
HotCounter=0
for loop in range(0,len(df.columns),3):
    #print(loop)
    if loop ==0:
        ColourList.append(StartColour) # this sets the first datatset to grey
    elif ColourCounter ==0:
        ColourList.append(GreyColour)
        ColourCounter=ColourCounter+1
    elif ColourCounter ==1:
        ColourList.append(CoolColour[CoolCounter])
        CoolCounter=CoolCounter+1
        ColourCounter=ColourCounter+1
    elif ColourCounter ==2:
        ColourList.append(GreyColour)
        ColourCounter=ColourCounter+1
    elif ColourCounter ==3:
        ColourList.append(HotColour[HotCounter])
        HotCounter=HotCounter+1
        ColourCounter=0
# Create a new figure for the plot
plt.figure(figsize=(12, 8))
for i in range(0,len(df.columns)-3,3): # this is where you say which satasets can be exclused. the  -3,3 will remove the first and last data set
    x=int(i/3)  # to polot x which ic the number of datsets then it muct be an interge. somtines if you divide by i then it nolonger is hence the need for int()
    #print(i,x)
    if i == 0:
        #plt.plot(df.iloc[:, i+1], df.iloc[:, i+2], c=ColourList[x])    # this is added when you want to include the first initail heating cycle elce keep hidden
        True
    else:
        plt.plot(df.iloc[:, i+1], df.iloc[:, i+2], c=ColourList[x]) #plots everything after the i==0
#plt.title('VM6 oven dry)' , fontsize=16)
plt.xlabel('temperature/ °C', fontsize=14)
plt.ylabel('heat flow/ Wg$^{-1}$', fontsize=14)
plt.tight_layout()
plt.show()





