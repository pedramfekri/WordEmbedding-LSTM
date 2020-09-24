'''
this class converts a csv file to text format
'''

import os
import pandas as pd

csv_file = "main_dataframe.csv"
dataset = pd.read_csv(csv_file)
print(dataset.shape[0])
with open("dataset.txt", 'w', encoding="utf-8") as f:
    for i in range(dataset.shape[0]):
        f.write(dataset.iloc[i, 0])
        f.write('\n')
