import lasio as las
import pandas as pd
import numpy as np
import os

#read las file
def read_las(file_path):
    las_file = las.read(file_path)
    df = las_file.df()
    return df

path='/Users/rianrachmanto/pypro/data/SPG Logs for Pressure Prediction'

#iterate through all las files in the folder, add colomn called 'well' based on the file name without _Logs suffix
list_las=[]
for file in os.listdir(path):
    if file.endswith('.las'):
        las_file = las.read(path+'/'+file)
        df = las_file.df()
        well_name = file.split('_')[0]
        df['well'] = well_name
        list_las.append(df)
print(list_las)

#concatenate all dataframes in the list
df = pd.concat(list_las)
#save the dataframe to csv
df.to_csv('/Users/rianrachmanto/pypro/data/SPG Logs for Pressure Prediction/all_well_new.csv')