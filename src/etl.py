import numpy as np
import pandas as pd
import lasio
from scipy.spatial import cKDTree

#load asc file
data=pd.read_csv('/Users/rianrachmanto/pypro/project/press_pred/data/Sepinggan Data for Pressure Prediction/Mudlog Ascii/S-5RD2.ASC',
                 delim_whitespace=True,skiprows=2)
# Renaming columns to make them more descriptive and unique
new_column_names = {
    "ft": "Bit Depth",
    "ft.1": "TVD_Depth",
    "ft/hr": "ROP2",
    "klbs": "WOH",
    "klbs.1": "WOB",
    "rpm": "RPM",
    "lb*ft": "TD TRQ",
    "psi": "SPP",
    "gpm": "FLWpmps",
    "ppg": "MW IN",
    "deg": "TMP IN",
    "F": "TMP OUT",
    "ppg.1": "ECD at BD",
    "U": "T GAS",
    "%": "C1 main",
    "%.1": "C2 main",
    "%.2": "C3 main",
    "%.3": "iC4 main",
    "%.4": "nC4 main",
    "%.5": "iC5 main",
    "%.6": "nC5 main"
}

# Apply the renaming
data.rename(columns=new_column_names, inplace=True)
add_column = {'Well_Name':'R-3RD2'}
data = data.assign(**add_column)

print(data.head())
print(data.describe())

# Load las file
las=lasio.read('/Users/rianrachmanto/pypro/project/press_pred/data/Sepinggan Data for Pressure Prediction/OH Logs/S-5RD2_Logs.las')
las=las.df()
add_column = {'Well_Name':'R-3RD2'}
las = las.assign(**add_column)
las.reset_index(inplace=True)
#replace TVDSS to TVD_Depth
las.rename(columns={'TVDSS':'TVD_Depth'},inplace=True)
print(las.head())
print(las.describe())

#merge las and asc on TVD_Depth
data['TVD_Depth'] = data['TVD_Depth'].astype(float)
las['TVD_Depth'] = las['TVD_Depth'].astype(float)
#merge based on the closest TVD_Depth
tree = cKDTree(data[['TVD_Depth']])
distances, indices = tree.query(las[['TVD_Depth']])
las['TVD_Depth'] = data['TVD_Depth'].values[indices]
merged = pd.merge(las, data, on=['TVD_Depth','Well_Name'], how='inner')
print(merged.head())
#remove columns ppm.9 and Nb
#merged = merged.drop(columns=['ppm.9','Nb.'],axis=1,inplace=False)
print(merged.shape)
#dropna
merged.dropna(inplace=True)
print(merged.shape)
#save to csv with filename taken from the asc file
merged.to_csv('/Users/rianrachmanto/pypro/project/press_pred/data/Sepinggan Data for Pressure Prediction/Merged Data/S-5RD2.csv',index=False)

#press=pd.read_excel('/Users/rianrachmanto/pypro/project/press_pred/data/Sepinggan Data for Pressure Prediction/GSA_MDT-RFT_data.xls')
#print(press.head())


