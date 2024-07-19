import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

#load excel file
def load_las(file):
    df=pd.read_csv(file)
    # df.drop(columns=['DT','UNKNOWN:1', 'UNKNOWN:2','UNKNOWN:3', 'UNKNOWN:4', 
    #                      'UNKNOWN:5', 'UNKNOWN:6', 'UNKNOWN:7'], inplace=True)
    df.drop(columns=['DT'], inplace=True)
    df.rename(columns={'well':'WELL_NAME'}, inplace=True)
    df['DEPTH']=df['DEPTH'].astype(float)
    return df

def load_excel(file):
    df=pd.read_excel(file)
    df.rename(columns={'MARKER_MD':'DEPTH'}, inplace=True)
    df['DEPTH']=df['DEPTH'].astype(float)
    replacing=['SEPINGGAN','SEDANDANG',
               'SEGUNI','SEJADI']
    #remove words in replacing list from WELL_NAME
    df['WELL_NAME']=df['WELL_NAME'].apply(lambda x: ' '.join([word for word in x.split() if word not in replacing]))
    return df



file_ex='/Users/rianrachmanto/pypro/data/data_press/GSA_MDT-RFT_data.xlsx'
file_las='/Users/rianrachmanto/pypro/data/SPG Logs for Pressure Prediction/all_well_new.csv'

df_las=load_las(file_las)
df_ex=load_excel(file_ex)


#copy df_ex PRESSURE to df_las if the WELL_NAME match and copy pressure to closset DEPTH

def copy_pressure(df_las, df_ex):
    #copy pressure to df_las
    df_las['PRESSURE']=np.nan
    df_las['DEPTH_PRESSURE']=np.nan
    for i in range(len(df_ex)):
        well_name=df_ex.loc[i,'WELL_NAME']
        pressure=df_ex.loc[i,'PRESSURE']
        depth=df_ex.loc[i,'DEPTH']
        #find the closest depth in df_las
        tree = KDTree(df_las[['DEPTH']])
        dist, ind = tree.query([[depth]])
        ind=ind[0][0]
        df_las.loc[ind,'PRESSURE']=pressure
        df_las.loc[ind,'DEPTH_PRESSURE']=depth
    return df_las

df_las=copy_pressure(df_las, df_ex)
#save to csv
df_las.to_csv('/Users/rianrachmanto/pypro/data/SPG Logs for Pressure Prediction/all_well_pressure_new.csv', index=False)
print(df_las.head())