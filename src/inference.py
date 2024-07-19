import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import lasio as las

def load_model():
    with open('/Users/rianrachmanto/pypro/project/press_pred/Model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(df):
    model = load_model()
    prediction = model.predict(df)
    return prediction

def main():
    #import las
    las_file = las.read('/Users/rianrachmanto/pypro/data/SJ-3RD2_OH Log + Mudlog 1.las')
    df = las_file.df()
    df = df.reset_index()
    df.drop(['NPHI','C1','C2',
             'C3','IC4','NC4','IC5','NC5','MW'],axis=1,inplace=True)
    df = df.dropna()
    #predict
    prediction = predict(df)
    print(prediction)
    #insert prediction to df
    df['PRESSURE'] = prediction
    df['PPG']=df['PRESSURE']/0.052/df['DEPTH']
    print(df.tail(10))
    #save to excel
    df.to_excel('/Users/rianrachmanto/pypro/data/data_press/sjd-3.xlsx', index=False)
    #df.to_csv('/Users/rianrachmanto/pypro/project/press_pred/data/pre_sejad3RDtwo.csv', index=False)
if __name__ == '__main__':
    main()