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
    las_file = las.read('/Users/rianrachmanto/pypro/project/press_pred/data/SJ-2RD2BP1_OH Log + Mudlog 1.las')
    df = las_file.df()
    df = df.reset_index()
    # drop column DT
    df = df.drop(columns=['DT'])
    df = df.dropna()
    #predict
    prediction = predict(df)
    print(prediction)
    #insert prediction to df
    df['PRESSURE'] = prediction
    df['PPG']=df['PRESSURE']/0.052/df['DEPTH']
    print(df.tail(10))
    #save to excel
    df.to_excel('/Users/rianrachmanto/pypro/project/press_pred/data/pre_sejad2RDtwo.xlsx', index=False)
    #df.to_csv('/Users/rianrachmanto/pypro/project/press_pred/data/pre_sejad3RDtwo.csv', index=False)
if __name__ == '__main__':
    main()