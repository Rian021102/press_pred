import numpy as np
import pandas as pd
#import train_test_split
from sklearn.model_selection import train_test_split

def load_data(path):
    df=pd.read_csv(path)
    #drop columns DEPTH_PRESSURE
    df.drop(['DEPTH_PRESSURE'],axis=1,inplace=True)
    #except for WELL_NAME, convert all columns to float
    for col in df.columns:
        if col != 'WELL_NAME':
            df[col]=df[col].astype(float)
    #splitting the data
    X = df.drop(['PRESSURE'], axis=1)
    y = df['PRESSURE']
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,y_train,X_test,y_test

def clean_train(X_train,y_train):
   #merge back the X_train and y_train
    df=pd.concat([X_train,y_train],axis=1)
    #drop WELL_NAME
    df.drop(['WELL_NAME'],axis=1,inplace=True)
    #drop rows with missing values
    df.dropna(inplace=True)
    #split the data into X_train and y_train
    X_train = df.drop(['PRESSURE'], axis=1)
    y_train = df['PRESSURE']
    return X_train,y_train

def clean_test(X_test,y_test):
    #merge back the X_train and y_train
    df=pd.concat([X_test,y_test],axis=1)
    #drop WELL_NAME
    df.drop(['WELL_NAME'],axis=1,inplace=True)
    #drop rows with missing values
    df.dropna(inplace=True)
    #split the data into X_train and y_train
    X_test = df.drop(['PRESSURE'], axis=1)
    y_test = df['PRESSURE']
    return X_test,y_test

