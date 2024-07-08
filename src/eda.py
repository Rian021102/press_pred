import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(X_train,y_train):
    # merge the X_train and y_train
    df = pd.concat([X_train, y_train], axis=1)
    # check for missing values
    print(df.isnull().sum())
    # plot histogram for each feature
    df.hist(bins=30, figsize=(10, 10))
    plt.show()
    # plot correlation matrix heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()
    
    