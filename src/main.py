import numpy as np
import pandas as pd
from eda import perform_eda
from datacheck import load_data, clean_train, clean_test
#import standardscaler
from sklearn.preprocessing import StandardScaler
from basic_train import base_train
from train_random import train_model
from train_tree import train_model_tree
from trainxgb import train_model_xgb
from finetunerandom import train_tune_random
# from builddnn import build_dnn, plot_loss, plot_mae, plot_predictions
from trainlass import train_lasso
from trainols import train_ols
from base_train_mod import base_train_fold
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    path = '/Users/rianrachmanto/pypro/data/SPG Logs for Pressure Prediction/all_well_pressure_new.csv'
    X_train, y_train, X_test, y_test = load_data(path)
    print(X_train.head())
    print(y_train.head())
    X_train, y_train = clean_train(X_train, y_train)
    X_test, y_test = clean_test(X_test, y_test)
    # # perform log transformation on X_train and X_test
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # perform_eda(X_train, y_train)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model, history = build_dnn(X_train_scaled, y_train, X_test_scaled, y_test)
    # plot_loss(history)
    # plot_mae(history)
    

    # # Train and evaluate models
    # base_train(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

    #train fold
    base_train_fold(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

    #train ols
    # mse, r2, mae, y_pred = train_ols(X_train_scaled, y_train, X_test_scaled, y_test)

    #train lasso
    # mse, r2, mae, y_pred = train_lasso(X_train_scaled, y_train, X_test_scaled, y_test)
   
    # mse, r2, mae, y_pred,feature_importances = train_model(X_train, y_train, X_test, y_test)
    # print('Mean Squared Error:', mse)
    # print('R2 Score:', r2)
    # print('Mean Absolute Error:', mae)

    # # # Plot predictions
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred, alpha=0.3)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    # plt.xlabel('Actual')
    # plt.ylabel('Predicted')
    # plt.title('Actual vs. Predicted')

    # # Plot residuals
    # residuals = y_test - y_pred
    # plt.figure(figsize=(10, 6))
    # plt.hist(residuals, bins=30, edgecolor='black')
    # plt.xlabel('Residuals')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Residuals')

    # features = X_train.columns
    # indices = np.argsort(feature_importances)
    # plt.figure(figsize=(10, 6))
    # plt.title('Feature Importances')
    # plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    # plt.xlabel('Relative Importance')
   
    plt.show()

if __name__ == '__main__':
    main()
