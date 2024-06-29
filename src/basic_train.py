import pandas as pd
import numpy as np
#import linear regression model
from sklearn.linear_model import LinearRegression
#import SVR model
from sklearn.svm import SVR
#import RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor
#import XGBRegressor model
from xgboost import XGBRegressor
#import mean_squared_error
#import ridge regression model
from sklearn.linear_model import Ridge
# import laso regression model
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
#import mean_absolute_error
from sklearn.metrics import mean_absolute_error
# import R2 score
from sklearn.metrics import r2_score

def base_train(X_train,y_train,X_test,y_test,X_train_scaled,X_test_scaled):
    #Linear Regression
    #create a model
    model = LinearRegression()
    #fit the model
    model.fit(X_train_scaled, y_train)
    #predict the model
    y_pred = model.predict(X_test_scaled)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('Linear Regression')
    print('Mean Squared Error:', mse)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    #calculate the R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2 Score:', r2)
    
    #SVR
    #create a model
    model = SVR()
    #fit the model
    model.fit(X_train_scaled, y_train)
    #predict the model
    y_pred = model.predict(X_test_scaled)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('SVR')
    print('Mean Squared Error:', mse)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    #calculate the R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2 Score:', r2)
    
    #RandomForestRegressor
    #create a model
    model = RandomForestRegressor()
    #fit the model
    model.fit(X_train, y_train)
    #predict the model
    y_pred = model.predict(X_test)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('RandomForestRegressor')
    print('Mean Squared Error:', mse)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    #calculate the R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2 Score:', r2)
    
    #XGBRegressor
    #create a model
    model = XGBRegressor()
    #fit the model
    model.fit(X_train, y_train)
    #predict the model
    y_pred = model.predict(X_test)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('XGBRegressor')
    print('Mean Squared Error:', mse)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    #calculate the R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2 Score:', r2)

    #Ridge
    #create a model
    model = Ridge()
    #fit the model
    model.fit(X_train_scaled, y_train)
    #predict the model
    y_pred = model.predict(X_test_scaled)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('Ridge')
    print('Mean Squared Error:', mse)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    #calculate the R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2 Score:', r2)

    #Lasso
    #create a model
    model = Lasso()
    #fit the model
    model.fit(X_train_scaled, y_train)
    #predict the model
    y_pred = model.predict(X_test_scaled)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('Lasso')
    print('Mean Squared Error:', mse)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mae)
    #calculate the R2 score
    r2 = r2_score(y_test, y_pred)   
    print('R2 Score:', r2)

    return
