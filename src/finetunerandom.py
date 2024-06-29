import numpy as np
import pandas as pd
#import random forest regressor
from sklearn.ensemble import RandomForestRegressor
# import mean squared error
from sklearn.metrics import mean_squared_error
# import r2 score
from sklearn.metrics import r2_score
# import mean absolute error
from sklearn.metrics import mean_absolute_error
# import pickle
import pickle
#import GridSearchCV
from sklearn.model_selection import GridSearchCV

'''
This function is used to fine tune the random forest regressor model using hyperparameter tuning
'''

def train_tune_random(X_train,y_train,X_test,y_test):
    #create random forest regressor model
    rf = RandomForestRegressor()
    #create parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    #create GridSearchCV object
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    #fit the model
    grid_search.fit(X_train, y_train)
    #predict the target values
    y_pred = grid_search.predict(X_test)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    #calculate the r2 score
    r2 = r2_score(y_test, y_pred)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    #print the mean squared error
    print("Mean Squared Error:",mse)
    #print the r2 score
    print("R2 Score:",r2)
    #print the mean absolute error
    print("Mean Absolute Error:",mae)
    #feature importance
    feature_importances = grid_search.best_estimator_.feature_importances_
    #save model
    with open('/Users/rianrachmanto/pypro/project/press_pred/Model/model_tuned.pkl', 'wb') as f:
        pickle.dump(grid_search, f)
    
    return mse, r2, mae,y_pred,feature_importances
