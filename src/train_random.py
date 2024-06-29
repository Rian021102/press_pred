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

def train_model(X_train,y_train,X_test,y_test):
    #train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    #predict the test data
    y_pred = model.predict(X_test)
    #calculate the mean squared error
    #feature importance
    feature_importances = model.feature_importances_
    mse = mean_squared_error(y_test, y_pred)
    #calculate the r2 score
    r2 = r2_score(y_test, y_pred)
    #calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    #save model
    with open('/Users/rianrachmanto/pypro/project/press_pred/Model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return mse, r2, mae,y_pred,feature_importances