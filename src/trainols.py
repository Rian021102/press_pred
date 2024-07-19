import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

def train_ols(X_train_scaled,y_train,X_test_scaled,y_test):
    # OLS requires manual handling for constants
    model = sm.OLS(y_train, sm.add_constant(X_train_scaled))
    results = model.fit()
    y_pred = results.predict(sm.add_constant(X_test_scaled))
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = results.rsquared
    #save the model
    with open('/Users/rianrachmanto/pypro/project/press_pred/Model/ols_model.pkl', 'wb') as f:
        pickle.dump(results, f)
    return mse, r2, mae, y_pred