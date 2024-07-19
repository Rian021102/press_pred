import pandas as pd
#import lasso regression
from sklearn.linear_model import Lasso
#import mean squared error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_lasso(X_train_scaled,y_train,X_test_scaled,y_test):
    #create lasso regression model
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean Squared Error:', mse)
    print('R2 Score:', r2)
    print('Mean Absolute Error:', mae)
    return mse, r2, mae, y_pred