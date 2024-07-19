import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

def base_train_fold(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):
    # Setup KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Helper function to perform training and evaluation
    def evaluate_model(model, X_train, y_train, X_test, y_test, use_scaled=False):
        if use_scaled:
            X_t, y_t = X_train_scaled, X_test_scaled
        else:
            X_t, y_t = X_train, X_test

        # Training using cross-validation
        neg_mse = cross_val_score(model, X_t, y_train, cv=kfold, scoring='neg_mean_squared_error')
        mse_scores = -neg_mse  # Convert to positive MSE scores
        avg_mse = np.mean(mse_scores)  # Calculate average MSE

        # Fit model on the whole training set
        model.fit(X_t, y_train)
        y_pred = model.predict(y_t)
        
        # Calculate other metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print results
        model_name = type(model).__name__
        print(f"{model_name}")
        print("Cross-Validated MSE:", avg_mse)
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R2 Score:", r2)

    # Initialize and evaluate models
    evaluate_model(LinearRegression(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(Ridge(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(Lasso(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(ElasticNet(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(SVR(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(RandomForestRegressor(), X_train, y_train, X_test, y_test)
    evaluate_model(XGBRegressor(), X_train, y_train, X_test, y_test)
    evaluate_model(KNeighborsRegressor(), X_train, y_train, X_test, y_test, use_scaled=True)
    evaluate_model(KernelRidge(), X_train, y_train, X_test, y_test, use_scaled=True)
    
    # OLS requires manual handling for constants
    model = sm.OLS(y_train, sm.add_constant(X_train_scaled))
    results = model.fit()
    y_pred_ols = results.predict(sm.add_constant(X_test_scaled))
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    mae_ols = mean_absolute_error(y_test, y_pred_ols)
    r2_ols = r2_score(y_test, y_pred_ols)
    print('OLS')
    print("Mean Squared Error:", mse_ols)
    print("Mean Absolute Error:", mae_ols)
    print("R2 Score:", r2_ols)

    return

