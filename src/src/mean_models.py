import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax_model(endog_returns, exog_driver=None, order=(1,0,1), seasonal_order=(0,0,0,0)):
    """
    Fits an ARMA / ARIMA / SARIMA / ARIMAX / SARIMAX Mean Model.

    Parameters:
        endog_returns (pd.Series): Target asset return series.
        exog_driver (pd.DataFrame/Series): External driver (e.g. VIX returns).
        order (tuple): (p, d, q)
        seasonal_order (tuple): (P, D, Q, s)

    Returns:
        model_fit: Fitted SARIMAXResults object.
        residuals: In-sample model residuals.
    """
    clean_endog = endog_returns.dropna()
    clean_exog = None
    
    if exog_driver is not None:
        clean_exog = exog_driver.loc[clean_endog.index].dropna()
        clean_endog = clean_endog.loc[clean_exog.index]
        
    model = SARIMAX(
        clean_endog,
        exog=clean_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    res = model.fit(disp=False)
    residuals = res.resid
    
    return res, residuals

def evaluate_train_test_accuracy(endog_returns, test_ratio=0.2):
    """
    Performs 80/20 Train-Test split validation and calculates classical Machine Learning
    accuracy metrics: R2 Score, RMSE, MAE, MAPE, and Directional Hit Rate (%).
    """
    series = endog_returns.dropna()
    n = len(series)
    train_size = int(n * (1.0 - test_ratio))
    
    train_data = series.iloc[:train_size]
    test_data = series.iloc[train_size:]
    
    # Fit model on training set
    res_train, _ = fit_sarimax_model(train_data, order=(1,0,1))
    
    # Out-of-sample forecast for test set duration
    predictions = res_train.forecast(steps=len(test_data))
    
    y_true = test_data.values
    y_pred = predictions.values
    
    # Calculations
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Non-zero mask for MAPE
    nz_mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[nz_mask] - y_pred[nz_mask]) / y_true[nz_mask])) * 100.0 if np.sum(nz_mask) > 0 else 0.0
    
    # Directional Hit Rate (% of days model correctly predicted Up vs Down sign)
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = np.mean(correct_direction) * 100.0
    
    return {
        "train_size": train_size,
        "test_size": len(test_data),
        "r2_score": float(r2_score),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "directional_accuracy": float(directional_accuracy)
    }

def get_sarimax_diagnostics(model_fit):
    """
    Extracts summary statistics (AIC, BIC, Log-Likelihood) from fitted SARIMAX model.
    """
    return {
        "model_type": model_fit.model.__class__.__name__,
        "aic": float(model_fit.aic),
        "bic": float(model_fit.bic),
        "log_likelihood": float(model_fit.llf),
        "p_orders": model_fit.model.order,
        "seasonal_orders": model_fit.model.seasonal_order,
        "has_exog": model_fit.model.k_exog > 0
    }
