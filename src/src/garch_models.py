import numpy as np
import pandas as pd
from arch import arch_model

def fit_garch_volatility_models(returns_series, p=1, q=1, horizon=10):
    """
    Fits GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) volatility models 
    and generates h-step-ahead conditional volatility forecasts.

    Returns:
        results (dict): Fitted model objects, diagnostic parameters, and volatility forecast arrays.
    """
    clean_returns = returns_series.dropna() * 100.0 # Scale to percentage returns for numerical stability
    
    # -------------------------------------------------------------
    # 1. FIT STANDARD GARCH(1,1)
    # -------------------------------------------------------------
    garch_spec = arch_model(clean_returns, mean='AR', lags=1, vol='Garch', p=p, q=q, dist='normal')
    garch_fit = garch_spec.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=horizon, method='analytic')
    garch_vol_pred = np.sqrt(garch_forecast.variance.iloc[-1].values) / 100.0 # Scale back
    
    # -------------------------------------------------------------
    # 2. FIT EGARCH(1,1) (Exponential GARCH - Leverage Effect)
    # -------------------------------------------------------------
    egarch_spec = arch_model(clean_returns, mean='AR', lags=1, vol='EGARCH', p=p, q=q, dist='normal')
    egarch_fit = egarch_spec.fit(disp='off')
    # EGARCH requires simulation method for multi-period analytic forecasts
    method_egarch = 'simulation' if horizon > 1 else 'analytic'
    egarch_forecast = egarch_fit.forecast(horizon=horizon, method=method_egarch)
    egarch_vol_pred = np.sqrt(egarch_forecast.variance.iloc[-1].values) / 100.0
    
    # Extract leverage parameter gamma if present
    gamma_param = 0.0
    for param_name in ['gamma[1]', 'gamma', 'gamma_1']:
        if param_name in egarch_fit.params:
            gamma_param = float(egarch_fit.params[param_name])
            break
        
    # -------------------------------------------------------------
    # 3. FIT GJR-GARCH(1,1) (Threshold Asymmetric GARCH)
    # -------------------------------------------------------------
    gjr_spec = arch_model(clean_returns, mean='AR', lags=1, vol='GARCH', p=p, o=1, q=q, dist='normal')
    gjr_fit = gjr_spec.fit(disp='off')
    gjr_forecast = gjr_fit.forecast(horizon=horizon, method='analytic')
    gjr_vol_pred = np.sqrt(gjr_forecast.variance.iloc[-1].values) / 100.0
    
    # Historical in-sample conditional volatility
    garch_cond_vol = (garch_fit.conditional_volatility / 100.0).loc[returns_series.index]
    egarch_cond_vol = (egarch_fit.conditional_volatility / 100.0).loc[returns_series.index]
    gjr_cond_vol = (gjr_fit.conditional_volatility / 100.0).loc[returns_series.index]
    
    return {
        "garch_fit": garch_fit,
        "egarch_fit": egarch_fit,
        "gjr_fit": gjr_fit,
        "garch_cond_vol": garch_cond_vol,
        "egarch_cond_vol": egarch_cond_vol,
        "gjr_cond_vol": gjr_cond_vol,
        "garch_forecast_vol": garch_vol_pred,
        "egarch_forecast_vol": egarch_vol_pred,
        "gjr_forecast_vol": gjr_vol_pred,
        "egarch_leverage_gamma": gamma_param,
        "has_leverage_effect": bool(gamma_param < 0)
    }
