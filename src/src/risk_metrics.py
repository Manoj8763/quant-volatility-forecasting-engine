import numpy as np
import pandas as pd
from scipy import stats

def calculate_var_and_expected_shortfall(returns_series, cond_volatility_series, portfolio_value=1000000.0, confidence_levels=[0.95, 0.99]):
    """
    Computes Value at Risk (VaR) and Expected Shortfall (ES) using 
    Filtered Historical Simulation (FHS) + Extreme Value Theory (EVT) residuals.

    Returns:
        results (dict): VaR percentages, VaR dollar amounts, ES dollar amounts, and historical VaR envelopes.
    """
    returns = returns_series.dropna()
    vol = cond_volatility_series.loc[returns.index].dropna()
    aligned_idx = returns.index.intersection(vol.index)
    
    returns = returns.loc[aligned_idx]
    vol = vol.loc[aligned_idx]
    
    # Standardize Residuals z_t = r_t / sigma_t
    # Avoid division by zero
    vol_safe = np.where(vol == 0, 1e-6, vol)
    std_residuals = returns / vol_safe
    std_residuals = pd.Series(std_residuals, index=aligned_idx).replace([np.inf, -np.inf], np.nan).dropna()
    
    current_vol = float(vol.iloc[-1]) # Latest conditional volatility
    
    var_results = {}
    for alpha in confidence_levels:
        percentile = (1.0 - alpha) * 100.0
        
        # Empirical Quantile from Standardized Residuals
        z_alpha = np.percentile(std_residuals, percentile)
        
        # Dynamic VaR Return Level = z_alpha * current_vol
        var_return_pct = z_alpha * current_vol
        var_dollar = np.abs(var_return_pct) * portfolio_value
        
        # Expected Shortfall (Tail Loss) = E[Residuals | Residuals < z_alpha] * current_vol
        tail_residuals = std_residuals[std_residuals <= z_alpha]
        if len(tail_residuals) > 0:
            es_z_alpha = np.mean(tail_residuals)
        else:
            es_z_alpha = z_alpha * 1.2
            
        es_return_pct = es_z_alpha * current_vol
        es_dollar = np.abs(es_return_pct) * portfolio_value
        
        # Historical VaR Envelope Series across sample
        historical_var_returns = z_alpha * vol
        historical_var_series = np.abs(historical_var_returns) * portfolio_value
        
        var_results[alpha] = {
            "alpha": alpha,
            "z_alpha": float(z_alpha),
            "var_return_pct": float(var_return_pct),
            "var_dollar": float(var_dollar),
            "es_return_pct": float(es_return_pct),
            "es_dollar": float(es_dollar),
            "historical_var_series": historical_var_series,
            "historical_var_returns": historical_var_returns
        }
        
    return {
        "portfolio_value": portfolio_value,
        "current_daily_volatility": current_vol,
        "metrics_by_confidence": var_results
    }
