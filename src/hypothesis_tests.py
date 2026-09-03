import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def run_adf_test(series, regression='c'):
    """
    Augmented Dickey-Fuller (ADF) Test.
    H0: Unit root is present (Non-stationary).
    H1: Series is stationary.
    """
    clean_series = series.dropna()
    res = adfuller(clean_series, autolag='AIC', regression=regression)
    
    return {
        "test_name": "Augmented Dickey-Fuller (ADF)",
        "test_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags_used": int(res[2]),
        "n_obs": int(res[3]),
        "critical_values": res[4],
        "is_stationary": bool(res[1] < 0.05),
        "conclusion": "Stationary (Reject H0)" if res[1] < 0.05 else "Non-Stationary (Fail to Reject H0)"
    }

def run_kpss_test(series, regression='c'):
    """
    KPSS Stationarity Test.
    H0: Series is stationary.
    H1: Unit root is present (Non-stationary).
    """
    clean_series = series.dropna()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = kpss(clean_series, regression=regression, nlags='auto')
        
    return {
        "test_name": "KPSS Stationarity Test",
        "test_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags_used": int(res[2]),
        "critical_values": res[3],
        "is_stationary": bool(res[1] > 0.05),
        "conclusion": "Stationary (Fail to Reject H0)" if res[1] > 0.05 else "Non-Stationary (Reject H0)"
    }

def run_ljung_box_test(residuals, lags=10):
    """
    Ljung-Box Q-Test for residual autocorrelation.
    H0: Residuals are independently distributed (White noise).
    """
    clean_res = residuals.dropna()
    res = acorr_ljungbox(clean_res, lags=[lags], return_df=True)
    p_val = float(res['lb_pvalue'].values[0])
    stat = float(res['lb_stat'].values[0])
    
    return {
        "test_name": "Ljung-Box Q-Test",
        "test_stat": stat,
        "p_value": p_val,
        "lags": lags,
        "is_white_noise": bool(p_val > 0.05),
        "conclusion": "White Noise (Fail to Reject H0)" if p_val > 0.05 else "Autocorrelated Residuals (Reject H0)"
    }

def run_arch_lm_test(residuals, lags=5):
    """
    Engle's ARCH-LM Test for Conditional Heteroskedasticity.
    H0: No ARCH effects (Constant variance).
    H1: ARCH effects present (Volatility clustering).
    """
    clean_res = residuals.dropna()
    res = het_arch(clean_res, nlags=lags)
    lm_stat, p_val = float(res[0]), float(res[1])
    
    return {
        "test_name": "Engle's ARCH-LM Test",
        "test_stat": lm_stat,
        "p_value": p_val,
        "lags": lags,
        "has_arch_effects": bool(p_val < 0.05),
        "conclusion": "ARCH Effects Present (Reject H0 - GARCH Required)" if p_val < 0.05 else "Homoskedastic (No ARCH Effects)"
    }

def run_johansen_test(df_prices, det_order=0, k_ar_diff=1):
    """
    Johansen Cointegration Test across multiple price series.
    """
    clean_df = df_prices.dropna().copy()
    res = coint_johansen(clean_df, det_order=det_order, k_ar_diff=k_ar_diff)
    
    trace_stats = res.lr1
    crit_vals = res.cvt # 90%, 95%, 99%
    
    r_cointegrating = 0
    num_series = clean_df.shape[1]
    for i in range(num_series):
        if trace_stats[i] > crit_vals[i, 1]: # 95% critical value
            r_cointegrating += 1
            
    return {
        "test_name": "Johansen Cointegration Test",
        "cointegrating_rank": r_cointegrating,
        "trace_stats": trace_stats.tolist(),
        "crit_values_95": crit_vals[:, 1].tolist(),
        "is_cointegrated": bool(r_cointegrating > 0),
        "conclusion": f"Cointegration Detected (Rank r = {r_cointegrating})" if r_cointegrating > 0 else "No Cointegration Detected (r = 0)"
    }

def run_kupiec_pof_test(actual_returns, var_series, confidence_level=0.99):
    """
    Kupiec Proportion of First Failures (POF) Likelihood Ratio Test for VaR.
    H0: Model breach rate equals expected alpha (1 - confidence_level).
    """
    returns = actual_returns.values if isinstance(actual_returns, (pd.Series, pd.DataFrame)) else actual_returns
    var_vals = var_series.values if isinstance(var_series, (pd.Series, pd.DataFrame)) else var_series
    
    N = len(returns)
    # Breach occurs when actual loss exceeds VaR (returns < -VaR)
    breaches = (returns < -np.abs(var_vals)).astype(int)
    x = np.sum(breaches) # Total number of breaches
    p = 1.0 - confidence_level # Expected breach rate (e.g. 0.01 for 99%)
    
    p_hat = x / N if N > 0 else 0
    
    if x == 0:
        lr_stat = -2 * N * np.log(1 - p)
    else:
        lr_stat = -2 * ((N - x) * np.log(1 - p) + x * np.log(p)) + 2 * ((N - x) * np.log(1 - p_hat) + x * np.log(p_hat))
        
    p_value = 1.0 - stats.chi2.cdf(lr_stat, df=1)
    
    return {
        "test_name": "Kupiec POF Backtest",
        "total_obs": int(N),
        "observed_breaches": int(x),
        "expected_breaches": float(N * p),
        "observed_breach_rate": float(p_hat),
        "expected_breach_rate": float(p),
        "lr_stat": float(lr_stat),
        "p_value": float(p_value),
        "is_passed": bool(p_value > 0.05),
        "conclusion": "VaR Model Accepted (Fail to Reject H0)" if p_value > 0.05 else "VaR Model Rejected (Incorrect Breach Frequency)"
    }

def run_christoffersen_test(actual_returns, var_series):
    """
    Christoffersen Independence Test for VaR Breach Clustering.
    H0: VaR breaches are independently distributed over time.
    """
    returns = actual_returns.values if isinstance(actual_returns, (pd.Series, pd.DataFrame)) else actual_returns
    var_vals = var_series.values if isinstance(var_series, (pd.Series, pd.DataFrame)) else var_series
    
    breaches = (returns < -np.abs(var_vals)).astype(int)
    
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(breaches)):
        prev, curr = breaches[i-1], breaches[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        elif prev == 1 and curr == 1: n11 += 1
        
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
    
    if p01 == 0 or p11 == 0 or p == 0:
        lr_stat = 0.0
    else:
        num = ((1 - p)**(n00 + n10)) * (p**(n01 + n11))
        den = ((1 - p01)**n00) * (p01**n01) * ((1 - p11)**n10) * (p11**n11)
        lr_stat = -2 * np.log(num / den) if den > 0 and num > 0 else 0.0
        
    p_value = 1.0 - stats.chi2.cdf(lr_stat, df=1)
    
    return {
        "test_name": "Christoffersen Independence Test",
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "p01_rate": float(p01),
        "p11_rate": float(p11),
        "lr_stat": float(lr_stat),
        "p_value": float(p_value),
        "is_passed": bool(p_value > 0.05),
        "conclusion": "Independent Breaches (Fail to Reject H0)" if p_value > 0.05 else "Clustered Breaches (Reject H0 - Model Fails Independence)"
    }
