import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

def fit_vecm_pairs_trading(df_prices, asset_a="JPM", asset_b="BAC", k_ar_diff=1, z_entry=2.0):
    """
    Fits Vector Error Correction Model (VECM) for Statistical Arbitrage (Pairs Trading).

    Returns:
        spread (pd.Series): Cointegrating equilibrium spread.
        z_score (pd.Series): Standardized spread Z-score.
        signals (pd.Series): Trading signals (-1 = Short Spread, +1 = Long Spread, 0 = Hold).
        vecm_res: Fitted VECM object.
    """
    if asset_a not in df_prices.columns or asset_b not in df_prices.columns:
        raise KeyError(f"Assets {asset_a} and {asset_b} must be present in df_prices DataFrame.")
        
    prices = df_prices[[asset_a, asset_b]].dropna().copy()
    
    # Johansen test
    joh_res = coint_johansen(prices, det_order=0, k_ar_diff=k_ar_diff)
    beta = joh_res.evec[:, 0] # Cointegrating vector
    
    # Fit VECM
    vecm = VECM(prices, k_ar_diff=k_ar_diff, coint_rank=1, deterministic='c')
    vecm_res = vecm.fit()
    
    # Cointegrating Spread: S_t = Price_A - beta_ratio * Price_B
    beta_ratio = -beta[1] / beta[0] if beta[0] != 0 else 1.0
    spread = prices[asset_a] - beta_ratio * prices[asset_b]
    
    # Rolling Z-Score
    mean_spread = spread.rolling(window=30, min_periods=5).mean()
    std_spread = spread.rolling(window=30, min_periods=5).std()
    z_score = (spread - mean_spread) / std_spread
    z_score = z_score.fillna(0.0)
    
    # Generate Trading Signals
    signals = pd.Series(0, index=z_score.index)
    signals[z_score > z_entry] = -1 # Short Spread (Short Asset A, Long Asset B)
    signals[z_score < -z_entry] = 1  # Long Spread (Long Asset A, Short Asset B)
    
    # Current active signal
    latest_z = float(z_score.iloc[-1]) if len(z_score) > 0 else 0.0
    if latest_z > z_entry:
        latest_signal_desc = f"SHORT {asset_a} / LONG {asset_b} (Overpriced Spread)"
    elif latest_z < -z_entry:
        latest_signal_desc = f"LONG {asset_a} / SHORT {asset_b} (Underpriced Spread)"
    else:
        latest_signal_desc = "NEUTRAL / HOLD (Spread in Fair Value Zone)"
        
    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "beta_ratio": float(beta_ratio),
        "spread": spread,
        "z_score": z_score,
        "signals": signals,
        "latest_z_score": latest_z,
        "latest_signal_desc": latest_signal_desc,
        "vecm_fit": vecm_res
    }
