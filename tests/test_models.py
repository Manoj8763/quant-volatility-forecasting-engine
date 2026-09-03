import pytest
import pandas as pd
from src.data_loader import load_market_data
from src.garch_models import fit_garch_volatility_models
from src.risk_metrics import calculate_var_and_expected_shortfall

def test_garch_models_and_risk_metrics():
    df, _ = load_market_data(ticker="JPM", mode="csv")
    garch_results = fit_garch_volatility_models(df['Return_Clean'], horizon=5)
    
    assert len(garch_results['garch_forecast_vol']) == 5
    assert len(garch_results['egarch_forecast_vol']) == 5
    
    risk_res = calculate_var_and_expected_shortfall(df['Return_Clean'], garch_results['egarch_cond_vol'], portfolio_value=1000000.0)
    assert 0.99 in risk_res['metrics_by_confidence']
    assert risk_res['metrics_by_confidence'][0.99]['var_dollar'] > 0
    assert risk_res['metrics_by_confidence'][0.99]['es_dollar'] > risk_res['metrics_by_confidence'][0.99]['var_dollar']
