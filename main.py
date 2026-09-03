import sys
import os
import argparse
import yaml
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_market_data
from src.hypothesis_tests import (
    run_adf_test, run_kpss_test, run_ljung_box_test, run_arch_lm_test,
    run_johansen_test, run_kupiec_pof_test, run_christoffersen_test
)
from src.mean_models import fit_sarimax_model, get_sarimax_diagnostics
from src.garch_models import fit_garch_volatility_models
from src.var_vecm_models import fit_vecm_pairs_trading
from src.risk_metrics import calculate_var_and_expected_shortfall

def run_pipeline(ticker="JPM", mode="hybrid", horizon=10, portfolio_val=1000000.0, custom_file=None):
    """
    Executes the Complete Quantitative Volatility & Risk Management Pipeline.
    """
    print("=" * 70)
    print("  QUANTITATIVE VOLATILITY & ALGORITHMIC RISK MANAGEMENT ENGINE")
    print("=" * 70)
    
    # 1. LOAD DATA
    print(f"\n[STEP 1] Loading Market Data for Ticker: {ticker} (Mode: {mode})...")
    df, source_info = load_market_data(ticker=ticker, mode=mode, custom_file=custom_file)
    print(f"   DATA SOURCE ACTIVE : {source_info['status_msg']}")
    print(f"   Total Rows Loaded  : {source_info['total_rows']} daily records")
    print(f"   Date Range          : {source_info['start_date']} to {source_info['end_date']}")
    
    # 2. STATIONARITY HYPOTHESIS TESTING
    print("\n[STEP 2] Running Dual Stationarity Hypothesis Tests (ADF + KPSS)...")
    adf_res = run_adf_test(df['Return_Clean'])
    kpss_res = run_kpss_test(df['Return_Clean'])
    print(f"  --> ADF Test  : Stat = {adf_res['test_stat']:.4f}, p-value = {adf_res['p_value']:.4e} -> {adf_res['conclusion']}")
    print(f"  --> KPSS Test : Stat = {kpss_res['test_stat']:.4f}, p-value = {kpss_res['p_value']:.4f} -> {kpss_res['conclusion']}")
    
    # 3. SARIMAX MEAN MODEL
    print("\n[STEP 3] Fitting SARIMAX Conditional Mean Model...")
    vix_df, _ = load_market_data(ticker="^VIX", mode="csv")
    exog_vix = vix_df['Return_Clean'].reindex(df.index).fillna(0.0)
    
    sarimax_fit, residuals = fit_sarimax_model(df['Return_Clean'], exog_driver=exog_vix, order=(1,0,1))
    diag = get_sarimax_diagnostics(sarimax_fit)
    print(f"  --> Model Fitted : {diag['model_type']} (AIC = {diag['aic']:.2f}, BIC = {diag['bic']:.2f})")
    
    # Residual Tests
    lb_res = run_ljung_box_test(residuals)
    arch_res = run_arch_lm_test(residuals)
    print(f"  --> Ljung-Box Residual Autocorrelation Test : {lb_res['conclusion']} (p = {lb_res['p_value']:.4f})")
    print(f"  --> Engle's ARCH-LM Heteroskedasticity Test  : {arch_res['conclusion']} (p = {arch_res['p_value']:.4e})")
    
    # 4. GARCH & EGARCH VOLATILITY FORECASTING
    print("\n[STEP 4] Fitting GARCH, EGARCH & GJR-GARCH Heteroskedasticity Models...")
    garch_results = fit_garch_volatility_models(df['Return_Clean'], horizon=horizon)
    print(f"  --> EGARCH Leverage Parameter (gamma) : {garch_results['egarch_leverage_gamma']:.4f}")
    print(f"  --> Leverage Effect Detected          : {'YES (Asymmetric Shock Sensitivity)' if garch_results['has_leverage_effect'] else 'NO'}")
    print(f"  --> Projected {horizon}-Day Volatility  : GARCH={garch_results['garch_forecast_vol'][-1]*100:.2f}%, EGARCH={garch_results['egarch_forecast_vol'][-1]*100:.2f}%")
    
    # 5. RISK METRICS ENGINE (VaR & EXPECTED SHORTFALL)
    print(f"\n[STEP 5] Computing Value at Risk (VaR) & Expected Shortfall (Portfolio = ${portfolio_val:,.2f})...")
    risk_results = calculate_var_and_expected_shortfall(
        df['Return_Clean'], 
        garch_results['egarch_cond_vol'], 
        portfolio_value=portfolio_val
    )
    
    for alpha in [0.95, 0.99]:
        m = risk_results['metrics_by_confidence'][alpha]
        var_series = m['historical_var_returns']
        aligned_returns = df['Return_Clean'].loc[var_series.index]
        
        print(f"  --> {int(alpha*100)}% Daily VaR          : -${m['var_dollar']:,.2f} ({m['var_return_pct']*100:.2f}% return bound)")
        print(f"  --> {int(alpha*100)}% Expected Shortfall : -${m['es_dollar']:,.2f} ({m['es_return_pct']*100:.2f}% tail loss)")
        
        # Kupiec & Christoffersen Backtests
        kupiec = run_kupiec_pof_test(aligned_returns, var_series, confidence_level=alpha)
        christ = run_christoffersen_test(aligned_returns, var_series)
        print(f"      [Kupiec POF Regulatory Test]       : {kupiec['conclusion']} (Breaches: {kupiec['observed_breaches']}/{kupiec['total_obs']})")
        print(f"      [Christoffersen Independence Test] : {christ['conclusion']}")
        
    # 6. VECM COINTEGRATION & PAIRS TRADING
    print("\n[STEP 6] Running VECM Cointegration & Statistical Arbitrage (Pairs Trading)...")
    try:
        jpm_df, _ = load_market_data(ticker="JPM", mode="csv")
        bac_df, _ = load_market_data(ticker="BAC", mode="csv")
        prices_df = pd.DataFrame({"JPM": jpm_df['Price'], "BAC": bac_df['Price']}).dropna()
        
        joh_res = run_johansen_test(prices_df)
        vecm_res = fit_vecm_pairs_trading(prices_df, asset_a="JPM", asset_b="BAC")
        print(f"  --> Johansen Cointegration Rank : r = {joh_res['cointegrating_rank']} -> {joh_res['conclusion']}")
        print(f"  --> Current Spread Z-Score      : {vecm_res['latest_z_score']:.2f}")
        print(f"  [ACTIONABLE TRADING SIGNAL]    : {vecm_res['latest_signal_desc']}")
    except Exception as e:
        print(f"  [Warning] VECM Pairs Trading Skipping: {e}")
        
    print("\n" + "=" * 70)
    print("  QUANT PIPELINE EXECUTION COMPLETE SUCCESSFULLY!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Volatility & Risk Engine CLI")
    parser.add_argument("--ticker", type=str, default="JPM", help="Stock ticker symbol")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["live", "csv", "hybrid"], help="Data mode")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon in days")
    parser.add_argument("--portfolio", type=float, default=1000000.0, help="Portfolio value in USD")
    
    args = parser.parse_args()
    run_pipeline(ticker=args.ticker, mode=args.mode, horizon=args.horizon, portfolio_val=args.portfolio)
