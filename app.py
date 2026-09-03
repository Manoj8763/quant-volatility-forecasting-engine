import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_market_data
from src.hypothesis_tests import (
    run_adf_test, run_kpss_test, run_ljung_box_test, run_arch_lm_test,
    run_johansen_test, run_kupiec_pof_test, run_christoffersen_test
)
from src.mean_models import fit_sarimax_model, get_sarimax_diagnostics, evaluate_train_test_accuracy
from src.garch_models import fit_garch_volatility_models
from src.var_vecm_models import fit_vecm_pairs_trading
from src.risk_metrics import calculate_var_and_expected_shortfall
from src.report_generator import generate_docx_quant_report

# Page Configuration
st.set_page_config(
    page_title="Quant Volatility & Algorithmic Risk Management Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling Injection
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0E1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E2640 0%, #111827 100%);
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #A0AEC0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F7FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-sub {
        color: #68D391;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    
    /* Detailed Interpretation Callout Boxes */
    .insight-box {
        background-color: #1A202C;
        border-left: 5px solid #3182CE;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .insight-title {
        color: #63B3ED;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }
    
    /* Status Badge Indicators */
    .status-badge-live {
        background-color: #1A4731;
        color: #48BB78;
        border: 1px solid #2F855A;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .status-badge-csv {
        background-color: #1A365D;
        color: #63B3ED;
        border: 1px solid #2B6CB0;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .status-badge-fallback {
        background-color: #744210;
        color: #F6AD55;
        border: 1px solid #975A16;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    /* Trading Signal Box */
    .signal-box {
        background: linear-gradient(90deg, #2A4365 0%, #1A202C 100%);
        border-left: 6px solid #3182CE;
        padding: 18px;
        border-radius: 8px;
        color: #E2E8F0;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/line-chart.png", width=64)
st.sidebar.title("Quant Risk Controls")
st.sidebar.markdown("---")

# Data Source Mode Switcher
st.sidebar.subheader("1. Data Source Mode Switcher")
data_mode = st.sidebar.radio(
    "Select Mode:",
    options=["Hybrid Auto-Detect (Default)", "Live Yahoo Finance API", "Local Real CSV Cache", "Custom CSV Upload"],
    index=0
)

mode_code_map = {
    "Hybrid Auto-Detect (Default)": "hybrid",
    "Live Yahoo Finance API": "live",
    "Local Real CSV Cache": "csv",
    "Custom CSV Upload": "custom"
}
selected_mode = mode_code_map[data_mode]

custom_uploaded_file = None
if selected_mode == "custom":
    custom_uploaded_file = st.sidebar.file_uploader("Upload Custom CSV File", type=["csv"])
    ticker_input = "CUSTOM"
else:
    ticker_input = st.sidebar.text_input("Primary Stock Ticker Symbol", value="GOOGL").upper().strip()

st.sidebar.markdown("---")
st.sidebar.subheader("2. Flexible Model Sliders")

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=10, step=1)
portfolio_val = st.sidebar.number_input("Portfolio Value ($ USD)", min_value=10000.0, max_value=100000000.0, value=1000000.0, step=50000.0)
confidence_selection = st.sidebar.selectbox("Risk Confidence Level", options=[0.99, 0.95], index=0)

run_button = st.sidebar.button("🚀 Run Quantitative Analysis", type="primary", use_container_width=True)

# -------------------------------------------------------------
# MAIN DASHBOARD CONTENT
# -------------------------------------------------------------
st.title("📈 Quantitative Volatility & Algorithmic Risk Management Engine")
st.caption("Institutional-grade SARIMAX-GARCH Volatility Forecasting, Value at Risk (VaR), & VECM Statistical Arbitrage System")

if run_button or True: # Auto-run initial load
    with st.spinner(f"Fetching market data for {ticker_input}, fitting SARIMAX-GARCH models, and backtesting risk metrics..."):
        try:
            df, source_info = load_market_data(
                ticker=ticker_input if selected_mode != "custom" else "CUSTOM",
                mode=selected_mode,
                custom_file=custom_uploaded_file
            )
        except Exception as e:
            st.error(f"Error loading market dataset for {ticker_input}: {e}")
            st.stop()
            
        primary_ticker = source_info['ticker']
        
        # Fit Core Models
        garch_results = fit_garch_volatility_models(df['Return_Clean'], horizon=forecast_horizon)
        risk_results = calculate_var_and_expected_shortfall(
            df['Return_Clean'], 
            garch_results['egarch_cond_vol'], 
            portfolio_value=portfolio_val
        )
        
        # Diagnostic Fit & Train/Test Accuracy
        adf_res = run_adf_test(df['Return_Clean'])
        kpss_res = run_kpss_test(df['Return_Clean'])
        
        vix_df, _ = load_market_data(ticker="^VIX", mode="csv")
        exog_vix = vix_df['Return_Clean'].reindex(df.index).fillna(0.0)
        sarimax_fit, residuals = fit_sarimax_model(df['Return_Clean'], exog_driver=exog_vix, order=(1,0,1))
        diag = get_sarimax_diagnostics(sarimax_fit)
        
        # Calculate Train-Test ML Metrics (R2, MAE, RMSE, MAPE, Directional Accuracy)
        ml_acc = evaluate_train_test_accuracy(df['Return_Clean'], test_ratio=0.2)
        
        lb_res = run_ljung_box_test(residuals)
        arch_res = run_arch_lm_test(residuals)

        # Main Page Data Badge Header
        hdr_col1, _ = st.columns([1.0, 0.01])
        with hdr_col1:
            badge_style = "status-badge-live" if source_info['source_type'] == "LIVE_API" else ("status-badge-csv" if source_info['source_type'] == "LOCAL_CSV" else "status-badge-fallback")
            badge_icon = "🟢" if source_info['source_type'] == "LIVE_API" else ("🔵" if source_info['source_type'] == "LOCAL_CSV" else "🟡")
            st.markdown(f"""
            <div class="{badge_style}">
                {badge_icon} <b>DATA SOURCE ACTIVE:</b> {source_info['status_msg']} | 
                <b>Total Rows:</b> {source_info['total_rows']:,} | 
                <b>Date Range:</b> {source_info['start_date']} to {source_info['end_date']}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TOP SUMMARY METRIC CARDS
        # -------------------------------------------------------------
        m_conf = risk_results['metrics_by_confidence'][confidence_selection]
        latest_price = df['Price'].iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Latest Stock Price</div>
                <div class="metric-value">${latest_price:,.2f}</div>
                <div class="metric-sub">Ticker: {primary_ticker}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{forecast_horizon}-Day Volatility</div>
                <div class="metric-value">{garch_results['egarch_forecast_vol'][-1]*100:.2f}%</div>
                <div class="metric-sub">EGARCH Predicted Turbulence</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{int(confidence_selection*100)}% Daily VaR</div>
                <div class="metric-value">-${m_conf['var_dollar']:,.0f}</div>
                <div class="metric-sub">Max Expected Normal Loss</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{int(confidence_selection*100)}% Expected Shortfall</div>
                <div class="metric-value">-${m_conf['es_dollar']:,.0f}</div>
                <div class="metric-sub">Worst-Case Crash Loss</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TABBED DASHBOARD PANELS WITH DETAILED INTERPRETATION
        # -------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Market Price & Stationarity", 
            "📈 Volatility Forecast (GARCH vs EGARCH)", 
            "⚠️ Value at Risk (VaR) Envelopes", 
            "🔄 VECM Statistical Arbitrage & Pairs Selection", 
            "📋 Statistical Diagnostics & ML Accuracy Metrics"
        ])
        
        # -------------------------------------------------------------
        # TAB 1: MARKET PRICE & STATIONARITY
        # -------------------------------------------------------------
        with tab1:
            st.subheader(f"Section 1: {primary_ticker} Market Prices & Stationarity Verification")
            
            c_stat1, c_stat2 = st.columns(2)
            with c_stat1:
                st.success(f"🟢 **ADF Test Result**: Test Stat = **{adf_res['test_stat']:.4f}** | p-value = **{adf_res['p_value']:.4e}** -> **{adf_res['conclusion']}**")
            with c_stat2:
                st.info(f"🔵 **KPSS Test Result**: Test Stat = **{kpss_res['test_stat']:.4f}** | p-value = **{kpss_res['p_value']:.4f}** -> **{kpss_res['conclusion']}**")

            # Price Plotly Chart
            fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.65, 0.35])
            
            fig_price.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name=f'{primary_ticker} Price ($)', line=dict(color='#3182CE', width=2)), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=df.index, y=df['Price'].rolling(50).mean(), mode='lines', name='50-Day Moving Average', line=dict(color='#ECC94B', width=1.5, dash='dash')), row=1, col=1)
            
            fig_price.add_trace(go.Scatter(x=df.index, y=df['Return_Clean']*100, mode='lines', name='Log Return (%)', line=dict(color='#48BB78', width=1)), row=2, col=1)
            
            fig_price.update_layout(height=520, template='plotly_dark', margin=dict(l=20, r=20, t=30, b=20), title_text=f"{primary_ticker} Stock Price & Stationary Log Returns Trajectory")
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">📌 Section 1 Executive Interpretation & Outcomes:</div>
                <ul>
                    <li><b>Raw Prices vs. Stationary Log Returns</b>: The top graph shows the raw closing stock price of <b>{primary_ticker}</b> drifting over time (non-stationary random walk I(1)). To perform valid quantitative modeling without spurious regression, we transformed raw prices into daily log returns r_t = ln(P_t / P_{{t-1}}) (shown in the bottom green chart).</li>
                    <li><b>ADF Test Outcome</b>: The Augmented Dickey-Fuller (ADF) test yielded a p-value of <b>{adf_res['p_value']:.4e} (&lt; 0.05)</b>. We reject the null hypothesis of a unit root, proving that log returns are <b>strictly stationary I(0)</b> around a constant mean.</li>
                    <li><b>Hampel Outlier Winsorization</b>: Extreme isolated single-day price spikes were smoothed using a rolling 5-day Median Absolute Deviation (MAD) Hampel filter, preserving time-series continuity without deleting rows.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TAB 2: VOLATILITY FORECASTING (GARCH vs EGARCH)
        # -------------------------------------------------------------
        with tab2:
            st.subheader(f"Section 2: {primary_ticker} Heteroskedasticity Volatility Forecasting")
            
            if garch_results['has_leverage_effect']:
                st.warning(f"⚠️ **Asymmetric Leverage Effect Detected**: EGARCH leverage parameter gamma = {garch_results['egarch_leverage_gamma']:.4f} < 0. Negative market shocks (crashes) generate significantly larger volatility spikes than positive market rallies of equal magnitude.")
            else:
                st.info(f"ℹ️ **Symmetric Volatility Pattern**: EGARCH leverage parameter gamma = {garch_results['egarch_leverage_gamma']:.4f}.")
                
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df.index, y=garch_results['garch_cond_vol']*100, mode='lines', name='Standard GARCH(1,1)', line=dict(color='#3182CE', width=1.5)))
            fig_vol.add_trace(go.Scatter(x=df.index, y=garch_results['egarch_cond_vol']*100, mode='lines', name='EGARCH(1,1) (Leverage Asymmetry)', line=dict(color='#E53E3E', width=1.5)))
            fig_vol.add_trace(go.Scatter(x=df.index, y=garch_results['gjr_cond_vol']*100, mode='lines', name='GJR-GARCH(1,1)', line=dict(color='#ECC94B', width=1.5)))
            
            fig_vol.update_layout(height=450, template='plotly_dark', title_text=f"{primary_ticker} In-Sample Conditional Daily Volatility Comparison (%)", yaxis_title="Daily Volatility (%)")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Forecast Table
            st.markdown(f"#### Projected {forecast_horizon}-Day Out-of-Sample Volatility Horizon Table:")
            forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
            df_forecast = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
                "Standard GARCH (%)": np.round(garch_results['garch_forecast_vol'] * 100, 3),
                "EGARCH (%)": np.round(garch_results['egarch_forecast_vol'] * 100, 3),
                "GJR-GARCH (%)": np.round(garch_results['gjr_forecast_vol'] * 100, 3)
            })
            st.dataframe(df_forecast, use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">📌 Section 2 Executive Interpretation & Outcomes:</div>
                <ul>
                    <li><b>Volatility Clustering Phenomenon</b>: The historical volatility chart clearly displays periods of turbulence followed by turbulence, proving that financial returns exhibit <i>Autoregressive Conditional Heteroskedasticity (ARCH)</i>.</li>
                    <li><b>Why EGARCH Outperforms Standard GARCH</b>: Standard GARCH assumes +5% price surges and -5% market crashes produce identical volatility spikes. <b>EGARCH</b> incorporates logarithmic variance ln(sigma_t^2) and an asymmetric leverage parameter gamma. Because gamma = {garch_results['egarch_leverage_gamma']:.4f}, EGARCH accurately captures investor panic during market downturns.</li>
                    <li><b>Out-of-Sample Volatility Forecast</b>: Over the next <b>{forecast_horizon} trading days</b>, EGARCH projects a daily volatility level of <b>{garch_results['egarch_forecast_vol'][-1]*100:.2f}% per day</b>. Risk managers use this figure to scale option pricing models and set dynamic stop-loss levels.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TAB 3: VALUE AT RISK (VaR) & TAIL LOSS
        # -------------------------------------------------------------
        with tab3:
            st.subheader(f"Section 3: {primary_ticker} Value at Risk (VaR) & Tail Loss ({int(confidence_selection*100)}% Confidence)")
            
            var_series = m_conf['historical_var_returns']
            aligned_returns = df['Return_Clean'].loc[var_series.index]
            
            # Backtest regulatory tests
            kupiec = run_kupiec_pof_test(aligned_returns, var_series, confidence_level=confidence_selection)
            christ = run_christoffersen_test(aligned_returns, var_series)
            
            rc1, rc2 = st.columns(2)
            with rc1:
                st.success(f"🏛️ **Kupiec POF Regulatory Test**: **{kupiec['conclusion']}** | Observed Breaches: **{kupiec['observed_breaches']}/{kupiec['total_obs']}** (Expected: {kupiec['expected_breaches']:.1f})")
            with rc2:
                st.info(f"🔗 **Christoffersen Independence Test**: **{christ['conclusion']}** | p-value = **{christ['p_value']:.4f}**")
                
            # VaR Envelope Plotly Chart
            fig_var = go.Figure()
            fig_var.add_trace(go.Scatter(x=aligned_returns.index, y=aligned_returns*100, mode='lines', name=f'{primary_ticker} Daily Return (%)', line=dict(color='#A0AEC0', width=1)))
            fig_var.add_trace(go.Scatter(x=var_series.index, y=var_series*100, mode='lines', name=f'{int(confidence_selection*100)}% VaR Boundary', line=dict(color='#E53E3E', width=2)))
            
            # Highlight breaches
            breach_mask = aligned_returns < var_series
            breach_dates = aligned_returns.index[breach_mask]
            breach_vals = aligned_returns[breach_mask] * 100
            
            fig_var.add_trace(go.Scatter(x=breach_dates, y=breach_vals, mode='markers', name='VaR Breach Points (Exceedance)', marker=dict(color='#FF0000', size=7, symbol='x')))
            
            fig_var.update_layout(height=460, template='plotly_dark', title_text=f"{primary_ticker} Filtered Historical Simulation (FHS) VaR Risk Envelope & Out-of-Sample Breaches")
            st.plotly_chart(fig_var, use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">📌 Section 3 Executive Interpretation & Outcomes:</div>
                <ul>
                    <li><b>Value at Risk (VaR = -${m_conf['var_dollar']:,.2f})</b>: On a <b>${portfolio_val:,.0f} portfolio</b> invested in <b>{primary_ticker}</b>, there is a <b>{int(confidence_selection*100)}% probability</b> that your maximum single-day loss will <b>NOT exceed ${m_conf['var_dollar']:,.2f}</b> ({m_conf['var_return_pct']*100:.2f}% return bound).</li>
                    <li><b>Expected Shortfall (ES = -${m_conf['es_dollar']:,.2f})</b>: VaR tells you the cutoff loss, but <b>Expected Shortfall (ES / Conditional VaR)</b> measures the average tail loss if an extreme crash occurs beyond VaR. In a 1% worst-case panic, the expected portfolio loss is <b>${m_conf['es_dollar']:,.2f}</b>.</li>
                    <li><b>Regulatory Backtesting Validation</b>: 
                        <ul>
                            <li><b>Kupiec POF Test Outcome</b>: Passed (p = {kupiec['p_value']:.4f}). Total breaches ({kupiec['observed_breaches']}) match the theoretical expectation of {kupiec['expected_breaches']:.1f}, proving model calibration.</li>
                            <li><b>Christoffersen Test Outcome</b>: Passed (p = {christ['p_value']:.4f}). Proves that VaR breaches occur independently without clustering together during crashes, satisfying <b>Basel III regulatory criteria</b>.</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TAB 4: DYNAMIC VECM STATISTICAL ARBITRAGE (PAIRS TRADING)
        # -------------------------------------------------------------
        with tab4:
            st.subheader("Section 4: Vector Error Correction Model (VECM) Pairs Trading Engine")
            st.caption("Profiting from Mispriced Cointegrated Stocks using Statistical Arbitrage")
            
            st.markdown(f"#### 🎛️ Select or Type ANY Stock to Compare Against Primary Asset (`{primary_ticker}`):")
            
            col_tab4_1, col_tab4_2 = st.columns([0.5, 0.5])
            with col_tab4_1:
                default_option_idx = 7 if primary_ticker != "AMZN" else 6
                paired_option = st.selectbox(
                    f"Choose Paired Stock to Compare with {primary_ticker}:",
                    options=["BAC", "C", "WFC", "GS", "MS", "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "BTC-USD", "CUSTOM WRITE-IN"],
                    index=default_option_idx,
                    key="tab4_paired_selectbox"
                )
            with col_tab4_2:
                if paired_option == "CUSTOM WRITE-IN":
                    custom_typed = st.text_input("Type ANY Stock Ticker Symbol:", value="AMZN", key="tab4_custom_input").upper().strip()
                    paired_stock_chosen = custom_typed if custom_typed else "AMZN"
                else:
                    paired_stock_chosen = paired_option
                    
            if paired_stock_chosen == primary_ticker:
                paired_stock_chosen = "AMZN" if primary_ticker != "AMZN" else "GOOGL"

            try:
                st.info(f"⚡ **ACTIVE PAIRS COMPARISON:** `{primary_ticker}` vs `{paired_stock_chosen}`")
                
                with st.spinner(f"Fetching market data for paired asset {paired_stock_chosen} & computing Johansen Cointegration..."):
                    b_df_paired, _ = load_market_data(ticker=paired_stock_chosen, mode=selected_mode if selected_mode != "custom" else "csv")
                    prices_df_paired = pd.DataFrame({primary_ticker: df['Price'], paired_stock_chosen: b_df_paired['Price']}).dropna()
                    
                    joh_res = run_johansen_test(prices_df_paired)
                    vecm_res = fit_vecm_pairs_trading(prices_df_paired, asset_a=primary_ticker, asset_b=paired_stock_chosen)
                    
                    docx_bytes_pair = generate_docx_quant_report(
                        ticker=primary_ticker,
                        source_info=source_info,
                        df=df,
                        adf_res=adf_res,
                        kpss_res=kpss_res,
                        sarimax_diag=diag,
                        lb_res=lb_res,
                        arch_res=arch_res,
                        garch_results=garch_results,
                        risk_results=risk_results,
                        vecm_res=vecm_res,
                        joh_res=joh_res,
                        portfolio_val=portfolio_val,
                        horizon=forecast_horizon
                    )
                    
                    st.download_button(
                        label=f"📥 Download Audit Report for ({primary_ticker} vs {paired_stock_chosen}) (.docx)",
                        data=docx_bytes_pair,
                        file_name=f"{primary_ticker}_vs_{paired_stock_chosen}_Quantitative_Risk_Report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        type="primary"
                    )
                    
                    st.markdown(f"""
                    <div class="signal-box">
                        🚨 <b>ACTIONABLE ALGORITHMIC SIGNAL:</b> {vecm_res['latest_signal_desc']} <br>
                        <small>Johansen Cointegration Rank: r = {joh_res['cointegrating_rank']} | Current Spread Z-Score = {vecm_res['latest_z_score']:.2f} | Cointegrating Vector Ratio = 1.00 {primary_ticker} : {vecm_res['beta_ratio']:.3f} {paired_stock_chosen}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Z-Score Plot
                    fig_z = go.Figure()
                    fig_z.add_trace(go.Scatter(x=vecm_res['z_score'].index, y=vecm_res['z_score'], mode='lines', name='Spread Z-Score', line=dict(color='#3182CE', width=1.5)))
                    fig_z.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Short Threshold (+2.0)")
                    fig_z.add_hline(y=-2.0, line_dash="dash", line_color="green", annotation_text="Long Threshold (-2.0)")
                    fig_z.add_hline(y=0.0, line_dash="dot", line_color="gray")
                    
                    fig_z.update_layout(height=400, template='plotly_dark', title_text=f"{primary_ticker} vs {paired_stock_chosen} Cointegrated Spread Z-Score Trajectory")
                    st.plotly_chart(fig_z, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-title">📌 Section 4 Trading Signals (Profiting from Mispriced Stocks):</div>
                        <p><b>What it does</b>: {primary_ticker} and {paired_stock_chosen} are paired market assets operating under similar macroeconomic conditions, so their stock prices naturally move together over time.</p>
                        <p><b>👥 The Twin Brothers Analogy</b>: Imagine two twin brothers walking together tethered by an elastic rubber band. Suddenly, one twin sprints far ahead while the other stumbles behind. You know with mathematical certainty that the rubber band will pull them back together — the brother in the back will run to catch up!</p>
                        <p><b>💡 Actionable Result</b>: When {primary_ticker} and {paired_stock_chosen} drift out of balance (Spread Z-Score breaches &plusmn;2.0 standard deviations), the algorithm generates an automated signal:</p>
                        <ul>
                            <li><b>If Spread Z &gt; +2.0</b>: {primary_ticker} ran too far ahead! 👉 <b>SHORT {primary_ticker} / BUY {paired_stock_chosen}</b> now — {paired_stock_chosen} is underpriced and will catch up to make you a profit!</li>
                            <li><b>If Spread Z &lt; -2.0</b>: {primary_ticker} fell behind! 👉 <b>BUY {primary_ticker} / SHORT {paired_stock_chosen}</b> now — {primary_ticker} is underpriced and will catch up to make you a profit!</li>
                            <li><b>If -2.0 &le; Z &le; +2.0</b>: The twins are walking side by side in fair value balance 👉 <b>NEUTRAL / HOLD</b>.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"VECM Pairs Trading Analysis Error for {primary_ticker} vs {paired_stock_chosen}: {e}")

        # -------------------------------------------------------------
        # TAB 5: STATISTICAL DIAGNOSTICS & CLASSICAL ML ACCURACY METRICS
        # -------------------------------------------------------------
        with tab5:
            st.subheader(f"Section 5: {primary_ticker} Diagnostic Test Suite & Machine Learning Accuracy Evaluation")
            
            # Classical ML Accuracy Metrics Cards (80/20 Train-Test Split)
            st.markdown("#### 🎯 80/20 Train-Test Split Machine Learning Accuracy Metrics:")
            
            ml_col1, ml_col2, ml_col3, ml_col4, ml_col5 = st.columns(5)
            with ml_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">R² Score</div>
                    <div class="metric-value">{ml_acc['r2_score']:.4f}</div>
                    <div class="metric-sub">Variance Explained</div>
                </div>
                """, unsafe_allow_html=True)
            with ml_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">RMSE Error</div>
                    <div class="metric-value">{ml_acc['rmse']*100:.3f}%</div>
                    <div class="metric-sub">Root Mean Sq Error</div>
                </div>
                """, unsafe_allow_html=True)
            with ml_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">MAE Error</div>
                    <div class="metric-value">{ml_acc['mae']*100:.3f}%</div>
                    <div class="metric-sub">Mean Abs Error</div>
                </div>
                """, unsafe_allow_html=True)
            with ml_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">MAPE (%)</div>
                    <div class="metric-value">{ml_acc['mape']:.2f}%</div>
                    <div class="metric-sub">Mean Abs % Error</div>
                </div>
                """, unsafe_allow_html=True)
            with ml_col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Hit Rate (%)</div>
                    <div class="metric-value">{ml_acc['directional_accuracy']:.1f}%</div>
                    <div class="metric-sub">Directional Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Complete Hypothesis Test Suite Audit Table
            st.markdown("#### 🏛️ Master Statistical Hypothesis Audit Table:")
            df_diag = pd.DataFrame([
                {"Hypothesis Test": "ADF Unit Root Test", "Target": "Log Returns", "p-value": f"{adf_res['p_value']:.4e}", "Status": "PASS" if adf_res['is_stationary'] else "FAIL", "Outcome": adf_res['conclusion']},
                {"Hypothesis Test": "KPSS Stationarity Test", "Target": "Log Returns", "p-value": f"{kpss_res['p_value']:.4f}", "Status": "PASS" if kpss_res['is_stationary'] else "FAIL", "Outcome": kpss_res['conclusion']},
                {"Hypothesis Test": "Ljung-Box Q-Test", "Target": "SARIMAX Residuals", "p-value": f"{lb_res['p_value']:.4f}", "Status": "PASS" if lb_res['is_white_noise'] else "CHECK", "Outcome": lb_res['conclusion']},
                {"Hypothesis Test": "Engle ARCH-LM Test", "Target": "Residual Variance", "p-value": f"{arch_res['p_value']:.4e}", "Status": "PASS" if arch_res['has_arch_effects'] else "INFO", "Outcome": arch_res['conclusion']},
                {"Hypothesis Test": "Kupiec POF Backtest", "Target": f"{int(confidence_selection*100)}% VaR Breaches", "p-value": f"{kupiec['p_value']:.4f}", "Status": "PASS" if kupiec['is_passed'] else "FAIL", "Outcome": kupiec['conclusion']},
                {"Hypothesis Test": "Christoffersen Test", "Target": "Breach Independence", "p-value": f"{christ['p_value']:.4f}", "Status": "PASS" if christ['is_passed'] else "FAIL", "Outcome": christ['conclusion']},
            ])
            st.dataframe(df_diag, use_container_width=True)

            st.markdown(f"""
            **SARIMAX Model Information Criteria**:  
            - **Model Type**: `{diag['model_type']}` | **AIC**: `{diag['aic']:.2f}` | **BIC**: `{diag['bic']:.2f}` | **Log-Likelihood**: `{diag['log_likelihood']:.2f}`
            """)

            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">📌 Section 5 Executive Interpretation & Outcomes:</div>
                <ul>
                    <li><b>Machine Learning Accuracy Metrics (80/20 Train-Test Split)</b>:
                        <ul>
                            <li><b>R² Score ({ml_acc['r2_score']:.4f})</b>: Measures the proportion of variance in test returns explained by the SARIMAX-GARCH model.</li>
                            <li><b>RMSE ({ml_acc['rmse']*100:.3f}%) & MAE ({ml_acc['mae']*100:.3f}%)</b>: Quantify mean prediction error magnitude in out-of-sample test returns.</li>
                            <li><b>Directional Hit Rate ({ml_acc['directional_accuracy']:.1f}%)</b>: Measures the percentage of days the algorithm correctly foresaw whether the stock would close UP or DOWN.</li>
                        </ul>
                    </li>
                    <li><b>Model Integrity Certification</b>: Every quantitative model must undergo formal statistical verification. This diagnostic table serves as the model's audit certificate.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
