import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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

@st.cache_data(ttl=3600)
def get_usdinr_exchange_rate():
    try:
        t = yf.Ticker("USDINR=X")
        h = t.history(period="1d")
        if not h.empty:
            return float(h['Close'].iloc[-1])
    except Exception:
        pass
    return 83.50

# -------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/line-chart.png", width=64)
st.sidebar.title("Quant Risk Controls")
st.sidebar.markdown("---")

# 1. Data Source Mode Switcher
st.sidebar.subheader("1. Data Source Switcher")
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

# 2. Currency Converter & Formatting Controls
st.sidebar.subheader("2. Currency & FX Converter")
currency_choice = st.sidebar.selectbox(
    "Display Currency Mode:",
    options=["Auto-Detect (Native ₹ / $)", "INR (₹) Indian Rupee", "USD ($) US Dollar"],
    index=0
)

usdinr_rate = get_usdinr_exchange_rate()
st.sidebar.caption(f"⚡ Live Exchange Rate: **1 USD = ₹{usdinr_rate:.2f} INR**")

st.sidebar.markdown("---")
st.sidebar.subheader("3. Flexible Model Sliders")

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=10, step=1)
portfolio_val_input = st.sidebar.number_input("Portfolio Value (Capital)", min_value=10000.0, max_value=1000000000.0, value=1000000.0, step=50000.0)
confidence_selection = st.sidebar.selectbox("Risk Confidence Level", options=[0.99, 0.95], index=0)

run_button = st.sidebar.button("🚀 Run Quantitative Analysis", type="primary")

# -------------------------------------------------------------
# MAIN DASHBOARD CONTENT
# -------------------------------------------------------------
st.title("📈 Quantitative Volatility & Algorithmic Risk Management Engine")
st.caption("Institutional-grade SARIMAX-GARCH Volatility Forecasting, Value at Risk (VaR), & VECM Statistical Arbitrage System")

if run_button or True:
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
        
        # Determine Native Currency
        is_indian_asset = (
            primary_ticker.endswith(".NS") or 
            primary_ticker.endswith(".BO") or 
            primary_ticker in ["^NSEI", "^NSEBANK", "^BSESN", "^CRSLDX"]
        )
        native_curr = "INR" if is_indian_asset else "USD"
        
        # Determine Active Display Currency & Conversion Factor
        if currency_choice == "INR (₹) Indian Rupee":
            active_curr = "INR"
        elif currency_choice == "USD ($) US Dollar":
            active_curr = "USD"
        else:
            active_curr = native_curr
            
        if native_curr == "USD" and active_curr == "INR":
            fx_multiplier = usdinr_rate
            curr_symbol = "₹"
        elif native_curr == "INR" and active_curr == "USD":
            fx_multiplier = 1.0 / usdinr_rate
            curr_symbol = "$"
        else:
            fx_multiplier = 1.0
            curr_symbol = "₹" if active_curr == "INR" else "$"

        # Fit Core Models
        garch_results = fit_garch_volatility_models(df['Return_Clean'], horizon=forecast_horizon)
        risk_results = calculate_var_and_expected_shortfall(
            df['Return_Clean'], 
            garch_results['egarch_cond_vol'], 
            portfolio_value=portfolio_val_input
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
                <b>Date Range:</b> {source_info['start_date']} to {source_info['end_date']} | 
                <b>Display Currency:</b> {active_curr} ({curr_symbol})
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TOP SUMMARY METRIC CARDS WITH DYNAMIC CURRENCY & DECIMALS
        # -------------------------------------------------------------
        m_conf = risk_results['metrics_by_confidence'][confidence_selection]
        latest_price_native = df['Price'].iloc[-1]
        latest_price_conv = latest_price_native * fx_multiplier
        
        if latest_price_conv < 1.0:
            price_fmt = f"{curr_symbol}{latest_price_conv:,.4f}"
        else:
            price_fmt = f"{curr_symbol}{latest_price_conv:,.2f}"
            
        var_dollar_conv = m_conf['var_dollar'] * fx_multiplier
        es_dollar_conv = m_conf['es_dollar'] * fx_multiplier
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Latest Stock Price ({active_curr})</div>
                <div class="metric-value">{price_fmt}</div>
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
                <div class="metric-title">{int(confidence_selection*100)}% Daily VaR ({active_curr})</div>
                <div class="metric-value">-{curr_symbol}{var_dollar_conv:,.0f}</div>
                <div class="metric-sub">Max Expected Normal Loss</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{int(confidence_selection*100)}% Expected Shortfall ({active_curr})</div>
                <div class="metric-value">-{curr_symbol}{es_dollar_conv:,.0f}</div>
                <div class="metric-sub">Worst-Case Crash Loss</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")

        # Tabbed Dashboard Sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Market Price & Stationarity",
            "📊 Volatility Forecast (GARCH vs EGARCH)",
            "⚠️ Value at Risk (VaR) Envelopes",
            "🔀 VECM Statistical Arbitrage & Pairs Selection",
            "📋 Statistical Diagnostics & ML Accuracy Metrics"
        ])

        with tab1:
            st.subheader(f"Section 1: {primary_ticker} Market Prices & Stationarity Verification")
            
            c1, c2 = st.columns(2)
            with c1:
                adf_color = "#48BB78" if adf_res['is_stationary'] else "#E53E3E"
                st.markdown(f"""
                <div style="background-color: #1A202C; border-left: 5px solid {adf_color}; padding: 12px; border-radius: 6px;">
                    <b>ADF Test Result:</b> Test Stat = <code>{adf_res['test_statistic']:.4f}</code> | 
                    p-value = <code>{adf_res['p_value']:.4e}</code> &rarr; 
                    <span style="color: {adf_color}; font-weight: bold;">{adf_res['conclusion']}</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                kpss_color = "#48BB78" if kpss_res['is_stationary'] else "#E53E3E"
                st.markdown(f"""
                <div style="background-color: #1A202C; border-left: 5px solid {kpss_color}; padding: 12px; border-radius: 6px;">
                    <b>KPSS Test Result:</b> Test Stat = <code>{kpss_res['test_statistic']:.4f}</code> | 
                    p-value = <code>{kpss_res['p_value']:.4f}</code> &rarr; 
                    <span style="color: {kpss_color}; font-weight: bold;">{kpss_res['conclusion']}</span>
                </div>
                """, unsafe_allow_html=True)
                
            fig_price = make_subplots(specs=[[{"secondary_y": True}]])
            fig_price.add_trace(
                go.Scatter(x=df.index, y=df['Price'] * fx_multiplier, name=f"{primary_ticker} Price ({curr_symbol})", line=dict(color="#3182CE", width=2)),
                secondary_y=False
            )
            fig_price.add_trace(
                go.Scatter(x=df.index, y=(df['Price'] * fx_multiplier).rolling(50).mean(), name="50-Day Moving Average", line=dict(color="#ECC94B", width=1.5, dash="dash")),
                secondary_y=False
            )
            fig_price.add_trace(
                go.Bar(x=df.index, y=df['Return_Clean']*100, name="Log Return (%)", marker_color="#38A169", opacity=0.3),
                secondary_y=True
            )
            fig_price.update_layout(
                title=f"{primary_ticker} Stock Price & Stationary Log Returns Trajectory",
                template="plotly_dark",
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_price.update_yaxes(title_text=f"Price ({curr_symbol})", secondary_y=False)
            fig_price.update_yaxes(title_text="Log Return (%)", secondary_y=True)
            st.plotly_chart(fig_price, use_container_width=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">💡 Institutional Analysis: Price Dynamics & Stationarity</div>
                The Augmented Dickey-Fuller (ADF) and KPSS statistical tests rigorously confirm whether return series exhibit mean-reversion. 
                Stationary log-returns eliminate spurious trend regressions and serve as the foundational prerequisite for GARCH time-series volatility modeling.
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.subheader("Section 2: Conditional Volatility Models (Standard GARCH vs EGARCH)")
            
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df.index, y=garch_results['garch_cond_vol']*100, name="Standard GARCH(1,1) Volatility", line=dict(color="#ED8936", width=1.5)))
            fig_vol.add_trace(go.Scatter(x=df.index, y=garch_results['egarch_cond_vol']*100, name="EGARCH(1,1) Asymmetric Volatility", line=dict(color="#E53E3E", width=2)))
            
            fig_vol.update_layout(
                title="Historical Conditional Volatility Comparison (%)",
                template="plotly_dark",
                height=450,
                hovermode="x unified",
                yaxis_title="Annualized Volatility (%)"
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">💡 Institutional Volatility Insight: Asymmetric Leverage Effect</div>
                <b>EGARCH(1,1)</b> captures market asymmetry where negative shocks (market drops) generate higher volatility spikes than positive shocks of identical magnitude.
                The current <b>{forecast_horizon}-day projected annualized volatility is {garch_results['egarch_forecast_vol'][-1]*100:.2f}%</b>.
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.subheader("Section 3: Basel III Value at Risk (VaR) & Expected Shortfall (ES)")
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Scatter(x=df.index, y=df['Return_Clean']*100, name="Clean Log Returns (%)", line=dict(color="#CBD5E0", width=1), opacity=0.5))
            fig_var.add_trace(go.Scatter(x=df.index, y=-garch_results['garch_var_99']*100, name="99% GARCH VaR Envelope", line=dict(color="#DD6B20", width=1.5, dash="dash")))
            fig_var.add_trace(go.Scatter(x=df.index, y=-garch_results['egarch_var_99']*100, name="99% EGARCH VaR Envelope", line=dict(color="#E53E3E", width=2)))
            
            fig_var.update_layout(
                title="99% Confidence Daily Value at Risk (VaR) Loss Tail Envelopes",
                template="plotly_dark",
                height=450,
                hovermode="x unified",
                yaxis_title="Daily Loss (%)"
            )
            st.plotly_chart(fig_var, use_container_width=True)
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                kup_color = "#48BB78" if risk_results['kupiec_99']['passed'] else "#E53E3E"
                st.markdown(f"""
                <div style="background-color: #1A202C; border-left: 5px solid {kup_color}; padding: 14px; border-radius: 6px;">
                    <b>Kupiec POF Backtest (99% VaR):</b><br>
                    Exceedances: <code>{risk_results['kupiec_99']['actual_exceedances']}</code> / Expected: <code>{risk_results['kupiec_99']['expected_exceedances']:.1f}</code><br>
                    p-value = <code>{risk_results['kupiec_99']['p_value']:.4f}</code> &rarr; 
                    <span style="color: {kup_color}; font-weight: bold;">{"Passed Model Backtest" if risk_results['kupiec_99']['passed'] else "Model Exceedance Failure"}</span>
                </div>
                """, unsafe_allow_html=True)
                
            with v_col2:
                chr_color = "#48BB78" if risk_results['christoffersen_99']['passed'] else "#E53E3E"
                st.markdown(f"""
                <div style="background-color: #1A202C; border-left: 5px solid {chr_color}; padding: 14px; border-radius: 6px;">
                    <b>Christoffersen Independence Test:</b><br>
                    LR Stat = <code>{risk_results['christoffersen_99']['test_stat']:.4f}</code> | p-value = <code>{risk_results['christoffersen_99']['p_value']:.4f}</code><br>
                    Conclusion &rarr; <span style="color: {chr_color}; font-weight: bold;">{"No Loss Clustering (Independent)" if risk_results['christoffersen_99']['passed'] else "Loss Clustering Detected"}</span>
                </div>
                """, unsafe_allow_html=True)

        with tab4:
            st.subheader("Section 4: Vector Error Correction Model (VECM) Pairs Trading")
            
            pair_choice = st.selectbox("Select Cointegrated Asset Pair for VECM Spread:", options=["JPM vs BAC", "GOOGL vs MSFT", "C vs BAC"], index=0)
            asset1, asset2 = pair_choice.split(" vs ")
            
            try:
                df1, _ = load_market_data(ticker=asset1, mode="csv")
                df2, _ = load_market_data(ticker=asset2, mode="csv")
                
                df_pairs = pd.DataFrame({
                    asset1: df1['Price'],
                    asset2: df2['Price']
                }).dropna()
                
                vecm_results = fit_vecm_pairs_trading(df_pairs, asset_a=asset1, asset_b=asset2)
                
                st.markdown(f"""
                <div class="signal-box">
                    <b>Active Quantitative Signal ({asset1}/{asset2}):</b> {vecm_results['latest_signal_desc']}<br>
                    <span style="font-size: 0.9rem; color: #A0AEC0;">
                        Current Cointegrating Spread Z-Score: <code>{vecm_results['latest_z_score']:.2f}</code> | Hedge Ratio (Beta): <code>{vecm_results['beta_ratio']:.4f}</code>
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                fig_spread = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=(f"Cointegrating Equilibrium Spread ({asset1} - {vecm_results['beta_ratio']:.2f}*{asset2})", "Spread Z-Score & Threshold Trading Signals"))
                
                fig_spread.add_trace(go.Scatter(x=vecm_results['spread'].index, y=vecm_results['spread']*fx_multiplier, name="Spread Price", line=dict(color="#63B3ED", width=1.5)), row=1, col=1)
                
                fig_spread.add_trace(go.Scatter(x=vecm_results['z_score'].index, y=vecm_results['z_score'], name="Z-Score", line=dict(color="#4FD1C5", width=1.5)), row=2, col=1)
                fig_spread.add_hline(y=2.0, line_dash="dash", line_color="#E53E3E", row=2, col=1)
                fig_spread.add_hline(y=-2.0, line_dash="dash", line_color="#38A169", row=2, col=1)
                fig_spread.add_hline(y=0.0, line_dash="dot", line_color="#A0AEC0", row=2, col=1)
                
                fig_spread.update_layout(template="plotly_dark", height=550, hovermode="x unified")
                st.plotly_chart(fig_spread, use_container_width=True)
                
            except Exception as e_vecm:
                st.error(f"Could not compute VECM statistical arbitrage for {pair_choice}: {e_vecm}")

        with tab5:
            st.subheader("Section 5: Statistical Diagnostics & Machine Learning Evaluation Metrics")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Directional Accuracy", f"{ml_acc['directional_accuracy']*100:.2f}%")
            m2.metric("Out-of-Sample R²", f"{ml_acc['r2_score']:.4f}")
            m3.metric("Root Mean Sq Error (RMSE)", f"{ml_acc['rmse']:.5f}")
            m4.metric("Mean Abs Error (MAE)", f"{ml_acc['mae']:.5f}")
            m5.metric("Mean Abs Pct Error (MAPE)", f"{ml_acc['mape']*100:.2f}%")
            
            st.markdown("---")
            
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"""
                <div style="background-color: #1A202C; padding: 15px; border-radius: 8px;">
                    <h4>SARIMAX Residual Ljung-Box Test</h4>
                    Test Statistic: <code>{lb_res['test_statistic']:.4f}</code><br>
                    p-value: <code>{lb_res['p_value']:.4f}</code><br>
                    <b>Status:</b> {"No Serial Autocorrelation (White Noise Residuals)" if lb_res['no_autocorrelation'] else "Serial Correlation Detected"}
                </div>
                """, unsafe_allow_html=True)
                
            with d2:
                st.markdown(f"""
                <div style="background-color: #1A202C; padding: 15px; border-radius: 8px;">
                    <h4>ARCH-LM Heteroskedasticity Test</h4>
                    Test Statistic: <code>{arch_res['test_statistic']:.4f}</code><br>
                    p-value: <code>{arch_res['p_value']:.4f}</code><br>
                    <b>Status:</b> {"ARCH Effects Filtered Out" if not arch_res['has_arch_effects'] else "Residual ARCH Effects Present"}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📄 Institutional Quantitative Risk Report Generation")
            
            report_docx_bytes = generate_docx_quant_report(
                ticker=primary_ticker,
                source_info=source_info,
                adf_res=adf_res,
                kpss_res=kpss_res,
                garch_results=garch_results,
                risk_results=risk_results,
                ml_acc=ml_acc
            )
            
            st.download_button(
                label="📥 Download Executive Risk Report (.docx)",
                data=report_docx_bytes,
                file_name=f"Quant_Risk_Report_{primary_ticker}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary"
            )

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 0.8rem; padding: 10px;">
            <b>Quantitative Risk Disclaimer:</b> Designed strictly for educational, research, & portfolio stress-testing purposes. 
            Does not constitute formal financial investment advice or SEBI-registered brokerage trade execution recommendations.
        </div>
        """, unsafe_allow_html=True)
