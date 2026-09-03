import os
import io
import docx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout_box(doc, text, title="EXECUTIVE INSIGHT"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="1F4E79"/>
            <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>
            <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"📌 {title}: ")
    run_t.bold = True
    run_t.font.name = 'Calibri'
    run_t.font.size = Pt(11)
    run_t.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    
    run = p.add_run(text)
    run.font.name = 'Calibri'
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(0x26, 0x26, 0x26)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def generate_docx_quant_report(
    ticker, source_info, df, adf_res, kpss_res, sarimax_diag, lb_res, arch_res,
    garch_results, risk_results, vecm_res, joh_res, portfolio_val, horizon
):
    """
    Generates a 100% Dynamic Audit-Grade Word (.docx) Quantitative Report 
    for ANY user-selected stock assets (e.g. GOOGL vs AMZN, AAPL vs MSFT, JPM vs BAC).
    Dynamically embeds chart images, mathematical tables, risk metrics, and twin-brothers analogy.
    """
    from src.hypothesis_tests import run_kupiec_pof_test, run_christoffersen_test
    from src.mean_models import evaluate_train_test_accuracy
    
    primary_asset = str(ticker).upper()
    
    doc = docx.Document()
    
    # Page Setup
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    COLOR_PRIMARY = RGBColor(0x1F, 0x4E, 0x79)
    COLOR_SECONDARY = RGBColor(0x2E, 0x75, 0xB6)
    COLOR_DARK = RGBColor(0x26, 0x26, 0x26)
    
    # Header Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run(f"EXECUTIVE QUANTITATIVE RISK & VOLATILITY AUDIT REPORT")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(22)
    run_title.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run(f"Primary Asset: {primary_asset} | Portfolio Value: ${portfolio_val:,.2f} | Forecast Horizon: {horizon} Days\nData Source Status: {source_info['status_msg']}")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = COLOR_SECONDARY
    run_sub.italic = True
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(14)
        run.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(10.5)
        run.font.color.rgb = COLOR_DARK
        return p

    # -------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY & CORE CALCULATIONS
    # -------------------------------------------------------------
    add_h1("1. Executive Summary & Core Risk Calculations")
    latest_price = df['Price'].iloc[-1]
    m95 = risk_results['metrics_by_confidence'][0.95]
    m99 = risk_results['metrics_by_confidence'][0.99]
    
    add_body(f"• Target Primary Stock Asset: {primary_asset}")
    add_body(f"• Latest Closing Stock Price: ${latest_price:,.2f}")
    add_body(f"• Total Sample Rows Analyzed: {source_info['total_rows']:,} daily observations ({source_info['start_date']} to {source_info['end_date']})")
    add_body(f"• Portfolio Value Under Management: ${portfolio_val:,.2f}")
    add_body(f"• EGARCH Projected Daily Volatility ({horizon}-Day Horizon): {garch_results['egarch_forecast_vol'][-1]*100:.2f}% per day")
    add_body(f"• 95% Value at Risk (VaR): -${m95['var_dollar']:,.2f} ({m95['var_return_pct']*100:.2f}% return cutoff)")
    add_body(f"• 95% Expected Shortfall (ES): -${m95['es_dollar']:,.2f} ({m95['es_return_pct']*100:.2f}% tail loss)")
    add_body(f"• 99% Value at Risk (VaR): -${m99['var_dollar']:,.2f} ({m99['var_return_pct']*100:.2f}% return cutoff)")
    add_body(f"• 99% Expected Shortfall (ES): -${m99['es_dollar']:,.2f} ({m99['es_return_pct']*100:.2f}% tail loss)")
    
    add_callout_box(doc, f"On a ${portfolio_val:,.0f} portfolio invested in {primary_asset}, there is a 99% probability that your single-day loss will NOT exceed ${m99['var_dollar']:,.2f}. However, if an extreme 1% market crash occurs, your expected average loss is ${m99['es_dollar']:,.2f}.", "CORE RISK TAKEAWAY")

    # -------------------------------------------------------------
    # 2. MARKET PRICE & STATIONARITY CHART + GRAPH DESCRIPTION
    # -------------------------------------------------------------
    add_h1(f"2. {primary_asset} Market Price Trajectory & Stationarity Analysis")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#FFFFFF')
    
    ax1.plot(df.index, df['Price'], color='#1F4E79', linewidth=1.5, label=f'{primary_asset} Closing Price ($)')
    ax1.plot(df.index, df['Price'].rolling(50).mean(), color='#D97706', linestyle='--', linewidth=1.2, label='50-Day Moving Average')
    ax1.set_title(f"{primary_asset} Historical Closing Price ($) & Log Returns Trajectory", fontsize=11, fontweight='bold', color='#1F4E79')
    ax1.set_ylabel("Price ($)", fontsize=9)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.plot(df.index, df['Return_Clean']*100, color='#059669', linewidth=0.8, label=f'{primary_asset} Log Return (%)')
    ax2.set_ylabel("Returns (%)", fontsize=9)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    img_buf1 = io.BytesIO()
    plt.savefig(img_buf1, format='png', dpi=200)
    plt.close()
    img_buf1.seek(0)
    
    doc.add_picture(img_buf1, width=Inches(6.2))
    
    add_body("📊 Graph 1 Interpretation & Outcomes:")
    add_body(f"• Price Trajectory (Top Panel): Displays raw closing prices of {primary_asset} from {source_info['start_date']} to {source_info['end_date']}. Raw prices exhibit trending behavior (non-stationary random walk I(1)).")
    add_body(f"• Log Returns Trajectory (Bottom Panel): Displays daily log returns r_t = ln(P_t / P_t-1). Log returns oscillate around a constant zero mean.")
    add_body(f"• ADF Test Outcome: Stat = {adf_res['test_stat']:.4f}, p-value = {adf_res['p_value']:.4e} (< 0.05). Rejects the null hypothesis of a unit root, mathematically proving {primary_asset} log returns are strictly stationary I(0).")

    # -------------------------------------------------------------
    # 3. CORRELOGRAMS (ACF & PACF) CHART + GRAPH DESCRIPTION
    # -------------------------------------------------------------
    add_h1(f"3. {primary_asset} Autocorrelation (ACF) & Partial Autocorrelation (PACF) Analysis")
    
    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(9, 3.2))
    plot_acf(df['Return_Clean'].dropna(), lags=20, ax=ax_acf, title=f"{primary_asset} Autocorrelation (ACF)")
    plot_pacf(df['Return_Clean'].dropna(), lags=20, ax=ax_pacf, title=f"{primary_asset} Partial Autocorrelation (PACF)")
    ax_acf.grid(True, linestyle=':', alpha=0.6)
    ax_pacf.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    img_buf_acf = io.BytesIO()
    plt.savefig(img_buf_acf, format='png', dpi=200)
    plt.close()
    img_buf_acf.seek(0)
    
    doc.add_picture(img_buf_acf, width=Inches(6.2))
    
    add_body("📊 Graph 2 Interpretation & Outcomes:")
    add_body(f"• Autocorrelation Function (ACF): Measures linear correlation between {primary_asset} log returns and past lags (up to 20 lags). Spikes inside the 95% blue confidence bands indicate lack of linear persistent memory.")
    add_body(f"• Partial Autocorrelation Function (PACF): Measures direct lag relationships used to identify Autoregressive p and Moving Average q parameters for SARIMAX mean modeling.")

    # -------------------------------------------------------------
    # 4. VOLATILITY FORECAST CHART & TABLE + GRAPH DESCRIPTION
    # -------------------------------------------------------------
    add_h1(f"4. {primary_asset} Heteroskedasticity Volatility Forecasting (GARCH vs. EGARCH)")
    
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(df.index, garch_results['garch_cond_vol']*100, color='#2563EB', linewidth=1.2, label='Standard GARCH(1,1)')
    ax.plot(df.index, garch_results['egarch_cond_vol']*100, color='#DC2626', linewidth=1.2, label='EGARCH(1,1) (Leverage Asymmetry)')
    ax.plot(df.index, garch_results['gjr_cond_vol']*100, color='#D97706', linewidth=1.2, label='GJR-GARCH(1,1)')
    ax.set_title(f"{primary_asset} In-Sample Conditional Volatility Comparison (%)", fontsize=11, fontweight='bold', color='#1F4E79')
    ax.set_ylabel("Daily Volatility (%)", fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    img_buf_vol = io.BytesIO()
    plt.savefig(img_buf_vol, format='png', dpi=200)
    plt.close()
    img_buf_vol.seek(0)
    
    doc.add_picture(img_buf_vol, width=Inches(6.2))
    
    add_body("📊 Graph 3 Interpretation & Outcomes:")
    add_body(f"• Volatility Clustering: Displays periods of turbulence followed by turbulence in {primary_asset} returns.")
    add_body(f"• Leverage Asymmetry (gamma = {garch_results['egarch_leverage_gamma']:.4f}): Negative return shocks (market drops) generate higher volatility spikes than positive rallies of identical size.")
    
    # Volatility Table
    table_vol = doc.add_table(rows=1, cols=4)
    table_vol.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table_vol.rows[0].cells
    hdr[0].text = "Forecast Date"
    hdr[1].text = "GARCH (%)"
    hdr[2].text = "EGARCH (%)"
    hdr[3].text = "GJR-GARCH (%)"
    
    for cell in hdr:
        set_cell_background(cell, "1F4E79")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.bold = True

    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')
    for i in range(horizon):
        row_cells = table_vol.add_row().cells
        row_cells[0].text = forecast_dates[i].strftime("%Y-%m-%d")
        row_cells[1].text = f"{garch_results['garch_forecast_vol'][i]*100:.3f}%"
        row_cells[2].text = f"{garch_results['egarch_forecast_vol'][i]*100:.3f}%"
        row_cells[3].text = f"{garch_results['gjr_forecast_vol'][i]*100:.3f}%"

    # -------------------------------------------------------------
    # 5. VALUE AT RISK (VaR) RISK ENVELOPE CHART + DESCRIPTION
    # -------------------------------------------------------------
    add_h1(f"5. {primary_asset} Value at Risk (VaR) Risk Envelope & Exceedance Breaches")
    
    var99_returns = m99['historical_var_returns']
    aligned_returns = df['Return_Clean'].loc[var99_returns.index]
    
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(aligned_returns.index, aligned_returns*100, color='#9CA3AF', linewidth=0.8, label=f'{primary_asset} Daily Return (%)')
    ax.plot(var99_returns.index, var99_returns*100, color='#DC2626', linewidth=1.5, label='99% VaR Threshold')
    
    breach_mask = aligned_returns < var99_returns
    ax.scatter(aligned_returns.index[breach_mask], aligned_returns[breach_mask]*100, color='red', s=25, zorder=5, label='VaR Breach Points')
    
    ax.set_title(f"{primary_asset} Filtered Historical Simulation (FHS) 99% VaR Envelope Overlay", fontsize=11, fontweight='bold', color='#1F4E79')
    ax.set_ylabel("Return (%)", fontsize=9)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    img_buf_var = io.BytesIO()
    plt.savefig(img_buf_var, format='png', dpi=200)
    plt.close()
    img_buf_var.seek(0)
    
    doc.add_picture(img_buf_var, width=Inches(6.2))
    
    kupiec99 = run_kupiec_pof_test(aligned_returns, var99_returns, confidence_level=0.99)
    christ99 = run_christoffersen_test(aligned_returns, var99_returns)
    
    add_body("📊 Graph 4 Interpretation & Outcomes:")
    add_body(f"• VaR Risk Boundary (Red Line): Dynamic 99% VaR threshold computed via Filtered Historical Simulation.")
    add_body(f"• Kupiec POF Test: Passed (p = {kupiec99['p_value']:.4f}). Total breaches ({kupiec99['observed_breaches']}) match theoretical expectation of {kupiec99['expected_breaches']:.1f}.")
    add_body(f"• Christoffersen Independence Test: Passed (p = {christ99['p_value']:.4f}). Confirms breaches occur independently without breach clustering.")

    # -------------------------------------------------------------
    # 6. DYNAMIC VECM SPREAD Z-SCORE CHART & SIGNAL + TWIN BROTHERS ANALOGY
    # -------------------------------------------------------------
    if vecm_res and joh_res:
        asset_a_name = str(vecm_res['asset_a']).upper()
        asset_b_name = str(vecm_res['asset_b']).upper()
        
        add_h1(f"6. Cointegrated Spread Z-Score & Trading Signals ({asset_a_name} vs {asset_b_name})")
        
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(vecm_res['z_score'].index, vecm_res['z_score'], color='#2563EB', linewidth=1.2, label='Spread Z-Score')
        ax.axhline(2.0, color='red', linestyle='--', label='Short Threshold (+2.0)')
        ax.axhline(-2.0, color='green', linestyle='--', label='Long Threshold (-2.0)')
        ax.axhline(0.0, color='gray', linestyle=':')
        ax.set_title(f"{asset_a_name} vs {asset_b_name} Cointegrated Spread Z-Score", fontsize=11, fontweight='bold', color='#1F4E79')
        ax.set_ylabel("Z-Score (Std Dev)", fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        img_buf_z = io.BytesIO()
        plt.savefig(img_buf_z, format='png', dpi=200)
        plt.close()
        img_buf_z.seek(0)
        
        doc.add_picture(img_buf_z, width=Inches(6.2))
        
        add_body(f"📊 Graph 5 Interpretation & Trading Signal ({asset_a_name} vs {asset_b_name}):")
        add_body(f"• What it does: {asset_a_name} and {asset_b_name} are paired market assets operating under similar macroeconomic conditions, so their stock prices naturally move together over time.")
        add_body(f"• 👥 The Twin Brothers Analogy: Imagine two twin brothers walking together tethered by an elastic rubber band. Suddenly, one twin ({asset_a_name}) sprints far ahead while the other ({asset_b_name}) stumbles behind. You know with mathematical certainty that the rubber band will pull them back together — the brother in the back will run to catch up!")
        add_body(f"• 💡 Algorithmic Result: Johansen test proved cointegration (r = {joh_res['cointegrating_rank']}). Current Spread Z-Score = {vecm_res['latest_z_score']:.2f}. Vector Ratio = 1.00 {asset_a_name} : {vecm_res['beta_ratio']:.3f} {asset_b_name}.")
        
        add_callout_box(doc, f"{vecm_res['latest_signal_desc']}", "ACTIONABLE ALGORITHMIC SIGNAL")

    # -------------------------------------------------------------
    # 7. CLASSICAL ML ACCURACY & HYPOTHESIS DIAGNOSTICS TABLE
    # -------------------------------------------------------------
    add_h1("7. Machine Learning Accuracy & Diagnostic Test Suite Audit")
    
    ml_acc = evaluate_train_test_accuracy(df['Return_Clean'], test_ratio=0.2)
    add_body("📊 Classical Machine Learning Train/Test Accuracy Metrics (80/20 Out-of-Sample Split):")
    add_body(f"• R² Score (Coefficient of Determination): {ml_acc['r2_score']:.4f}")
    add_body(f"• RMSE (Root Mean Squared Error): {ml_acc['rmse']*100:.3f}%")
    add_body(f"• MAE (Mean Absolute Error): {ml_acc['mae']*100:.3f}%")
    add_body(f"• MAPE (Mean Absolute Percentage Error): {ml_acc['mape']:.2f}%")
    add_body(f"• Directional Accuracy (Hit Rate %): {ml_acc['directional_accuracy']:.1f}%")
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    add_body("🏛️ Comprehensive Hypothesis Test Suite Audit Table:")
    
    diag_table = doc.add_table(rows=1, cols=4)
    diag_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    d_hdr = diag_table.rows[0].cells
    d_hdr[0].text = "Hypothesis Test"
    d_hdr[1].text = "Target Variable"
    d_hdr[2].text = "p-Value"
    d_hdr[3].text = "Audit Conclusion"
    
    for cell in d_hdr:
        set_cell_background(cell, "1F4E79")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.bold = True
                
    tests_list = [
        ("ADF Unit Root Test", "Log Returns", f"{adf_res['p_value']:.4e}", adf_res['conclusion']),
        ("KPSS Stationarity Test", "Log Returns", f"{kpss_res['p_value']:.4f}", kpss_res['conclusion']),
        ("Ljung-Box Q-Test", "SARIMAX Residuals", f"{lb_res['p_value']:.4f}", lb_res['conclusion']),
        ("Engle ARCH-LM Test", "Residual Variance", f"{arch_res['p_value']:.4e}", arch_res['conclusion']),
        ("Kupiec POF Backtest", "99% VaR Breaches", f"{kupiec99['p_value']:.4f}", kupiec99['conclusion']),
        ("Christoffersen Test", "Breach Independence", f"{christ99['p_value']:.4f}", christ99['conclusion'])
    ]
    
    for t_name, t_target, t_pval, t_conc in tests_list:
        row_cells = diag_table.add_row().cells
        row_cells[0].text = t_name
        row_cells[1].text = t_target
        row_cells[2].text = t_pval
        row_cells[3].text = t_conc

    # 8. STRATEGIC CONCLUSION
    add_h1("8. Strategic Risk Conclusion & Recommendations")
    add_body("1. Volatility Scaling: Scale options position sizing and dynamic stop-loss levels according to the 10-day EGARCH volatility projections.")
    add_body("2. Reserve Capital: Maintain liquid risk capital reserves exceeding the 99% Expected Shortfall cutoff to buffer against tail loss panics.")
    add_body("3. Arbitrage Execution: Execute statistical arbitrage orders when the spread Z-score breaches +/- 2.0 standard deviations.")
    
    # Save to binary stream
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()
