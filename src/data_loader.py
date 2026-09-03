import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TICKER_MAP = {
    "JPM": "JPM_real_data.csv",
    "JPMC": "JPM_real_data.csv",
    "C": "C_real_data.csv",
    "BAC": "BAC_real_data.csv",
    "GOOG": "GOOGL_real_data.csv",
    "GOOGL": "GOOGL_real_data.csv",
    "MSFT": "MSFT_real_data.csv",
    "AMZN": "AMZN_real_data.csv",
    "NVDA": "NVDA_real_data.csv",
    "^VIX": "VIX_real_data.csv",
    "^GSPC": "GSPC_real_data.csv"
}

def detect_date_and_price_cols(df):
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    
    date_candidates = ["date", "timestamp", "time", "datetime", "day"]
    date_col = None
    for c in cols:
        if c.lower() in date_candidates or "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = cols[0]
        
    price_candidates = ["adj close", "close", "price", "val", "value", "last"]
    price_col = None
    for cand in price_candidates:
        for c in cols:
            if c.lower() == cand:
                price_col = c
                break
        if price_col:
            break
            
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            price_col = cols[1] if len(cols) > 1 else cols[0]
            
    return date_col, price_col

def apply_hampel_filter(series, window=5, n_sigmas=3.0):
    series_clean = series.copy()
    rolling_median = series.rolling(window=window, min_periods=1).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    difference = (series - rolling_median).abs()
    
    outlier_idx = difference > threshold
    series_clean[outlier_idx] = rolling_median[outlier_idx]
    return series_clean

def load_market_data(ticker="JPM", mode="hybrid", start_date="2020-01-01", custom_file=None):
    df_raw = None
    source_type = "UNKNOWN"
    status_msg = ""
    
    ticker_upper = str(ticker).upper().strip()
    if ticker_upper == "JPMC":
        ticker_upper = "JPM"
    elif ticker_upper == "GOOGLE":
        ticker_upper = "GOOGL"
        
    clean_ticker_filename = ticker_upper.replace("^", "").replace("=", "_")
    csv_name = TICKER_MAP.get(ticker_upper, f"{clean_ticker_filename}_real_data.csv")
    local_csv_path = os.path.join(DATA_DIR, csv_name)
    
    if mode == "custom" and custom_file is not None:
        try:
            df_raw = pd.read_csv(custom_file)
            source_type = "CUSTOM_CSV"
            status_msg = f"Custom Upload ({getattr(custom_file, 'name', 'Uploaded File')})"
        except Exception as e:
            raise ValueError(f"Failed to read custom CSV file: {e}")

    elif mode == "csv":
        if os.path.exists(local_csv_path):
            df_raw = pd.read_csv(local_csv_path)
            source_type = "LOCAL_CSV"
            status_msg = f"Local CSV Cache ({csv_name})"
        elif os.path.exists(os.path.join(DATA_DIR, "JPM_real_data.csv")):
            df_raw = pd.read_csv(os.path.join(DATA_DIR, "JPM_real_data.csv"))
            source_type = "LOCAL_CSV_FALLBACK"
            status_msg = f"Local CSV Fallback (JPM_real_data.csv for {ticker_upper})"
        else:
            raise FileNotFoundError(f"Local CSV file '{local_csv_path}' not found.")

    elif mode == "live":
        try:
            t = yf.Ticker(ticker_upper)
            df_raw = t.history(start=start_date).reset_index()
            if df_raw.empty:
                df_raw = yf.download(ticker_upper, start=start_date).reset_index()
            source_type = "LIVE_API"
            status_msg = f"Live Yahoo Finance API ({ticker_upper})"
            if not df_raw.empty and len(df_raw) > 10:
                os.makedirs(DATA_DIR, exist_ok=True)
                df_raw.to_csv(local_csv_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Live yfinance download failed for {ticker_upper}: {e}")

    else: # hybrid mode (Default)
        try:
            t = yf.Ticker(ticker_upper)
            df_raw = t.history(start=start_date).reset_index()
            if df_raw.empty:
                df_raw = yf.download(ticker_upper, start=start_date).reset_index()
            if len(df_raw) > 10:
                source_type = "LIVE_API"
                status_msg = f"Live Yahoo Finance API ({ticker_upper})"
                os.makedirs(DATA_DIR, exist_ok=True)
                df_raw.to_csv(local_csv_path, index=False)
            else:
                raise ValueError("Downloaded series too short.")
        except Exception as e:
            if os.path.exists(local_csv_path):
                df_raw = pd.read_csv(local_csv_path)
                source_type = "HYBRID_FALLBACK"
                status_msg = f"Fallback Local CSV ({csv_name})"
            elif os.path.exists(os.path.join(DATA_DIR, "JPM_real_data.csv")):
                df_raw = pd.read_csv(os.path.join(DATA_DIR, "JPM_real_data.csv"))
                source_type = "HYBRID_FALLBACK"
                status_msg = f"Fallback Local CSV (JPM for {ticker_upper})"
            else:
                raise RuntimeError(f"Both live API and local CSV fallback failed for {ticker_upper}: {e}")

    date_col, price_col = detect_date_and_price_cols(df_raw)
    
    # Robust timezone parsing handling mixed offsets
    try:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], utc=True, format='mixed').dt.tz_localize(None)
    except Exception:
        try:
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], utc=True).dt.tz_localize(None)
        except Exception:
            df_raw[date_col] = pd.to_datetime(df_raw[date_col]).dt.tz_localize(None)

    df = df_raw.sort_values(date_col).dropna(subset=[price_col]).copy()
    df.set_index(date_col, inplace=True)
    
    df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=['Price'])
    
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    else:
        df['Volume'] = 0.0
        
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('B')
    df['Price'] = df['Price'].ffill().bfill()
    df['Volume'] = df['Volume'].fillna(0)
    
    df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
    df = df.dropna(subset=['Log_Return'])
    
    df['Return_Clean'] = apply_hampel_filter(df['Log_Return'], window=5, n_sigmas=3.0)
    
    source_info = {
        "ticker": ticker_upper,
        "mode": mode,
        "source_type": source_type,
        "status_msg": status_msg,
        "total_rows": len(df),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "date_col_used": date_col,
        "price_col_used": price_col
    }
    
    return df, source_info
