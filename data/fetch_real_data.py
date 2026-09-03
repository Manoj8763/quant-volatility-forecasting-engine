import os
import yfinance as yf
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

TICKERS = {
    "JPM": "JPM_real_data.csv",
    "C": "C_real_data.csv",
    "BAC": "BAC_real_data.csv",
    "^VIX": "VIX_real_data.csv",
    "^GSPC": "GSPC_real_data.csv"
}

def fetch_and_cache_data(start_date="2020-01-01"):
    """
    Downloads real-world historical daily market datasets via yfinance 
    and caches them as clean CSV files in the data directory.
    """
    print("=" * 60)
    print("Fetching Real-World Financial Market Data via yfinance...")
    print("=" * 60)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for symbol, filename in TICKERS.items():
        csv_path = os.path.join(DATA_DIR, filename)
        print(f"--> Downloading ticker '{symbol}' from {start_date}...")
        try:
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(start=start_date)
            if df.empty:
                print(f"[Warning] yfinance returned empty DataFrame for {symbol}. Attempting fallback download...")
                df = yf.download(symbol, start=start_date)
            
            df = df.reset_index()
            # Standardize date column
            date_col = 'Date' if 'Date' in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            
            # Save cleanly
            df.to_csv(csv_path, index=False)
            print(f" [SUCCESS] Saved {len(df)} daily rows to '{csv_path}'")
        except Exception as e:
            print(f" [ERROR] Downloading {symbol}: {e}")
            
    print("\nReal market data caching complete!")

if __name__ == "__main__":
    fetch_and_cache_data()
