import os
import pytest
import pandas as pd
from src.data_loader import load_market_data

def test_load_market_data_csv_mode():
    df, info = load_market_data(ticker="JPM", mode="csv")
    assert isinstance(df, pd.DataFrame)
    assert "Price" in df.columns
    assert "Log_Return" in df.columns
    assert "Return_Clean" in df.columns
    assert len(df) > 100
    assert info["mode"] == "csv"
    assert info["source_type"] == "LOCAL_CSV"

def test_load_market_data_hybrid_mode():
    df, info = load_market_data(ticker="JPM", mode="hybrid")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    assert info["source_type"] in ["LIVE_API", "HYBRID_FALLBACK"]
