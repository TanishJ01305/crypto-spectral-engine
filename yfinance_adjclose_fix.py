# FIXED: yfinance Adj Close KeyError workaround
import yfinance as yf
import pandas as pd

def load_yfinance(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    # Check for 'Adj Close', fallback to 'Close'
    if "Adj Close" in data.columns:
        price_df = data["Adj Close"]
    else:
        price_df = data["Close"]
    return price_df

# Example usage:
if __name__ == "__main__":
    asset_list = ["BTC-USD", "ETH-USD"]
    start = "2023-01-01"
    end = "2023-12-31"
    price_df = load_yfinance(asset_list, start, end)
    print(price_df.head())
