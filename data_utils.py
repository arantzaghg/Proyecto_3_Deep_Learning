import yfinance as yf
import pandas as pd
from indicators import get_indicators
from get_signals import signals
from normalization import get_normal_stats

def get_asset_data(ticker: str) -> pd.DataFrame:

    data = yf.download(ticker, period="15y", interval="1d")
   
    data = data.drop(columns=['Adj Close'], errors='ignore')

    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    data = data.copy()
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:train_size + test_size]
    val_data = data.iloc[train_size + test_size:]


    return train_data, test_data, val_data

def preprocess_data(data):
    data = get_indicators(data)
    data = signals(data)
    data, stats = get_normal_stats(data)
    data.dropna(inplace=True)
    return data

def get_target(data):
    X = data.drop(columns=['Open', 'High', 'Low', 'Volume', 'signal'])
    y = data['signal']
    return X, y






