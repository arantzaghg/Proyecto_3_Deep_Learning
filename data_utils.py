import yfinance as yf
import pandas as pd
from indicators import get_indicators
from get_signals import signals
from normalization import get_normal_stats, normalize_data

def get_asset_data(ticker: str) -> pd.DataFrame:

    """
    Fetch historical market data for a given ticker symbol using yfinance.
    
    Parameters:
    ticker (str): The ticker symbol of the asset.
    
    Returns:
    pd.DataFrame: DataFrame containing historical market data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    """

    data = yf.download(ticker, period="15y", interval="1d")
   
    data = data.drop(columns=['Adj Close'], errors='ignore')

    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    Split the data into training, testing, and validation sets.
    
    Parameters:
    data (pd.DataFrame): The complete dataset to be split.
    
    Returns:
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, testing, and validation datasets.
    """

    data = data.copy()
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:train_size + test_size]
    val_data = data.iloc[train_size + test_size:]


    return train_data, test_data, val_data

def preprocess_data(data, ticker, alpha, stage: str, stats: dict | None = None, include_close: bool = True):

    """
    Preprocess the data by adding indicators, generating signals, and normalizing.
    
    Parameters:
    data (pd.DataFrame): The raw market data.
    ticker (str): The ticker symbol of the asset.
    alpha (float): Parameter for signal generation.
    stage (str): Stage of processing - 'train' or 'test'.
    stats (dict | None): Normalization statistics for 'test' stage.
    include_close (bool): Whether to include 'Close' price in normalization.
    
    Returns:
    pd.DataFrame: The preprocessed data.
    """

    data = get_indicators(data)
    data = signals(data, ticker, alpha)

    if stage == "train":
        data, stats = get_normal_stats(data, include_close=include_close)
    else:
        if stats is None:
            raise ValueError("Give stats for normalization")
        data = normalize_data(data, stats, include_close=include_close)
    data.dropna(inplace=True)
    return data, stats

def get_target(data):

    """
    Separate features and target signal from the dataset.
    
    Parameters:
    data (pd.DataFrame): The preprocessed dataset containing features and 'signal' column.

    Returns:
    tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series.
    """
    
    X = data.drop(columns=['Open', 'High', 'Low', 'Volume', 'signal'])
    y = data['signal']
    return X, y

