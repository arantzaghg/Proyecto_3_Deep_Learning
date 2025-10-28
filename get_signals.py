import pandas as pd

import pandas as pd

def signals(data: pd.DataFrame, ticker: str, alpha: float) -> pd.DataFrame:

    """
    Generate trading signals based on future price movements.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing market data with 'Close' prices.
    ticker (str): The ticker symbol of the asset.
    alpha (float): Threshold for generating buy/sell signals.
    
    Returns:
    pd.DataFrame: DataFrame with an additional 'signal' column.
    """

    data = data.copy()
    data['Shift_10'] = data['Close'].shift(-10)

    data["signal"] = 0
    data.loc[data['Close'] * (1+alpha) < data["Shift_10"], "signal"] = 1
    data.loc[data['Close'] * (1-alpha) > data["Shift_10"], "signal"] = 2

    data.drop(columns=["Shift_10"], inplace=True)
    data["signal"] = data["signal"].astype(int)

    return data
