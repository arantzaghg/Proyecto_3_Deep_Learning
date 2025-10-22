import pandas as pd

def signals(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    data['pct_change'] = data['Close'].pct_change(periods=5)
    data.dropna(inplace=True)


    data["signal"] = 0
    data.loc[data["pct_change"] > 0.03, "signal"] = 1
    data.loc[data["pct_change"] < -0.03, "signal"] = 2

    data.drop(columns=["pct_change"], inplace=True)
    data["signal"] = data["signal"].astype(int)

    return data