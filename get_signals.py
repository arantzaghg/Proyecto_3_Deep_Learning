import pandas as pd

def signals(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    data['pct_change'] = data['Close'].pct_change(fill_method=None).dropna()

    data["signal"] = 0
    data.loc[data["pct_change"] > 0.01, "signal"] = 1
    data.loc[data["pct_change"] < -0.01, "signal"] = -1


    data["signal"] = data["signal"].astype(int)

    return data