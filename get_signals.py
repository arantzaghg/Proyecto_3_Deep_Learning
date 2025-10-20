import pandas as pd

def get_signals(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    data['pct_change'] = data['close'].pct_change(fill_method=None).dropna()
    data["signal"] = 0
    data.loc[data["pct_change"] > 0.012, "signal"] = 1
    data.loc[data["pct_change"] < -0.012, "signal"] = -1
    data["signal"] = data["signal"].astype(int)

    return data