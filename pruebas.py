import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_utils import get_asset_data, preprocess_data
from indicators import get_indicators
from get_signals import signals

def pruebas():
    
    ticker = "DIS"
    data = get_asset_data(ticker)
    data = get_indicators(data)
    

    r7 = data["Close"].pct_change(5).shift(-5)  # retorno futuro 7D (coherente con tu label)
    alpha_sugerido = np.nanpercentile(np.abs(r7), 33)  # ~ balance 0/1/2
    print(alpha_sugerido)



if __name__ == "__main__":
    pruebas()


    
    